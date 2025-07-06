import os
import json
import math
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcde
import torchsde
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import GroupShuffleSplit

# ====================== 1. CONFIGURATION ======================
CONFIG = {
    # --- File and Path ---
    "RAW_DATA_PATH": '/Users/gongzihui/Documents/Linear Noise SDE/trainingData.csv',
    "BASE_OUTPUT_DIR": './multi_task_results_v1',
    "PREDICTIONS_FILE": "test_predictions_v4.csv",

    # --- Data Preprocessing ---
    "TEST_SIZE": 0.1,
    "WAP_VALUE_REPLACE": 100,
    "WAP_NORMALIZATION_OFFSET": 105.0,

    # --- Data & Model Structure ---
    "IMG_HEIGHT": 20,
    "IMG_WIDTH": 26,
    "WINDOW_LEN":  4,
    "WINDOW_STRIDE": 2,

    "CNN_BLOCKS": 2,

    "TRANSFORMER_D_MODEL": 512,
    "TRANSFORMER_NHEAD": 8,
    "TRANSFORMER_NUM_LAYERS": 4,
    "TRANSFORMER_DIM_FEEDFORWARD": 512,

    "SDE_HIDDEN_DIM": 2500,
    "HIERARCHICAL_FEAT_DIM": 300,

    # --- Regularization & Training Strategy ---
    "DROPOUT_RATE": 0.55,
    "AUG_NOISE_STD": 0.05,
    "AUG_DROPOUT_RATE": 0.1,

    "EARLY_STOPPING_PATIENCE": 8,

    # --- Auto-determined at runtime ---
    "NUM_BUILDINGS": None,
    "NUM_FLOORS": None,

    # --- Training ---
    "BATCH_SIZE": 256,
    "EVAL_BATCH_SIZE": 64,

    "LR": 8e-4,
    "EPOCHS": 100,

    "VAL_RATIO": 0.15,
    "WEIGHT_DECAY": 5e-4,

    "CLIP_GRAD_NORM": 1.0,

    # --- Multi-task Loss Weights ---
    "LOSS_WEIGHT_BUILDING": 1.0,
    "LOSS_WEIGHT_FLOOR": 1.0,
    "LOSS_WEIGHT_COORDS": 15.0,

    "SEED": 3047,
}

# ====================== 2. DATASET & AUGMENTATION ======================
class TrajectoryWindowDataset(Dataset):
    def __init__(self, df, config, building_id_map, floor_id_map):
        self.samples = []
        self.wap_cols = [col for col in df.columns if col.startswith('WAP')]
        h, w = config["IMG_HEIGHT"], config["IMG_WIDTH"]
        self.building_id_map = building_id_map
        self.floor_id_map = floor_id_map
        for pid, group in df.groupby('PHONEID'):
            group = group.sort_values('TIMESTAMP').reset_index(drop=True)
            wap_vals = group[self.wap_cols].values.astype(np.float32)
            X_full_img = wap_vals.reshape(-1, 1, h, w)
            Y_coords_full = group[['LONGITUDE', 'LATITUDE']].values.astype(np.float32)
            Y_building_full = group['BUILDINGID'].map(self.building_id_map).values.astype(np.int64)
            Y_floor_full = group['FLOOR'].map(self.floor_id_map).values.astype(np.int64)
            for start in range(0, len(group) - config["WINDOW_LEN"] + 1, config["WINDOW_STRIDE"]):
                end = start + config["WINDOW_LEN"]
                self.samples.append((
                    X_full_img[start:end], Y_coords_full[end - 1],
                    Y_building_full[end - 1], Y_floor_full[end - 1]
                ))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        x, y_c, y_b, y_f = self.samples[idx]
        return (torch.tensor(x), torch.tensor(y_c), torch.tensor(y_b), torch.tensor(y_f))

def augment_wap_signals(x, config):
    if config["AUG_NOISE_STD"] > 0:
        x = x + torch.randn_like(x) * config["AUG_NOISE_STD"]
    if config["AUG_DROPOUT_RATE"] > 0:
        x = x * torch.bernoulli(torch.full_like(x, 1 - config["AUG_DROPOUT_RATE"]))
    return torch.clamp(x, 0, 1)

# ====================== 3. MODEL DEFINITION ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class FeatureExtractor_v4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn_pre = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.resnet_blocks = nn.Sequential(
            *[ResNetBlock(64, 64) for _ in range(config["CNN_BLOCKS"])]
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, config["IMG_HEIGHT"], config["IMG_WIDTH"])
            cnn_out_dim = self.resnet_blocks(self.cnn_pre(dummy_input)).view(1, -1).shape[1]
        self.feature_proj = nn.Linear(cnn_out_dim, config["TRANSFORMER_D_MODEL"])
        self.pos_encoder = PositionalEncoding(config["TRANSFORMER_D_MODEL"])
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config["TRANSFORMER_D_MODEL"],
            nhead=config["TRANSFORMER_NHEAD"],
            dim_feedforward=config["TRANSFORMER_DIM_FEEDFORWARD"],
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=config["TRANSFORMER_NUM_LAYERS"])

    def forward(self, x):
        b, t, c, h, w = x.shape
        x_reshaped = x.view(b * t, c, h, w)
        cnn_out = self.resnet_blocks(self.cnn_pre(x_reshaped))
        cnn_out_flat = cnn_out.view(b, t, -1)
        projected_feat = self.feature_proj(cnn_out_flat)
        pos_encoded_feat = self.pos_encoder(projected_feat)
        transformer_out = self.transformer_encoder(pos_encoded_feat)
        return transformer_out

class NeuralSDEFunc_v2(nn.Module):
    sde_type, noise_type = "ito", "diagonal"
    def __init__(self, feat_dim, hidden_dim):
        super().__init__()
        self.linear_y = nn.Linear(hidden_dim, hidden_dim)
        self.linear_z = nn.Linear(feat_dim, hidden_dim)
        self.f_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.g_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.Z = None
    def set_Z(self, coeffs, times): self.Z = torchcde.CubicSpline(coeffs, times)
    def f(self, t, y): return self.f_net(self.linear_y(y) + self.linear_z(self.Z.evaluate(t)))
    def g(self, t, y): return self.g_net(y)

class CNN_SDE_Model_v4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_extractor = FeatureExtractor_v4(config)
        self.func = NeuralSDEFunc_v2(config["TRANSFORMER_D_MODEL"], config["SDE_HIDDEN_DIM"])
        self.init_proj = nn.Linear(config["TRANSFORMER_D_MODEL"], config["SDE_HIDDEN_DIM"])
        sde_hidden, hier_feat_dim = config["SDE_HIDDEN_DIM"], config["HIERARCHICAL_FEAT_DIM"]
        self.building_head = nn.Linear(sde_hidden, config["NUM_BUILDINGS"])
        self.building_feat_proj = nn.Linear(config["NUM_BUILDINGS"], hier_feat_dim)
        self.floor_head = nn.Linear(sde_hidden + hier_feat_dim, config["NUM_FLOORS"])
        self.floor_feat_proj = nn.Linear(config["NUM_FLOORS"], hier_feat_dim)
        self.coord_head = nn.Sequential(
            nn.Linear(sde_hidden + hier_feat_dim, sde_hidden // 2), nn.ReLU(),
            nn.Dropout(config["DROPOUT_RATE"]), nn.Linear(sde_hidden // 2, 2)
        )
    def forward(self, x):
        z_sequence = self.feature_extractor(x)
        times = torch.linspace(0, 1, z_sequence.size(1), device=x.device)
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(z_sequence, times)
        self.func.set_Z(coeffs, times)
        y0 = self.init_proj(z_sequence[:, 0, :])
        dt = 1.0 / (z_sequence.size(1) - 1)
        sde_solution = torchsde.sdeint(self.func, y0, times, dt=dt, method='milstein').permute(1, 0, 2)
        final_sde_state = sde_solution[:, -1, :]
        pred_b = self.building_head(final_sde_state)
        building_features = self.building_feat_proj(F.softmax(pred_b, dim=1).detach())
        floor_input = torch.cat([final_sde_state, building_features], dim=1)
        pred_f = self.floor_head(floor_input)
        floor_features = self.floor_feat_proj(F.softmax(pred_f, dim=1).detach())
        coord_input = torch.cat([final_sde_state, floor_features], dim=1)
        pred_c = self.coord_head(coord_input)
        return pred_b, pred_f, pred_c

# ====================== 4. UTILITIES ======================
def euclidean_loss(pred, target):
    return torch.sqrt(torch.sum((pred - target) ** 2, dim=-1)).mean()

# ====================== 5. TRAINER ======================
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.model, self.optimizer, self.scheduler = None, None, None
        self.global_label_stats, self.building_id_map, self.floor_id_map = {}, {}, {}
        self.building_id_reverse_map, self.floor_id_reverse_map = {}, {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.config["BASE_OUTPUT_DIR"], f"run_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        self._setup_printing()
        print(f"All results will be saved to: {self.output_dir}")
        print(f"Current CONFIG: {json.dumps(self.config, indent=2)}")

    def _get_device(self) -> torch.device:
        device = torch.device('cpu')
        print(f"Using device: {device} (set to CPU for compatibility)")
        return device

    def _setup_printing(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler()])

    def _preprocess_data(self, df):
        wap_cols = [col for col in df.columns if col.startswith('WAP')]
        df[wap_cols] = df[wap_cols].replace({self.config["WAP_VALUE_REPLACE"]: -self.config["WAP_NORMALIZATION_OFFSET"]})
        df[wap_cols] = (df[wap_cols] + self.config["WAP_NORMALIZATION_OFFSET"]) / self.config["WAP_NORMALIZATION_OFFSET"]
        return df

    def _normalize_coordinates(self, df_train, df_test):
        logging.info("Standardizing coordinates...")
        for col in ['LONGITUDE', 'LATITUDE']:
            mean, std = df_train[col].mean(), df_train[col].std()
            self.global_label_stats[col] = {'mean': mean, 'std': std}
            df_train.loc[:, col] = (df_train[col] - mean) / (std + 1e-8)
            df_test.loc[:, col] = (df_test[col] - mean) / (std + 1e-8)

    def _create_id_maps(self, df_raw):
        all_b_ids, all_f_ids = sorted(df_raw['BUILDINGID'].unique()), sorted(df_raw['FLOOR'].unique())
        self.building_id_map = {id: i for i, id in enumerate(all_b_ids)}
        self.floor_id_map = {id: i for i, id in enumerate(all_f_ids)}
        self.building_id_reverse_map = {i: id for id, i in self.building_id_map.items()}
        self.floor_id_reverse_map = {i: id for id, i in self.floor_id_map.items()}
        self.config["NUM_BUILDINGS"], self.config["NUM_FLOORS"] = len(all_b_ids), len(all_f_ids)

    def load_and_split_data(self):
        logging.info("======== Step 1: Loading and Splitting Data ========")
        df_raw = pd.read_csv(self.config["RAW_DATA_PATH"])
        logging.info("Currently using 100% full data for training.")
        df_raw = self._preprocess_data(df_raw)
        splitter = GroupShuffleSplit(n_splits=1, test_size=self.config['TEST_SIZE'], random_state=self.config['SEED'])
        train_indices, test_indices = next(splitter.split(df_raw, groups=df_raw['USERID']))
        train_df, test_df = df_raw.iloc[train_indices].copy(), df_raw.iloc[test_indices].copy()
        self._create_id_maps(df_raw)
        self._normalize_coordinates(train_df, test_df)
        train_ds = TrajectoryWindowDataset(train_df, self.config, self.building_id_map, self.floor_id_map)
        test_ds = TrajectoryWindowDataset(test_df, self.config, self.building_id_map, self.floor_id_map)
        val_size = int(len(train_ds) * self.config["VAL_RATIO"])
        train_size = len(train_ds) - val_size
        train_set, val_set = random_split(train_ds, [train_size, val_size], generator=torch.Generator().manual_seed(self.config["SEED"]))
        dl_args = {'num_workers': os.cpu_count() // 2, 'pin_memory': True} if self.device.type != 'cpu' else {}
        train_loader = DataLoader(train_set, self.config["BATCH_SIZE"], shuffle=True, drop_last=True, **dl_args)
        val_loader = DataLoader(val_set, self.config["EVAL_BATCH_SIZE"], shuffle=False, **dl_args)
        test_loader = DataLoader(test_ds, self.config["EVAL_BATCH_SIZE"], shuffle=False, **dl_args)
        return train_loader, val_loader, test_loader

    def _run_train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        ce_loss = nn.CrossEntropyLoss()
        for X, Y_c, Y_b, Y_f in loader:
            X, Y_c, Y_b, Y_f = X.to(self.device), Y_c.to(self.device), Y_b.to(self.device), Y_f.to(self.device)
            X = augment_wap_signals(X, self.config)
            p_b, p_f, p_c = self.model(X)
            loss = (self.config["LOSS_WEIGHT_BUILDING"] * ce_loss(p_b, Y_b) +
                    self.config["LOSS_WEIGHT_FLOOR"] * ce_loss(p_f, Y_f) +
                    self.config["LOSS_WEIGHT_COORDS"] * euclidean_loss(p_c, Y_c))
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["CLIP_GRAD_NORM"])
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _evaluate_on_validation(self, loader):
        self.model.eval()
        p_c, t_c, p_b, t_b, p_f, t_f = [], [], [], [], [], []
        with torch.no_grad():
            for X, Y_c, Y_b, Y_f in loader:
                X, Y_c, Y_b, Y_f = X.to(self.device), Y_c.to(self.device), Y_b.to(self.device), Y_f.to(self.device)
                pred_b, pred_f, pred_c = self.model(X)
                p_c.append(pred_c.cpu()); t_c.append(Y_c.cpu())
                p_b.append(pred_b.argmax(1).cpu()); t_b.append(Y_b.cpu())
                p_f.append(pred_f.argmax(1).cpu()); t_f.append(Y_f.cpu())
        preds_c, trues_c = torch.cat(p_c), torch.cat(t_c)
        preds_b, trues_b = torch.cat(p_b), torch.cat(t_b)
        preds_f, trues_f = torch.cat(p_f), torch.cat(t_f)
        mean_long, std_long = self.global_label_stats['LONGITUDE']['mean'], self.global_label_stats['LONGITUDE']['std']
        mean_lat, std_lat = self.global_label_stats['LATITUDE']['mean'], self.global_label_stats['LATITUDE']['std']
        preds_denorm_long = preds_c[:, 0] * std_long + mean_long; preds_denorm_lat = preds_c[:, 1] * std_lat + mean_lat
        trues_denorm_long = trues_c[:, 0] * std_long + mean_long; trues_denorm_lat = trues_c[:, 1] * std_lat + mean_lat
        coord_error_m = torch.sqrt((preds_denorm_long - trues_denorm_long)**2 + (preds_denorm_lat - trues_denorm_lat)**2).mean().item()
        return {
            "coord_err_m": coord_error_m,
            "building_acc": (preds_b == trues_b).float().mean().item(),
            "floor_acc": (preds_f == trues_f).float().mean().item()
        }

    def train_model(self, train_loader, val_loader):
        print(f"\n======== Training (V4 Final Optimized Model) ========")
        self.model = CNN_SDE_Model_v4(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["LR"], weight_decay=self.config["WEIGHT_DECAY"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=3)
        best_val_coord_error, patience_counter = float('inf'), 0
        best_model_path = os.path.join(self.output_dir, 'best_model.pt')

        for epoch in range(1, self.config["EPOCHS"] + 1):
            train_loss = self._run_train_epoch(train_loader)
            val_metrics = self._evaluate_on_validation(val_loader)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | LR: {current_lr:.1e} | --- Val Metrics --- | "
                  f"Coord Error: {val_metrics['coord_err_m']:.3f}m | Building Acc: {val_metrics['building_acc']:.2%} | Floor Acc: {val_metrics['floor_acc']:.2%}")
            self.scheduler.step(val_metrics["coord_err_m"])
            if val_metrics["coord_err_m"] < best_val_coord_error:
                best_val_coord_error = val_metrics["coord_err_m"]
                torch.save(self.model.state_dict(), best_model_path)
                print(f"    -> New best validation error: {best_val_coord_error:.3f}m. Model saved. ★")
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= self.config["EARLY_STOPPING_PATIENCE"]:
                print(f"\n[!] Early stopping triggered after {self.config['EARLY_STOPPING_PATIENCE']} epochs with no improvement.")
                break
        print(f"======== Training Finished (Completed {epoch} epochs) ========")

    def run(self):
        try:
            train_loader, val_loader, test_loader = self.load_and_split_data()
            self.train_model(train_loader, val_loader)
            best_model_path = os.path.join(self.output_dir, 'best_model.pt')
            if os.path.exists(best_model_path):
                print(f"\nLoading best model '{best_model_path}' for final evaluation...")
                self.model.load_state_dict(torch.load(best_model_path, map_location=self.device, weights_only=True))
            else:
                print("\nWarning: Best model not found, using current model state for evaluation.")
            final_metrics = self.evaluate_and_save_results(test_loader)
            print(f"\n\n{'='*30} Final Summary Report {'='*30}\n"
                  f"  Mean Localization Error: {final_metrics['coord_err_m']:.4f} m\n"
                  f"  Building Classification Accuracy: {final_metrics['building_acc']:.4%}\n"
                  f"  Floor Classification Accuracy: {final_metrics['floor_acc']:.4%}\n{'='*72}")
        except KeyboardInterrupt:
            print("\n\n[!] Training interrupted by user.")
        except Exception as e:
            logging.error(f"Fatal error during execution: {e}", exc_info=True)

    def evaluate_and_save_results(self, loader):
        metrics = self._evaluate_on_validation(loader)
        self.model.eval()
        p_c, t_c, p_b, t_b, p_f, t_f = [], [], [], [], [], []
        with torch.no_grad():
            for X, Y_c, Y_b, Y_f in loader:
                X = X.to(self.device)
                pred_b, pred_f, pred_c = self.model(X)
                p_c.append(pred_c.cpu()); t_c.append(Y_c.cpu())
                p_b.append(pred_b.argmax(1).cpu()); t_b.append(Y_b.cpu())
                p_f.append(pred_f.argmax(1).cpu()); t_f.append(Y_f.cpu())
        preds_c, trues_c = torch.cat(p_c), torch.cat(t_c)
        preds_b, trues_b = torch.cat(p_b), torch.cat(t_b)
        preds_f, trues_f = torch.cat(p_f), torch.cat(t_f)
        mean_long, std_long = self.global_label_stats['LONGITUDE']['mean'], self.global_label_stats['LONGITUDE']['std']
        mean_lat, std_lat = self.global_label_stats['LATITUDE']['mean'], self.global_label_stats['LATITUDE']['std']
        preds_denorm_long = preds_c[:, 0] * std_long + mean_long; preds_denorm_lat = preds_c[:, 1] * std_lat + mean_lat
        trues_denorm_long = trues_c[:, 0] * std_long + mean_long; trues_denorm_lat = trues_c[:, 1] * std_lat + mean_lat
        distances = torch.sqrt((preds_denorm_long - trues_denorm_long)**2 + (preds_denorm_lat - trues_denorm_lat)**2)
        predictions_path = os.path.join(self.output_dir, self.config["PREDICTIONS_FILE"])
        pd.DataFrame({
            'True_Longitude': trues_denorm_long.numpy(), 'Pred_Longitude': preds_denorm_long.numpy(),
            'True_Latitude': trues_denorm_lat.numpy(), 'Pred_Latitude': preds_denorm_lat.numpy(),
            'True_BuildingID': [self.building_id_reverse_map[i.item()] for i in trues_b],
            'Pred_BuildingID': [self.building_id_reverse_map[i.item()] for i in preds_b],
            'True_Floor': [self.floor_id_reverse_map[i.item()] for i in trues_f],
            'Pred_Floor': [self.floor_id_reverse_map[i.item()] for i in preds_f],
            'Error_Meters': distances.numpy()
        }).to_csv(predictions_path, index=False)
        print(f"Prediction details saved to: {predictions_path}")
        return metrics

# ====================== 6. MAIN ENTRY ======================
if __name__ == '__main__':
    torch.manual_seed(CONFIG["SEED"])
    np.random.seed(CONFIG["SEED"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG["SEED"])
    trainer = ModelTrainer(CONFIG)
    trainer.run()
