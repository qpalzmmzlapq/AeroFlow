"""
Полное обучение AeroNet на DeepCFD.
Запуск: python train.py
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pickle
from pathlib import Path
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from model import AeroNet

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CKPT_DIR = HERE / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

# ---- Границы клипа ----
with open(DATA_DIR / "clip_bounds.json") as f:
    _b = json.load(f)
Y_BOUNDS = {k: tuple(v) for k, v in _b.items()}
print(f"Clip bounds: {Y_BOUNDS}")

# ---- Config ----
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-5
N_EPOCHS=300
VAL_SPLIT = 0.1
SEED = 42
DEVICE = "cpu"

def preprocess_one(x_np, y_np):
    x = torch.from_numpy(x_np).float().unsqueeze(0)
    y = torch.from_numpy(y_np).float().unsqueeze(0)
    x = F.interpolate(x, size=(176, 80), mode="bilinear", align_corners=False).squeeze(0)
    y = F.interpolate(y, size=(176, 80), mode="bilinear", align_corners=False).squeeze(0)

    x[0] = torch.clamp(x[0], -1.0, 1.0)
    x[1] = torch.round(x[1])

    mask_flow = (x[1] != 0).float()
    y[0] = y[0] * mask_flow
    y[1] = y[1] * mask_flow

    y[0] = torch.clamp(y[0], Y_BOUNDS["ux"][0], Y_BOUNDS["ux"][1])
    y[1] = torch.clamp(y[1], Y_BOUNDS["uy"][0], Y_BOUNDS["uy"][1])
    y[2] = torch.clamp(y[2], Y_BOUNDS["p"][0],  Y_BOUNDS["p"][1])

    return x, y


def preprocess_all(X, Y):
    n = len(X)
    x_list, y_list = [], []
    for i in range(n):
        x, y = preprocess_one(X[i], Y[i])
        x_list.append(x)
        y_list.append(y)
    return torch.stack(x_list), torch.stack(y_list)


def compute_stats(X, Y):
    x_mean = X.mean(dim=(0, 2, 3), keepdim=True)
    x_std = X.std(dim=(0, 2, 3), keepdim=True) + 1e-8
    y_mean = Y.mean(dim=(0, 2, 3), keepdim=True)
    y_std = Y.std(dim=(0, 2, 3), keepdim=True) + 1e-8
    return x_mean, x_std, y_mean, y_std


def normalize(X, Y, stats):
    x_mean, x_std, y_mean, y_std = stats
    X = X.clone()
    X[:, 0:1] = (X[:, 0:1] - x_mean[:, 0:1]) / x_std[:, 0:1]
    X[:, 2:3] = (X[:, 2:3] - x_mean[:, 2:3]) / x_std[:, 2:3]
    # Нормируем Y, но НЕ трогаем пиксели внутри тела
    Y_norm = (Y - y_mean) / y_std
    mask = (Y[:, 0:1] == 0) & (Y[:, 1:2] == 0)   # и Ux, и Uy равны 0
    Y_norm = Y_norm * (~mask).float()            # обнуляем внутри тела
    return X, Y_norm


class TensorDataset(Dataset):
    def __init__(self, X, Y, indices):
        self.X = X
        self.Y = Y
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        return self.X[idx], self.Y[idx]


def rel_l2(pred, target, eps=1e-8, weights=(3.0, 1.0, 1.0)):
    """Weighted rel_L2: Ux в 3 раза важнее Uy и P."""
    total = 0.0
    for c, w in enumerate(weights):
        diff = (pred[:, c] - target[:, c]).reshape(pred.shape[0], -1)
        tgt = target[:, c].reshape(target.shape[0], -1)
        rel = torch.norm(diff, dim=1) / (torch.norm(tgt, dim=1) + eps)
        total = total + w * rel.mean()
    return total / sum(weights)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    per_ch = None
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = rel_l2(pred, yb)
        total += loss.item() * xb.size(0)
        n += xb.size(0)
        diff = (pred - yb).reshape(pred.shape[0], pred.shape[1], -1)
        tgt = yb.reshape(yb.shape[0], yb.shape[1], -1)
        ch = (torch.norm(diff, dim=2) / (torch.norm(tgt, dim=2) + 1e-8)).mean(dim=0)
        per_ch = ch.cpu().numpy() if per_ch is None else per_ch + ch.cpu().numpy()
    return total / n, per_ch / len(loader)


def main():
    print(f"Device: {DEVICE}")

    print(f"\nЗагрузка...")
    X_raw = np.asarray(pickle.load(open(DATA_DIR / "dataX.pkl", "rb")), dtype=np.float32)
    Y_raw = np.asarray(pickle.load(open(DATA_DIR / "dataY.pkl", "rb")), dtype=np.float32)
    print(f"X: {X_raw.shape}, Y: {Y_raw.shape}")

    print(f"\nПрепроцессинг всех примеров...")
    t0 = time.time()
    X, Y = preprocess_all(X_raw, Y_raw)
    print(f"  Готово за {time.time()-t0:.1f}s. Форма: {tuple(X.shape)}")

    N = len(X)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    n_val = int(N * VAL_SPLIT)
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    stats = compute_stats(X[train_idx], Y[train_idx])
    X_n, Y_n = normalize(X, Y, stats)
    print(f"Нормировка: mean/std по train")

    train_ds = TensorDataset(X_n, Y_n, train_idx)
    val_ds = TensorDataset(X_n, Y_n, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=False)

    model = AeroNet(in_c=3, num_c=3, base=16).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: AeroNet, params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    best_val = float("inf")
    history = []
    print(f"\n{'Ep':>4} | {'train':>9} | {'val':>9} | {'Ux':>7} | "
          f"{'Uy':>7} | {'P':>7} | {'lr':>8} | {'time':>6}")
    print("-" * 80)

    for ep in range(1, N_EPOCHS + 1):
        t0 = time.time()
        model.train()
        total, nb = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = rel_l2(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * xb.size(0)
            nb += xb.size(0)
        train_loss = total / nb

        val_loss, per_ch = validate(model, val_loader, DEVICE)
        sched.step()
        dt = time.time() - t0

        print(f"{ep:>4} | {train_loss:>9.5f} | {val_loss:>9.5f} | "
              f"{per_ch[0]:>7.4f} | {per_ch[1]:>7.4f} | {per_ch[2]:>7.4f} | "
              f"{opt.param_groups[0]['lr']:>8.2e} | {dt:>5.1f}s")

        history.append({"epoch": ep, "train": train_loss, "val": val_loss,
                       "per_channel": per_ch.tolist(), "time": dt})

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "val_loss": val_loss,
                "stats": {"x_mean": stats[0], "x_std": stats[1],
                         "y_mean": stats[2], "y_std": stats[3]},
                "y_bounds": Y_BOUNDS,
            }, CKPT_DIR / "best.pt")

    print(f"\n✓ Лучший val loss: {best_val:.5f}")
    print(f"  Чекпоинт: {CKPT_DIR / 'best.pt'}")

    with open(CKPT_DIR / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  История: {CKPT_DIR / 'history.json'}")


if __name__ == "__main__":
    main()