"""
Визуализация финальной AeroNet (229 эпох).
Показывает target vs pred vs error для Ux, Uy, P
на трёх примерах: лучший, средний, худший.

Запуск: python visualize.py
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pickle, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import AeroNet

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CKPT = HERE / "checkpoints" / "best.pt"

with open(DATA_DIR / "clip_bounds.json") as f:
    Y_BOUNDS = {k: tuple(v) for k, v in json.load(f).items()}


def preprocess_one(x_np, y_np):
    x = torch.from_numpy(x_np).float().unsqueeze(0)
    y = torch.from_numpy(y_np).float().unsqueeze(0)
    x = F.interpolate(x, size=(176, 80), mode="bilinear", align_corners=False).squeeze(0)
    y = F.interpolate(y, size=(176, 80), mode="bilinear", align_corners=False).squeeze(0)
    x[0] = torch.clamp(x[0], -1.0, 1.0)
    x[1] = torch.round(x[1])
    mask = (x[1] != 0).float()
    y[0] = y[0] * mask
    y[1] = y[1] * mask
    y[0] = torch.clamp(y[0], *Y_BOUNDS["ux"])
    y[1] = torch.clamp(y[1], *Y_BOUNDS["uy"])
    y[2] = torch.clamp(y[2], *Y_BOUNDS["p"])
    return x, y


def load_model():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"Checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f}")
    model = AeroNet(in_c=3, num_c=3, base=16)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["stats"]


def predict(model, stats, x, mask):
    """x: [3, 176, 80] — уже нормированный. Возвращает денормированный pred."""
    x_mean = stats["x_mean"]; x_std = stats["x_std"]
    y_mean = stats["y_mean"]; y_std = stats["y_std"]

    # Нормируем вход (как в train)
    x_n = x.clone()
    x_n[0:1] = (x_n[0:1] - x_mean[:, 0:1]) / x_std[:, 0:1]
    x_n[2:3] = (x_n[2:3] - x_mean[:, 2:3]) / x_std[:, 2:3]
    # канал 1 (map_val) — не нормируем

    with torch.no_grad():
        pred_n = model(x_n.unsqueeze(0)).squeeze(0)

    # Денормируем выход
    pred = pred_n * y_std.squeeze().view(3, 1, 1) + y_mean.squeeze().view(3, 1, 1)
    return pred


def compute_rel_l2(pred, target):
    """rel_L2 per-channel."""
    rel = []
    for c in range(3):
        diff = pred[c] - target[c]
        r = torch.norm(diff) / (torch.norm(target[c]) + 1e-8)
        rel.append(r.item())
    return rel


def plot_sample(ax_row, x, y_true, y_pred, mask, title_prefix, sample_id):
    """Три строки: Ux, Uy, P. Четыре колонки: target, pred, |error|, error в жидкости."""
    names = ["Ux", "Uy", "P"]
    cmaps = ["RdBu_r", "RdBu_r", "RdBu_r"]

    for c, name in enumerate(names):
        # Target
        vmin = float(y_true[c].min())
        vmax = float(y_true[c].max())
        if abs(vmin - vmax) < 1e-6:
            vmax = vmin + 1e-3

        ax_row[c, 0].imshow(y_true[c].numpy(), cmap=cmaps[c], vmin=vmin, vmax=vmax)
        ax_row[c, 0].set_title(f"{title_prefix} {name} TARGET", fontsize=9)
        ax_row[c, 0].axis("off")

        # Predicted
        ax_row[c, 1].imshow(y_pred[c].numpy(), cmap=cmaps[c], vmin=vmin, vmax=vmax)
        ax_row[c, 1].set_title(f"{name} PRED", fontsize=9)
        ax_row[c, 1].axis("off")

        # Absolute error
        err = (y_pred[c] - y_true[c]).abs().numpy()
        im = ax_row[c, 2].imshow(err, cmap="hot")
        ax_row[c, 2].set_title(f"{name} |ERR| max={err.max():.4f}", fontsize=9)
        ax_row[c, 2].axis("off")
        plt.colorbar(im, ax=ax_row[c, 2], fraction=0.046, pad=0.04)

        # Error внутри жидкости
        err_fluid = err * mask.squeeze().numpy()
        ax_row[c, 3].imshow(err_fluid, cmap="hot")
        ax_row[c, 3].set_title(f"{name} |ERR| (fluid)", fontsize=9)
        ax_row[c, 3].axis("off")


def main():
    model, stats = load_model()
    X = np.asarray(pickle.load(open(DATA_DIR / "dataX.pkl", "rb")), dtype=np.float32)
    Y = np.asarray(pickle.load(open(DATA_DIR / "dataY.pkl", "rb")), dtype=np.float32)

    # Найдём лучший, средний, худший пример
    print("\nВычисляем метрики на всём датасете...")
    rels = []
    samples_cache = []
    for i in range(len(X)):
        x, y_true = preprocess_one(X[i], Y[i])
        mask = (x[1:2] != 0).float()
        pred = predict(model, stats, x, mask)
        rel = compute_rel_l2(pred, y_true)
        rels.append(rel)
        samples_cache.append((x, y_true, pred, mask))

    rels = np.array(rels)
    mean_ux = rels[:, 0].mean()
    print(f"Средние по датасету: Ux={mean_ux:.4f}, Uy={rels[:,1].mean():.4f}, P={rels[:,2].mean():.4f}")

    # Сортируем по Ux
    order = np.argsort(rels[:, 0])
    worst_id = int(order[-1])
    best_id = int(order[0])
    mid_id = int(order[len(order) // 2])

    print(f"\nЛучший пример: {best_id}, Ux rel_L2 = {rels[best_id, 0]:.4f}")
    print(f"Средний пример: {mid_id}, Ux rel_L2 = {rels[mid_id, 0]:.4f}")
    print(f"Худший пример: {worst_id}, Ux rel_L2 = {rels[worst_id, 0]:.4f}")

    # Создаём три больших фигуры
    for label, sid in [("best", best_id), ("mid", mid_id), ("worst", worst_id)]:
        x, y_true, pred, mask = samples_cache[sid]
        fig, axes = plt.subplots(3, 4, figsize=(18, 11))
        plot_sample(axes, x, y_true, pred, mask, f"[{label} #{sid}]", sid)

        rel = rels[sid]
        fig.suptitle(
            f"AeroNet — {label.upper()} (idx={sid})  |  "
            f"Ux rel_L2={rel[0]:.4f}, Uy={rel[1]:.4f}, P={rel[2]:.4f}",
            fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        out = f"viz_{label}.png"
        plt.savefig(out, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"✓ Сохранено: {out}")

    # Гистограмма распределения ошибок Ux
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for c, (ax, name) in enumerate(zip(axes, ["Ux", "Uy", "P"])):
        ax.hist(rels[:, c], bins=50, color="steelblue", edgecolor="white")
        ax.axvline(rels[:, c].mean(), color="red", linestyle="--",
                   label=f"mean={rels[:, c].mean():.4f}")
        ax.axvline(np.median(rels[:, c]), color="green", linestyle=":",
                   label=f"median={np.median(rels[:, c]):.4f}")
        ax.set_title(f"{name} rel_L2 на 981 примерах")
        ax.set_xlabel("rel_L2")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("viz_histogram.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("✓ Сохранено: viz_histogram.png")

    # Гистограмма предсказаний Ux vs true Ux на одном примере
    x, y_true, pred, mask = samples_cache[mid_id]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    mask_f = mask.squeeze().numpy().astype(bool)

    ux_true = y_true[0].numpy()[mask_f]
    ux_pred = pred[0].numpy()[mask_f]
    axes[0].hist(ux_true, bins=80, alpha=0.6, label="true", color="blue")
    axes[0].hist(ux_pred, bins=80, alpha=0.6, label="pred", color="red")
    axes[0].set_title(f"Ux распределение (пример {mid_id}, только жидкость)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Scatter true vs pred
    axes[1].scatter(ux_true, ux_pred, s=1, alpha=0.3)
    lim = [min(ux_true.min(), ux_pred.min()), max(ux_true.max(), ux_pred.max())]
    axes[1].plot(lim, lim, "k--", linewidth=1)
    axes[1].set_xlabel("true Ux")
    axes[1].set_ylabel("pred Ux")
    axes[1].set_title("Scatter: true vs pred")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("viz_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("✓ Сохранено: viz_distribution.png")

    # Открыть все картинки
    import subprocess, sys
    for f in ["viz_best.png", "viz_mid.png", "viz_worst.png",
              "viz_histogram.png", "viz_distribution.png"]:
        try:
            if sys.platform.startswith("win"):
                import os; os.startfile(f)
            elif sys.platform == "darwin":
                subprocess.run(["open", f])
            else:
                subprocess.run(["xdg-open", f])
        except Exception:
            pass


if __name__ == "__main__":
    main()