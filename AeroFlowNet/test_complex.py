"""
Тест на СЛОЖНОЙ геометрии: biplane (два профиля NACA друг над другом).

Запуск: python test_complex.py
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pickle, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import AeroNet

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
CKPT = HERE / "checkpoints" / "best.pt"

with open(DATA_DIR / "clip_bounds.json") as f:
    Y_BOUNDS = {k: tuple(v) for k, v in json.load(f).items()}


# ========================
# NACA 4-digit генератор
# ========================
def naca4_outline(m, p, t, n=200, chord=1.0, x_offset=0.0, y_offset=0.0):
    """
    Возвращает контур NACA в виде массива точек.
    m: max camber, p: position, t: thickness
    """
    x = np.linspace(0, 1, n)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x
                  - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    yc = np.where(x < p,
                  m / p**2 * (2 * p * x - x**2),
                  m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(x < p,
                   2 * m / p**2 * (p - x),
                   2 * m / (1 - p)**2 * (p - x))
    theta = np.arctan(dyc)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Контур: от нижней кромки, вверх по верхней, обратно
    x_out = np.concatenate([xl[::-1], xu[1:]])
    y_out = np.concatenate([yl[::-1], yu[1:]])

    # Масштаб по хорде + сдвиг
    x_out = x_out * chord + x_offset
    y_out = y_out * chord + y_offset

    return x_out, y_out


def shape_biplane():
    """
    Два профиля NACA 0012 друг над другом.
    Нижний сдвинут назад относительно верхнего.
    Классическая конфигурация биплана.
    """
    def fn(u, v):
        # Верхнее крыло
        x1, y1 = naca4_outline(m=0.0, p=0.4, t=0.12,
                                chord=1.0, x_offset=-0.55, y_offset=0.15)
        # Нижнее крыло (сдвинуто назад)
        x2, y2 = naca4_outline(m=0.0, p=0.4, t=0.12,
                                chord=1.0, x_offset=-0.35, y_offset=-0.15)

        from matplotlib.path import Path
        path1 = Path(np.column_stack([x1, y1]))
        path2 = Path(np.column_stack([x2, y2]))

        pts = np.column_stack([u.flatten(), v.flatten()])
        inside = path1.contains_points(pts) | path2.contains_points(pts)
        return inside.reshape(u.shape)
    return fn


def shape_wing_with_flap():
    """
    Крыло NACA 2412 + отклонённый закрылок позади.
    Классическая механизация крыла.
    """
    def fn(u, v):
        # Основной профиль
        x1, y1 = naca4_outline(m=0.02, p=0.4, t=0.12,
                                chord=0.8, x_offset=-0.7, y_offset=0.0)
        # Закрылок — меньше, отклонён вниз
        x2, y2 = naca4_outline(m=0.02, p=0.4, t=0.10,
                                chord=0.3, x_offset=0.10, y_offset=-0.10)
        # Отклонить закрылок вниз (поворот на -15° вокруг x=0.10)
        angle = np.deg2rad(-15)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx = x2 - 0.10
        dy = y2 - (-0.10)
        x2_rot = 0.10 + dx * cos_a - dy * sin_a
        y2_rot = -0.10 + dx * sin_a + dy * cos_a

        from matplotlib.path import Path
        path1 = Path(np.column_stack([x1, y1]))
        path2 = Path(np.column_stack([x2_rot, y2_rot]))

        pts = np.column_stack([u.flatten(), v.flatten()])
        inside = path1.contains_points(pts) | path2.contains_points(pts)
        return inside.reshape(u.shape)
    return fn


# ========================
# Создание примера
# ========================
def create_new_sample(shape_fn, X_ref):
    H, W = X_ref.shape[1], X_ref.shape[2]
    wall_sdf = X_ref[2].copy()

    orig_map = X_ref[1].copy()
    new_map = orig_map.copy()
    new_map[new_map == 0] = 1

    bh, bw = 35, 35
    ch, cw = H // 2, W // 2

    uu, vv = np.meshgrid(
        np.linspace(-1, 1, 2 * bh),
        np.linspace(-1, 1, 2 * bw),
        indexing="ij",
    )
    body_local = shape_fn(uu, vv)
    body_full = np.zeros((H, W), dtype=bool)
    body_full[ch - bh:ch + bh, cw - bw:cw + bw] = body_local
    new_map[body_full] = 0

    dist_out = distance_transform_edt(~body_full)
    dist_in = distance_transform_edt(body_full)
    new_sdf = dist_out - dist_in
    new_sdf[body_full] = -50.0

    return np.stack([new_sdf, new_map, wall_sdf], axis=0).astype(np.float32)


# ========================
# Инференс
# ========================
def load_model():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    stats = ckpt["stats"]
    model = AeroNet(in_c=3, num_c=3, base=16)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, stats


def predict_x(X_raw, model, stats):
    x_mean, x_std = stats["x_mean"], stats["x_std"]
    y_mean, y_std = stats["y_mean"], stats["y_std"]

    x = torch.from_numpy(X_raw).float().unsqueeze(0)
    x = F.interpolate(x, size=(176, 80), mode="bilinear", align_corners=False).squeeze(0)
    x[0] = torch.clamp(x[0], -1.0, 1.0)
    x[1] = torch.round(x[1])

    x_n = x.clone()
    x_n[0] = (x_n[0] - x_mean[0, 0, 0, 0]) / x_std[0, 0, 0, 0]
    x_n[2] = (x_n[2] - x_mean[0, 2, 0, 0]) / x_std[0, 2, 0, 0]

    with torch.no_grad():
        pred_n = model(x_n.unsqueeze(0)).squeeze(0)

    pred = pred_n * y_std.squeeze().view(3, 1, 1) + y_mean.squeeze().view(3, 1, 1)
    mask_flow = (x[1] != 0).float()
    return pred, mask_flow


# ========================
# Невязка
# ========================
def divergence_residual(Ux, Uy):
    dUx_dx = np.gradient(Ux, axis=1)
    dUy_dy = np.gradient(Uy, axis=0)
    return np.abs(dUx_dx + dUy_dy)


# ========================
# Визуализация
# ========================
def visualize(name, X_raw, model, stats, save_dir="."):
    pred, mask = predict_x(X_raw, model, stats)

    Ux = pred[0].detach().numpy() * mask.numpy()
    Uy = pred[1].detach().numpy() * mask.numpy()
    P = pred[2].detach().numpy()
    mag = np.sqrt(Ux**2 + Uy**2)
    resid = divergence_residual(Ux, Uy) * mask.numpy()

    H, W = Ux.shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=180)

    # Magnitude — с bicubic
    ax = axes[0, 0]
    im = ax.imshow(mag, cmap="turbo", origin="upper",
                   interpolation="bicubic")
    ax.set_title(f"|U| — {name}", fontsize=13)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.axis("off")

    # Streamlines — плотнее, толще
    ax = axes[0, 1]
    ax.imshow(mag, cmap="Greys", alpha=0.25, origin="upper",
              interpolation="bicubic")
    strm = ax.streamplot(
        X, Y, Ux, Uy,
        color=mag, cmap="turbo",
        linewidth=1.6,
        density=2.4,
        arrowsize=1.3,
    )
    plt.colorbar(strm.lines, ax=ax, fraction=0.046)
    ax.set_title("Streamlines", fontsize=13)
    ax.invert_yaxis()
    ax.axis("off")

    # Pressure
    ax = axes[1, 0]
    im = ax.imshow(P, cmap="RdBu_r", origin="upper",
                   interpolation="bicubic")
    ax.set_title(f"P — {name}", fontsize=13)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.axis("off")

    # Residual (карта доверия)
    ax = axes[1, 1]
    im = ax.imshow(resid, cmap="inferno", origin="upper",
                   interpolation="bicubic")
    ax.set_title(f"|∇·u| — max={resid.max():.4f}", fontsize=13)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.axis("off")

    plt.suptitle(f"OOD: {name}", fontsize=15, fontweight="bold", y=1.00)
    plt.tight_layout()

    out = f"{save_dir}/complex_{name}.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")

    return out, float(resid.max()), float(resid[mask.numpy().astype(bool)].mean())


# ========================
# MAIN
# ========================
def main():
    model, stats = load_model()
    print("✓ Модель загружена")

    X_data = np.asarray(pickle.load(open(DATA_DIR / "dataX.pkl", "rb")), dtype=np.float32)
    X_ref = X_data[0]

    shapes = {
        "biplane":  shape_biplane(),
        "wing_flap": shape_wing_with_flap(),
    }

    print(f"\n{'='*72}")
    print(f"{'shape':>12} | {'resid max':>12} | {'resid mean':>12} | оценка")
    print(f"{'='*72}")

    results = []
    for name, fn in shapes.items():
        print(f"\nСоздание формы: {name}...")
        X_new = create_new_sample(fn, X_ref)
        _, rmax, rmean = visualize(name, X_new, model, stats)
        if rmax < 0.05:
            verdict = "✓ надёжно"
        elif rmax < 0.15:
            verdict = "○ терпимо"
        else:
            verdict = "⚠ модель врёт"
        print(f"{name:>12} | {rmax:>12.5f} | {rmean:>12.5f} | {verdict}")
        results.append((name, rmax, rmean))

    print(f"\n{'='*72}")
    print("Готово. Смотри complex_*.png")
    print(f"{'='*72}")

    # Открыть биплан — самую интересную
    import os, sys, subprocess
    try:
        if sys.platform.startswith("win"):
            os.startfile("complex_biplane.png")
        elif sys.platform == "darwin":
            subprocess.run(["open", "complex_biplane.png"])
        else:
            subprocess.run(["xdg-open", "complex_biplane.png"])
    except Exception:
        pass


if __name__ == "__main__":
    main()