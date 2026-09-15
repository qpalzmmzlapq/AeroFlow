"""
Диагностика Ux-проблемы.

Тест 1: Absolute error vs relative error — обманывает ли метрика.
Тест 2: Анализ аномальных примеров (876, 351, 484, 226).
Тест 3: Per-example разброс Ux — глобальная ли нормировка проблема.

Запуск: python diagnose.py
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pickle, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

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


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"Checkpoint: epoch={ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f}")

    stats = ckpt["stats"]
    x_mean, x_std = stats["x_mean"], stats["x_std"]
    y_mean, y_std = stats["y_mean"], stats["y_std"]

    model = AeroNet(in_c=3, num_c=3, base=16)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X_raw = np.asarray(pickle.load(open(DATA_DIR / "dataX.pkl", "rb")), dtype=np.float32)
    Y_raw = np.asarray(pickle.load(open(DATA_DIR / "dataY.pkl", "rb")), dtype=np.float32)

    print(f"\nДатасет: X={X_raw.shape}, Y={Y_raw.shape}")

    # ==========================================================
    # ПРЕПРОЦЕССИНГ ВСЕХ ПРИМЕРОВ
    # ==========================================================
    print("\nПрепроцессинг...")
    X_all, Y_all = [], []
    for i in range(len(X_raw)):
        x, y = preprocess_one(X_raw[i], Y_raw[i])
        X_all.append(x); Y_all.append(y)
    X_all = torch.stack(X_all)   # [N, 3, 176, 80]
    Y_all = torch.stack(Y_all)

    # Нормировка
    X_n = X_all.clone()
    X_n[:, 0:1] = (X_n[:, 0:1] - x_mean[:, 0:1]) / x_std[:, 0:1]
    X_n[:, 2:3] = (X_n[:, 2:3] - x_mean[:, 2:3]) / x_std[:, 2:3]
    Y_n = (Y_all - y_mean) / y_std

    # Инференс
    print("Инференс...")
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_n), 32):
            p = model(X_n[i:i+32])
            preds.append(p)
    preds = torch.cat(preds)

    # ==========================================================
    # МЕТРИКИ ПО КАЖДОМУ ПРИМЕРУ
    # ==========================================================
    # Relative L2 (по нормированным полям)
    diff_n = (preds - Y_n).reshape(len(preds), 3, -1)
    tgt_n = Y_n.reshape(len(Y_n), 3, -1)
    rel = (torch.norm(diff_n, dim=2) / (torch.norm(tgt_n, dim=2) + 1e-8)).numpy()

    # Absolute error (в физических единицах, после денормализации)
    Y_phys = Y_n * y_std + y_mean        # [N, 3, 176, 80]
    pred_phys = preds * y_std + y_mean
    abs_err = (pred_phys - Y_phys).abs()   # [N, 3, 176, 80]

    # ==========================================================
    # ТЕСТ 1: Absolute vs Relative
    # ==========================================================
    print(f"\n{'='*70}")
    print("ТЕСТ 1: Absolute error vs Relative error")
    print(f"{'='*70}")
    print(f"{'channel':>8} | {'rel_L2 mean':>12} | {'abs err mean':>13} | "
          f"{'true std':>10} | {'abs/std':>8}")
    print("-" * 70)

    for c, name in enumerate(["Ux", "Uy", "P"]):
        rel_mean = rel[:, c].mean()
        abs_mean = abs_err[:, c].mean().item()
        true_std = Y_phys[:, c].std().item()
        ratio = abs_mean / (true_std + 1e-8)
        print(f"{name:>8} | {rel_mean:>12.4f} | {abs_mean:>13.5f} | "
              f"{true_std:>10.5f} | {ratio:>8.4f}")

    print("\nИнтерпретация:")
    print("  abs/std < 0.15  → метрика rel_L2 обманывает, физически всё ок")
    print("  abs/std 0.15-0.3 → терпимо")
    print("  abs/std > 0.3   → реальная проблема")

    # ==========================================================
    # ТЕСТ 2: Аномальные примеры
    # ==========================================================
    print(f"\n{'='*70}")
    print("ТЕСТ 2: Аномальные примеры (плохие по всем трём каналам)")
    print(f"{'='*70}")

    # Найдём примеры, у которых плохие ВСЕ три канала
    anomaly_mask = (rel[:, 0] > 0.5) & (rel[:, 1] > 0.5) & (rel[:, 2] > 0.3)
    anomaly_ids = np.where(anomaly_mask)[0]
    print(f"Примеров с плохими ВСЕМИ каналами: {len(anomaly_ids)}")
    print(f"ID: {anomaly_ids[:20].tolist()}{'...' if len(anomaly_ids) > 20 else ''}")

    # Разбор топ-5 аномальных
    print(f"\nРазбор топ-5 аномальных:")
    for idx in anomaly_ids[:5]:
        print(f"\n--- Пример {idx} ---")
        print(f"  rel_L2: Ux={rel[idx,0]:.4f}, Uy={rel[idx,1]:.4f}, P={rel[idx,2]:.4f}")

        # Сырые данные
        mv = X_raw[idx, 1]
        uniq = np.unique(mv)
        print(f"  map_val unique: {uniq}")
        for v in uniq:
            n = (mv == v).sum()
            print(f"    {v}: {n} px ({100*n/mv.size:.2f}%)")

        # Диапазоны Y
        for c, name in enumerate(["Ux", "Uy", "P"]):
            y_c = Y_raw[idx, c]
            print(f"  {name}: min={y_c.min():.4f}, max={y_c.max():.4f}, "
                  f"mean={y_c.mean():.4f}, std={y_c.std():.4f}")

    # ==========================================================
    # ТЕСТ 3: Per-example разброс Ux
    # ==========================================================
    print(f"\n{'='*70}")
    print("ТЕСТ 3: Разброс Ux по примерам (глобальная vs per-example нормировка)")
    print(f"{'='*70}")

    # Для каждого примера — mean и std исходного Ux в жидкости
    means = []
    stds = []
    for i in range(len(X_raw)):
        mask_i = (Y_all[i, 0].numpy() != 0)   # где Ux не занулён
        if mask_i.sum() > 0:
            ux_i = Y_all[i, 0].numpy()[mask_i]
            means.append(ux_i.mean())
            stds.append(ux_i.std())

    means = np.array(means)
    stds = np.array(stds)

    print(f"Ux mean по примерам: min={means.min():.4f}, max={means.max():.4f}, "
          f"std={means.std():.4f}")
    print(f"Ux std по примерам:  min={stds.min():.4f}, max={stds.max():.4f}, "
          f"std={stds.std():.4f}")

    # Глобальная нормировка
    global_mean = y_mean[0, 0, 0, 0].item()
    global_std = y_std[0, 0, 0, 0].item()
    print(f"\nГлобальная нормировка: mean={global_mean:.4f}, std={global_std:.4f}")

    # Разброс relative to global std
    spread = stds.std() / (global_std + 1e-8)
    print(f"std(stds) / global_std = {spread:.4f}")
    print(f"Интерпретация:")
    print(f"  <0.2 → глобальная нормировка ОК")
    print(f"  0.2-0.5 → возможно, per-example помогла бы")
    print(f"  >0.5 → глобальная нормировка точно ломает")

    # ==========================================================
    # ТЕСТ 4: Сравнение с ground truth по абсолютной ошибке
    # ==========================================================
    print(f"\n{'='*70}")
    print("ТЕСТ 4: Где ошибка локализована (для худшего примера)")
    print(f"{'='*70}")

    worst_idx = int(np.argmax(rel[:, 0]))
    print(f"Худший по Ux: пример {worst_idx}, rel={rel[worst_idx, 0]:.4f}")

    # Ошибка в разных зонах
    y_true_w = Y_phys[worst_idx].numpy()
    y_pred_w = pred_phys[worst_idx].numpy()
    err_w = np.abs(y_pred_w - y_true_w)

    # Разбиваем на 4 зоны по map_val
    mv_w = X_all[worst_idx, 1].numpy()
    for label in [0, 1, 2, 3, 4]:
        zone = (np.round(mv_w) == label)
        if zone.sum() == 0:
            continue
        print(f"\n  zone label={label} ({zone.sum()} px):")
        for c, name in enumerate(["Ux", "Uy", "P"]):
            err_zone = err_w[c][zone]
            true_zone = y_true_w[c][zone]
            print(f"    {name}: abs_err mean={err_zone.mean():.5f}, "
                  f"true range=[{true_zone.min():.4f}, {true_zone.max():.4f}]")

    # ==========================================================
    # ИТОГ
    # ==========================================================
    print(f"\n{'='*70}")
    print("ИТОГ")
    print(f"{'='*70}")

    ux_abs_ratio = abs_err[:, 0].mean().item() / (Y_phys[:, 0].std().item() + 1e-8)
    print(f"1. Ux abs/std ratio: {ux_abs_ratio:.4f}")
    print(f"2. Аномальных примеров (все три канала плохие): {len(anomaly_ids)}")
    print(f"3. Разброс per-example std Ux: {spread:.4f}")
    print()
    print("Что делать:")
    if ux_abs_ratio < 0.15:
        print("  → Метрика обманывает. Ux физически ОК. Проблема в rel_L2.")
    elif ux_abs_ratio < 0.3:
        print("  → Терпимо. Можно учить дольше (100+ эпох).")
    else:
        print("  → Реальная проблема. Смотри Тест 3 и Тест 4.")

    if spread > 0.3:
        print("  → Per-example нормировка Ux может помочь.")


if __name__ == "__main__":
    main()