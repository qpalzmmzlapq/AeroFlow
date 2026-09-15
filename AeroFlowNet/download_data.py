"""
Скачивание и распаковка DeepCFD.
Запуск: python download_data.py
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
ZIP_PATH = DATA_DIR / "DeepCFD.zip"
URL = "https://zenodo.org/record/3666056/files/DeepCFD.zip?download=1"


def download(url, dest):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Уже скачано: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return

    print(f"Скачивание: {url}")
    print(f"→ {dest}")

    def progress(block_num, block_size, total_size):
        done = block_num * block_size
        if total_size > 0:
            pct = 100 * done / total_size
            mb = done / 1e6
            total = total_size / 1e6
            sys.stdout.write(f"\r  {pct:5.1f}%  {mb:7.1f} / {total:.1f} MB")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print(f"\n✓ Скачано: {dest.stat().st_size/1e6:.1f} MB")


def extract(zip_path, out_dir):
    print(f"Распаковка: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    print(f"✓ Распаковано в: {out_dir}")


def find_pkl(root):
    found = {}
    for p in root.rglob("*.pkl"):
        if p.name in ("dataX.pkl", "dataY.pkl", "X.pkl", "Y.pkl"):
            found[p.name] = p
    return found


def main():
    # 1. Скачать
    try:
        download(URL, ZIP_PATH)
    except Exception as e:
        print(f"\n✗ Ошибка скачивания: {e}")
        return

    # 2. Распаковать
    try:
        extract(ZIP_PATH, DATA_DIR)
    except Exception as e:
        print(f"\n✗ Ошибка распаковки: {e}")
        return

    # 3. Найти .pkl и переместить в data/
    pkls = find_pkl(DATA_DIR)
    if not pkls:
        print("✗ .pkl не найдены после распаковки")
        return

    for name, path in pkls.items():
        target = DATA_DIR / name
        if path != target:
            path.rename(target)
        print(f"  {name}: {target} ({target.stat().st_size/1e6:.1f} MB)")

    # 4. Удалить zip для экономии места
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
        print(f"✓ Удалён {ZIP_PATH.name}")

    print(f"\n✓ Готово. Данные в: {DATA_DIR}")


if __name__ == "__main__":
    main()