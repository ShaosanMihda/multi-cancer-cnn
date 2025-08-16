# prepare_data.py
import os, shutil, random
from pathlib import Path

random.seed(42)

RAW_ROOT = Path("data/raw")
SPLIT_ROOT = Path("data_split")
TRAIN = SPLIT_ROOT/"train"
VAL = SPLIT_ROOT/"val"
TEST = SPLIT_ROOT/"test"

# Find the deepest directory that directly contains class folders (>=2 subdirs)
def find_dataset_root(root: Path) -> Path:
    candidates = []
    for p in root.rglob('*'):
        if p.is_dir():
            subdirs = [d for d in p.iterdir() if d.is_dir()]
            if len(subdirs) >= 2:
                # check that subdirs contain images
                has_images = any(any(q.suffix.lower() in {'.png','.jpg','.jpeg','.bmp'} for q in d.rglob('*')) for d in subdirs)
                if has_images:
                    candidates.append(p)
    # Pick the deepest path (longest)
    if not candidates:
        raise RuntimeError("Could not locate dataset root with class subfolders.")
    return max(candidates, key=lambda x: len(str(x)))

DS_ROOT = find_dataset_root(RAW_ROOT)
print(f"[INFO] Using dataset root: {DS_ROOT}")

# Collect classes
classes = sorted([d.name for d in DS_ROOT.iterdir() if d.is_dir()])
print(f"[INFO] Found classes ({len(classes)}): {classes}")

# Clean previous split
if SPLIT_ROOT.exists():
    shutil.rmtree(SPLIT_ROOT)
for p in [TRAIN, VAL, TEST]:
    p.mkdir(parents=True, exist_ok=True)
    for c in classes:
        (p/c).mkdir(parents=True, exist_ok=True)

# Copy with 80/10/10 split per class
for c in classes:
    imgs = []
    for ext in ('*.png','*.jpg','*.jpeg','*.bmp'):
        imgs.extend((DS_ROOT/c).rglob(ext))
    imgs = sorted(imgs)
    n = len(imgs)
    if n == 0:
        print(f"[WARN] No images for class {c}")
        continue
    random.shuffle(imgs)
    n_train = int(0.8*n)
    n_val = int(0.1*n)

    splits = {
        TRAIN/c: imgs[:n_train],
        VAL/c:   imgs[n_train:n_train+n_val],
        TEST/c:  imgs[n_train+n_val:]
    }

    for dst, files in splits.items():
        for src in files:
            shutil.copy2(src, dst/src.name)

print("[DONE] Created data_split/{train,val,test} folders.")
