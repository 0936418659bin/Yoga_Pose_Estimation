#!/usr/bin/env python3
"""
normalize_and_split_images.py

- Resize tất cả ảnh trong `src_root` (ví dụ: yoga/images) về target_size (W,H).
- Lưu ảnh đã resize vào `tmp_root` giữ nguyên cấu trúc con (relative paths).
- Chia dataset theo class (mỗi "class" = folder parent cuối cùng, ví dụ "Viparita Karani Wrong")
  sang train/val/test theo tỷ lệ (70/20/10) đảm bảo phân bố theo class.
- Tạo output_root/train/images/...  output_root/val/images/... output_root/test/images/...
"""

import os
import sys
import random
import shutil
from PIL import Image
from pathlib import Path
from tqdm import tqdm  # optional but nice; nếu không muốn, bỏ dòng này
import argparse

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def is_image_file(path: Path):
    return path.suffix.lower() in IMAGE_EXTS

def gather_image_files(src_root: Path):
    files_by_class = {}  # key: class_label (relative folder path from src_root's parent depth), value: list of file paths
    for root, dirs, files in os.walk(src_root):
        rootp = Path(root)
        # skip the root itself if it has no image files
        img_files = [rootp / f for f in files if is_image_file(rootp / f)]
        if not img_files:
            continue
        # define class label = relative path from src_root (two-level or more are OK)
        # For example: src_root / "Viparita Karani" / "Viparita Karani Wrong" => class = "Viparita Karani/Viparita Karani Wrong"
        rel = rootp.relative_to(src_root)
        class_label = str(rel).replace('\\', '/')
        files_by_class.setdefault(class_label, []).extend(img_files)
    return files_by_class

def resize_and_save(src_path: Path, dst_path: Path, size=(224,224), quality=95):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(src_path) as im:
            # convert to RGB to avoid issues with palette / RGBA
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            im_resized = im.resize(size, Image.LANCZOS)
            im_resized.save(dst_path, quality=quality)
    except Exception as e:
        print(f"[WARN] Failed to process {src_path}: {e}")

def split_list(items, ratios=(0.7,0.2,0.1), seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-6
    random.Random(seed).shuffle(items)
    n = len(items)
    n1 = int(n * ratios[0])
    n2 = int(n * (ratios[0] + ratios[1]))
    return items[:n1], items[n1:n2], items[n2:]

def main():
    parser = argparse.ArgumentParser(description="Resize images and split into train/val/test preserving folder structure.")
    parser.add_argument("--src_root", type=str, required=True, help="Source images root, e.g. yoga/images")
    parser.add_argument("--output_root", type=str, required=True, help="Output root, e.g. dataset_processed")
    parser.add_argument("--size", type=int, nargs=2, default=(224,224), help="Target size W H, default 224 224")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--dry_run", action="store_true", help="Don't actually write files; just print counts")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    output_root = Path(args.output_root)
    img_out_base = output_root  # we will create train/images..., val/images..., test/images...

    if not src_root.exists():
        print("Source root not found:", src_root)
        sys.exit(1)

    print("Gathering image files...")
    files_by_class = gather_image_files(src_root)
    print(f"Found {sum(len(v) for v in files_by_class.values())} images across {len(files_by_class)} classes.")

    counts = {"train":0,"val":0,"test":0}
    for class_label, file_list in files_by_class.items():
        train_list, val_list, test_list = split_list(list(file_list), ratios=(0.7,0.2,0.1), seed=args.seed)
        mapping = [("train", train_list), ("val", val_list), ("test", test_list)]
        for split_name, items in mapping:
            for src_path in items:
                rel_path = Path(class_label) / src_path.name
                dst_path = img_out_base / split_name / "images" / rel_path
                counts[split_name] += 1
                if args.dry_run:
                    continue
                resize_and_save(src_path, dst_path, size=tuple(args.size))
    print("Done.")
    print("Counts:", counts)
    print("Output structure example:", output_root / "train" / "images")

if __name__ == "__main__":
    main()
