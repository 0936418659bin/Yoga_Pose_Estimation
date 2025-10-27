#!/usr/bin/env python3
"""
normalize_and_reencode_and_split_videos.py

- Re-encode tất cả video trong src_root (e.g. yoga/videos) về FPS=30, resolution=1280x720 (720p).
- Lưu file đã encode vào output_root/{train|val|test}/videos/... giữ cấu trúc.
- Sử dụng ffmpeg nếu có, fallback sang OpenCV nếu không có ffmpeg.
"""

import os
import sys
import random
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.mpeg', '.mpg'}

def is_video_file(path: Path):
    return path.suffix.lower() in VIDEO_EXTS

def gather_video_files(src_root: Path):
    files_by_class = {}
    for root, dirs, files in os.walk(src_root):
        rootp = Path(root)
        vids = [rootp / f for f in files if is_video_file(rootp / f)]
        if not vids:
            continue
        rel = rootp.relative_to(src_root)
        class_label = str(rel).replace('\\','/')
        files_by_class.setdefault(class_label, []).extend(vids)
    return files_by_class

def ffmpeg_available():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False

def reencode_with_ffmpeg(src: Path, dst: Path, fps=30, width=1280, height=720):
    dst.parent.mkdir(parents=True, exist_ok=True)
    # -y overwrite, -hide_banner quiet is optional
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale='min({width},iw)':'min({height},ih)',pad={width}:{height}:(ow-iw)/2:(oh-ih)/2", 
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        str(dst)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[WARN] ffmpeg failed for {src}: {e}")
        return False

# Optional fallback using OpenCV:
def reencode_with_opencv(src: Path, dst: Path, fps=30, width=1280, height=720):
    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV not installed for fallback re-encode. Install opencv-python.")
        return False
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print("[WARN] Cannot open video with OpenCV:", src)
        return False
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(dst), fourcc, fps, (width, height))
    import math
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # resize/pad frame to target resolution preserving aspect ratio
        h, w = frame.shape[:2]
        scale = min(width/w, height/h)
        nw, nh = int(w*scale), int(h*scale)
        frame_resized = cv2.resize(frame, (nw, nh))
        top = (height - nh)//2
        left = (width - nw)//2
        canvas = 255 * np.ones((height, width, 3), dtype=frame.dtype)
        canvas[top:top+nh, left:left+nw] = frame_resized
        out.write(canvas)
    cap.release()
    out.release()
    return True

def split_list(items, ratios=(0.7,0.2,0.1), seed=42):
    random.Random(seed).shuffle(items)
    n = len(items)
    n1 = int(n * ratios[0])
    n2 = int(n * (ratios[0] + ratios[1]))
    return items[:n1], items[n1:n2], items[n2:]

def main():
    parser = argparse.ArgumentParser(description="Re-encode videos and split dataset.")
    parser.add_argument("--src_root", type=str, required=True, help="Source videos root, e.g. yoga/videos")
    parser.add_argument("--output_root", type=str, required=True, help="Output root, e.g. dataset_processed")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280, help="target width (default 1280 for 720p)")
    parser.add_argument("--height", type=int, default=720, help="target height (default 720)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Don't actually write files; just print planned operations")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    out_root = Path(args.output_root)

    if not src_root.exists():
        print("Source root not found:", src_root)
        sys.exit(1)

    files_by_class = gather_video_files(src_root)
    total = sum(len(v) for v in files_by_class.values())
    print(f"Found {total} videos across {len(files_by_class)} classes.")

    use_ffmpeg = ffmpeg_available()
    if use_ffmpeg:
        print("Using ffmpeg for re-encoding.")
    else:
        print("ffmpeg not available. Will try OpenCV fallback (needs opencv-python).")

    counts = {"train":0,"val":0,"test":0}
    for class_label, vids in files_by_class.items():
        train_v, val_v, test_v = split_list(list(vids), ratios=(0.7,0.2,0.1), seed=args.seed)
        mapping = [("train", train_v), ("val", val_v), ("test", test_v)]
        for split_name, items in mapping:
            for src_path in items:
                rel_path = Path(class_label) / src_path.name
                dst_path = out_root / split_name / "videos" / rel_path
                counts[split_name] += 1
                if args.dry_run:
                    continue
                if use_ffmpeg:
                    ok = reencode_with_ffmpeg(src_path, dst_path, fps=args.fps, width=args.width, height=args.height)
                    if not ok:
                        print(f"[WARN] ffmpeg failed for {src_path}.")
                else:
                    ok = reencode_with_opencv(src_path, dst_path, fps=args.fps, width=args.width, height=args.height)
                    if not ok:
                        print(f"[WARN] re-encode fallback failed for {src_path}.")
    print("Done. Counts:", counts)

if __name__ == "__main__":
    main()
