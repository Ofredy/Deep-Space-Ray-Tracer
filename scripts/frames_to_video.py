#!/usr/bin/env python3
"""
frames_to_video.py

Usage:
  python frames_to_video.py --input_dir /path/to/frames --fps 24
  python frames_to_video.py --input_dir /path/to/frames --fps 24 --out /custom/output.mp4
"""

import argparse, os, sys, re, glob
import cv2


def natural_key(s):
    # numeric-aware sort: img2.png < img10.png
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r'\d+|[\D]+', os.path.basename(s))]


def main():

    p = argparse.ArgumentParser(description="Make a video from a directory of frames.")
    p.add_argument("--input_dir", type=str, required=True, help="Directory containing image frames")
    p.add_argument("--fps", type=float, required=True, help="Frames per second (e.g., 24 or 30)")
    p.add_argument("--out", type=str, default=None, help="Optional output video path (.mp4)")
    args = p.parse_args()

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(args.input_dir, e)))
    files = sorted(files, key=natural_key)

    if not files:
        print(f"[ERR] No image files found in: {args.input_dir}")
        sys.exit(1)

    first = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        print(f"[ERR] Failed to read first frame: {files[0]}")
        sys.exit(1)

    # Drop alpha if present
    if first.ndim == 3 and first.shape[2] == 4:
        first = cv2.cvtColor(first, cv2.COLOR_BGRA2BGR)

    h, w = first.shape[:2]

    # Determine output path
    if args.out is not None:
        out_path = args.out
    else:
        # Default behavior (same as before)
        out_name = os.path.basename(os.path.normpath(args.input_dir)) or "output"
        out_path = os.path.join(os.path.dirname(os.path.normpath(args.input_dir)),
                                f"{out_name}.mp4")

    # 'mp4v' is widely supported; if you specifically want H.264, use ffmpeg externally.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, args.fps, (w, h))

    if not vw.isOpened():
        print("[ERR] Could not open VideoWriter. Check codecs/OpenCV build.")
        sys.exit(1)

    written = 0
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Skipping unreadable frame: {f}")
            continue

        # Drop alpha if needed
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # Ensure consistent size
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        vw.write(img)
        written += 1

    vw.release()
    print(f"[OK] Wrote {written} frames → {out_path} at {args.fps} fps")


if __name__ == "__main__":
    main()
