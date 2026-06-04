#!/usr/bin/env python3
"""
video_cut.py

Grab single-frame "shots" from a video at a few specified timestamps and save
each one as a PNG. Useful for pulling stills out of a clip for figures.

Run:
    python3 video_cut.py
(only needs OpenCV, no ROS)
"""

import os
import cv2

# ============================ CONFIG — edit these ============================
VIDEO_PATH = "/home/shaohang/workspace/underwater_ws/data_rosbag/output.avi"

TIMES_SEC  = [1, 16, 32, 55]    # timestamps (seconds) to grab a frame at
OUT_DIR    = "."                # where to write the PNGs
OUT_PREFIX = "shot"             # files: shot_001s.png, shot_016s.png, ...
# ===========================================================================


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, total, fps


def grab_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None


def main():
    cap, total, fps = open_video(VIDEO_PATH)
    duration = total / fps
    print(f"{VIDEO_PATH}: {total} frames @ {fps:.1f} fps ({duration:.1f}s)")
    os.makedirs(OUT_DIR, exist_ok=True)

    for t in TIMES_SEC:
        idx = int(round(t * fps))
        if idx > total - 1:
            if t > duration + 0.5:               # genuinely past the clip
                print(f"  skip {t:>3d}s -> past the end ({duration:.1f}s)")
                continue
            idx = total - 1                       # at the end -> use the last frame
        frame = grab_frame(cap, idx)
        if frame is None:
            print(f"  skip {t:>3d}s -> could not read frame {idx}")
            continue
        out = os.path.join(OUT_DIR, f"{OUT_PREFIX}_{t:03d}s.png")
        cv2.imwrite(out, frame)
        print(f"  {t:>3d}s -> frame {idx:6d} -> {out}")

    cap.release()


if __name__ == "__main__":
    main()
