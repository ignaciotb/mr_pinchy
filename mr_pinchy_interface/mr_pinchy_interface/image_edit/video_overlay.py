#!/usr/bin/env python3
"""
video_overlay.py

Composite frames sampled across a video into a single "ghost trail" image,
fading each frame's transparency according to its time in the clip. Useful for
visualizing motion / a trajectory as a long-exposure-style still.

Two modes (set MODE below):
    "still" : blend N frames sampled across the whole clip into ONE image and
              save it as a PNG. Alpha ramps with time so the trail fades.
    "video" : write a video where every output frame is a running blend of the
              most recent WINDOW frames (a moving motion-trail effect).

Run:
    python3 video_overlay.py
(only needs OpenCV + NumPy, no ROS)
"""

import cv2
import numpy as np

# ============================ CONFIG — edit these ============================
VIDEO_PATH = "/home/shaohang/workspace/underwater_ws/data_rosbag/jun2.avi"

MODE = "still"          # "still" -> one composite PNG 

# --- frame range to use (in frames; END < 0 means "to the end") ---
START_FRAME = 0
END_FRAME   = -1

# --- pick exact frames by time (seconds). If non-empty, this OVERRIDES the
#     START/END range + NUM_FRAMES sampling and blends these timestamps only.
#     Order matters: earliest = oldest (faintest) ghost, latest = solid base. ---
# FRAME_TIMES = [8.0, 14.0, 20.0, 26.0, 33.0]
FRAME_TIMES = [4.0, 12.0, 20.0, 31.0]

# --- "still" mode ---
NUM_FRAMES  = 5        # how many frames to blend; <= 0 means EVERY frame in range
FADE        = "linear"  # "linear" or "ease" — how alpha ramps with time
MIN_ALPHA   = 0.8     # opacity of the faintest (oldest) ghost
GHOST_MAX_ALPHA = 0.8  # opacity of the strongest ghost (the 2nd-newest frame).
                       # Keep < 1 so ghosts never fully bury the solid base.
FG_THRESH   = 25       # per-pixel diff-from-background (0-255) above which a
                       # pixel counts as "moving robot" and gets ghosted.
OUT_IMAGE   = "overlay_still.png"
# ===========================================================================


def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, total, fps


def frame_range(total):
    start = max(0, START_FRAME)
    end   = total if END_FRAME < 0 else min(END_FRAME, total)
    if end <= start:
        raise ValueError(f"Empty frame range [{start}, {end})")
    return start, end


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None


def time_alpha(t, fade):
    """Map normalized time t in [0,1] -> ghost opacity in [MIN_ALPHA, GHOST_MAX_ALPHA]."""
    w = t * t if fade == "ease" else t          # most-recent ghost (t=1) is strongest
    return MIN_ALPHA + (GHOST_MAX_ALPHA - MIN_ALPHA) * w


def foreground_mask(frame, bg):
    """Soft [0,1] per-pixel mask of where `frame` differs from the static bg.

    The robot moves; the scene doesn't. So |frame - bg| is large on the robot
    and ~0 on the background. We return a smoothed mask so ghost edges feather.
    """
    diff = np.abs(frame - bg).max(axis=2)        # strongest channel difference
    m = np.clip((diff - FG_THRESH) / FG_THRESH, 0.0, 1.0)
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=2.0)  # feather the silhouette edges
    return m[..., None]                          # HxWx1 so it broadcasts over BGR


def make_still(cap, start, end, fps):
    """Composite a fading motion trail of the moving robot onto a solid scene.

    These are full frames over a static background, so we can't just stack them
    (each opaque frame would bury the others). Instead we estimate the static
    background (per-pixel median), lay down the newest frame as the solid base,
    then paint ONLY the moving robot from each earlier frame on top with a
    time-decaying alpha. The latest pose stays crisp; older poses fade.
    """
    if FRAME_TIMES:
        # clamp to the last valid frame so a timestamp at/just past the very end
        # (e.g. t == duration) still resolves to a readable frame.
        idxs = np.array([min(int(round(t * fps)), end - 1) for t in FRAME_TIMES])
    elif NUM_FRAMES <= 0:
        idxs = np.arange(start, end)             # every frame in the range
    else:
        idxs = np.linspace(start, end - 1, NUM_FRAMES).astype(int)

    frames = []
    for idx in idxs:
        f = read_frame(cap, idx)
        if f is not None:
            frames.append((idx, f.astype(np.float32)))
    if not frames:
        raise RuntimeError("No frames could be read from the requested range")

    bg = np.median(np.stack([f for _, f in frames]), axis=0)   # static scene

    newest_idx, out = frames[-1]                 # solid base on the bottom
    out = out.copy()
    ghosts = frames[:-1]
    for k, (idx, f) in enumerate(ghosts):        # oldest -> 2nd-newest, on top
        t = k / max(1, len(ghosts) - 1)          # 0=oldest, 1=most-recent ghost
        a = time_alpha(t, FADE) * foreground_mask(f, bg)   # only the robot, faded
        out = a * f + (1.0 - a) * out
        print(f"  ghost {idx:6d}  alpha<= {time_alpha(t, FADE):.3f}")
    print(f"  base  {newest_idx:6d}  solid scene + latest pose")

    out = np.clip(out, 0, 255).astype(np.uint8)
    cv2.imwrite(OUT_IMAGE, out)
    print(f"Wrote composite of {len(frames)} frames -> {OUT_IMAGE}")


def main():
    cap, total, fps = open_video(VIDEO_PATH)
    start, end = frame_range(total)
    print(f"{VIDEO_PATH}: {total} frames @ {fps:.1f} fps, using [{start}, {end})")

    if MODE == "still":
        make_still(cap, start, end, fps)
    else:
        raise ValueError(f"Unknown MODE: {MODE!r} (use 'still' or 'video')")

    cap.release()


if __name__ == "__main__":
    main()
