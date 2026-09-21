"""Concatenate the recorded demo segments into the final demo video.

Uses the concat demuxer of the ffmpeg binary bundled with imageio-ffmpeg;
all segments share codec/parameters so -c copy is lossless and instant.

Usage:
    venv/Scripts/python.exe scripts/concat_video.py
Output:
    data/recording/pathsense_demo.mp4
"""
import os
import subprocess
import sys

import imageio_ffmpeg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG_DIR = os.path.join(ROOT, "data", "recording")
OUT = os.path.join(SEG_DIR, "pathsense_demo.mp4")

SEGMENTS = ["seg_intro.mp4", "seg_a.mp4", "seg_b.mp4", "seg_c.mp4",
            "seg_outro.mp4"]


def main():
    missing = [s for s in SEGMENTS
               if not os.path.exists(os.path.join(SEG_DIR, s))]
    if missing:
        print(f"Missing segments: {missing}")
        sys.exit(1)

    list_path = os.path.join(SEG_DIR, "concat_list.txt")
    with open(list_path, "w") as f:
        for s in SEGMENTS:
            f.write(f"file '{s}'\n")

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy", OUT,
    ]
    subprocess.run(cmd, check=True, cwd=SEG_DIR)
    size_mb = os.path.getsize(OUT) / 1e6
    print(f"[Concat] Saved {OUT} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
