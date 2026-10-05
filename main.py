"""Entry point: analyze a traffic video and write the result as JSON.

Local use:
    python main.py --input samples/traffic.mp4 --output results.json

Inside the Docker image the defaults below are used, so the container only needs
/app/data/input/video.mp4 to be mounted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.predict import analyze_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BEATECH-G smart road safety: video analysis")
    p.add_argument("--input", type=Path, default=Path("/app/data/input/video.mp4"))
    p.add_argument("--output", type=Path, default=Path("/app/data/output/results.json"))
    p.add_argument("--weights", default="yolov8n.pt",
                   help="YOLOv8 weights (downloaded automatically if missing)")
    p.add_argument("--every-n", type=int, default=3, help="analyze every n-th frame")
    p.add_argument("--conf", type=float, default=0.4, help="detection confidence threshold")
    p.add_argument("--device", default=None, help="'cpu', '0' for the first GPU, ...")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        sys.exit(f"Input video not found: {args.input}")

    print(f"Analyzing {args.input} ...")
    result = analyze_video(args.input, args.weights, every_n=args.every_n,
                           conf=args.conf, device=args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(result["araclar"])
    main_vehicle = result["arac_bilgisi"]
    summary = f"{main_vehicle['renk']} {main_vehicle['tip']}" if main_vehicle else "none"
    print(f"{n} vehicle(s) tracked, main vehicle: {summary}")
    print(f"Result written to {args.output}")


if __name__ == "__main__":
    main()
