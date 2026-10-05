"""Vehicle analysis for a single traffic video.

Pipeline
--------
frame sampling -> YOLOv8 detection (COCO vehicle classes) -> ByteTrack IDs
-> per-vehicle aggregation (type vote, color vote, first/last seen, best confidence)
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.colors import dominant_color

# COCO class id -> vehicle type (Turkish labels used in the competition output)
VEHICLE_CLASSES = {2: "otomobil", 3: "motosiklet", 5: "otobus", 7: "kamyon"}


@dataclass
class VehicleTrack:
    track_id: int
    first_seen: float
    last_seen: float
    best_conf: float = 0.0
    best_area: int = 0
    best_box: list[int] = field(default_factory=list)
    types: Counter = field(default_factory=Counter)
    colors: Counter = field(default_factory=Counter)
    hits: int = 0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "tip": self.types.most_common(1)[0][0],
            "renk": self.colors.most_common(1)[0][0] if self.colors else None,
            "ilk_gorulme_s": round(self.first_seen, 2),
            "son_gorulme_s": round(self.last_seen, 2),
            "confidence_score": round(self.best_conf, 2),
            "bbox_xyxy": self.best_box,
            "tespit_sayisi": self.hits,
        }


def analyze_video(video_path: str | Path, weights: str = "yolov8n.pt", every_n: int = 3,
                  conf: float = 0.4, device: str | None = None) -> dict:
    """Analyze one video and return the competition-style result dictionary."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video could not be opened: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model = YOLO(weights)

    tracks: dict[int, VehicleTrack] = {}
    untracked = 0          # detections the tracker had not confirmed yet
    idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % every_n:
            continue

        t = idx / fps
        result = model.track(frame, persist=True, conf=conf, classes=list(VEHICLE_CLASSES),
                             tracker="bytetrack.yaml", device=device, verbose=False)[0]
        if result.boxes is None or result.boxes.id is None:
            untracked += 0 if result.boxes is None else len(result.boxes)
            continue

        h, w = frame.shape[:2]
        for box, cls_id, score, tid in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                           result.boxes.conf.tolist(), result.boxes.id.tolist()):
            tid = int(tid)
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            track = tracks.setdefault(tid, VehicleTrack(track_id=tid, first_seen=t, last_seen=t))
            track.last_seen = t
            track.hits += 1
            track.types[VEHICLE_CLASSES[int(cls_id)]] += 1
            color = dominant_color(frame[y1:y2, x1:x2])
            if color:
                track.colors[color] += 1
            area = (x2 - x1) * (y2 - y1)
            if score > track.best_conf:
                track.best_conf = float(score)
            if area > track.best_area:
                track.best_area, track.best_box = area, [x1, y1, x2, y2]

    cap.release()

    vehicles = sorted(tracks.values(), key=lambda tr: tr.first_seen)
    # The "main" vehicle is the one that appears largest in the frame.
    main = max(vehicles, key=lambda tr: tr.best_area, default=None)

    return {
        "video_id": video_path.name,
        "video": {
            "fps": round(fps, 2),
            "kare_sayisi": frame_count,
            "sure_s": round(frame_count / fps, 2) if frame_count else None,
            "analiz_edilen_her_n_kare": every_n,
        },
        "arac_bilgisi": None if main is None else {
            "track_id": main.track_id,
            "tip": main.to_dict()["tip"],
            "renk": main.to_dict()["renk"],
            "plaka": None,  # plate recognition is not implemented in this version
            "confidence_score": round(main.best_conf, 2),
        },
        "araclar": [v.to_dict() for v in vehicles],
        "tespitler": [
            {
                "zaman_saniye": round(v.first_seen, 2),
                "kategori": "arac",
                "etiket": v.to_dict()["tip"],
                "track_id": v.track_id,
                "confidence_score": round(v.best_conf, 2),
            }
            for v in vehicles
        ],
    }
