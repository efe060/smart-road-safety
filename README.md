# Smart Road Safety: Video Analysis Module

Vehicle detection, tracking and color recognition for traffic video. This is the computer-vision module our team **BEATECH-G** built for the **TEKNOFEST 2026 "5G and AI for Smart Road Safety"** competition, where we reached the **semi-finals**.

> 🇹🇷 TEKNOFEST 2026 "5G ve Yapay Zeka ile Akıllı Yol Güvenliği" yarışması için ekibimiz BEATECH-G'nin geliştirdiği video analiz modülü (yarı finalist).

## What it does

Give it a traffic video and it writes a JSON file describing every vehicle it saw:

```
video ──► frame sampling ──► YOLOv8 detection ──► ByteTrack IDs ──► per-vehicle summary ──► results.json
          (every n-th frame)   (car, motorcycle,     (same car keeps     (type vote, color vote,
                                bus, truck)           the same ID)        first/last seen)
```

- **Detection:** YOLOv8n on the COCO vehicle classes (car, motorcycle, bus, truck)
- **Tracking:** ByteTrack, so a vehicle is counted once and its type/color are decided by majority vote over all frames instead of a single noisy frame
- **Color:** HSV pixel voting on the center of each box (see [`src/colors.py`](src/colors.py)). Averaging RGB mixes body paint with windows and road, so this is more reliable
- **Main vehicle:** the one that appears largest in the frame is reported as `arac_bilgisi`

## Output

```json
{
  "video_id": "video.mp4",
  "video": { "fps": 25.0, "kare_sayisi": 100, "sure_s": 4.0, "analiz_edilen_her_n_kare": 3 },
  "arac_bilgisi": {
    "track_id": 1, "tip": "otobus", "renk": "mavi",
    "plaka": null, "confidence_score": 0.91
  },
  "araclar": [
    { "track_id": 1, "tip": "otobus", "renk": "mavi",
      "ilk_gorulme_s": 0.0, "son_gorulme_s": 3.96,
      "confidence_score": 0.91, "bbox_xyxy": [0, 229, 794, 774], "tespit_sayisi": 34 }
  ],
  "tespitler": [
    { "zaman_saniye": 0.0, "kategori": "arac", "etiket": "otobus",
      "track_id": 1, "confidence_score": 0.91 }
  ]
}
```

## Usage

```bash
pip install -r requirements.txt
python main.py --input traffic.mp4 --output results.json
```

Useful options: `--every-n 3` (analyze every n-th frame), `--conf 0.4` (detection threshold), `--device cpu|0`, `--weights your_model.pt`. The default `yolov8n.pt` is downloaded automatically on first run.

### Docker (GPU)

```bash
docker build -t beatech-g .
docker run --gpus all \
  -v $(pwd)/video.mp4:/app/data/input/video.mp4 \
  -v $(pwd)/out:/app/data/output \
  beatech-g
```

## Project structure

```
├── main.py          # command line entry point, writes the JSON
├── src/
│   ├── predict.py   # detection + tracking + per-vehicle aggregation
│   └── colors.py    # HSV-voting color classifier
├── Dockerfile
└── requirements.txt
```

## Limitations

This repository is the video-analysis part of a larger system, and it is honest about what is **not** here:

- **No license plate reading.** `plaka` is always `null`.
- **No in-cabin analysis** (phone or cigarette use, passengers). That needs a custom-trained model; the stock COCO weights cannot do it.
- **Vehicle types are the four COCO classes.** Finer types (sedan, hatchback, minibus) need fine-tuning on a vehicle dataset.
- **The 5G parts are not in this code.** The competition design report describes network-aware behavior (Turkcell Open Gateway Number Verification and Quality on Demand APIs, MEC deployment, a Flutter app). None of that is implemented here.
- Color recognition is rule-based, so strong shadows, night footage or heavy reflections can fool it. It has been unit-tested on solid colors, not benchmarked on a labeled dataset.

## Roadmap

- [ ] Fine-tune a detector on Turkish road footage with finer vehicle types
- [ ] License plate detection + OCR
- [ ] Benchmark the color classifier on a labeled set
- [ ] Export to ONNX/TensorRT for edge deployment

## Tech

Python · Ultralytics YOLOv8 · ByteTrack · OpenCV · Docker
