"""
detect.py — Weapon Detection Script
====================================
Run YOLOv8 detection on a single image or a folder of images.

Usage:
    python detect.py                        # uses default test.jpg
    python detect.py --source myimage.jpg
    python detect.py --source input/        # folder of images
    python detect.py --source input/ --conf 0.4 --output results/
"""

import argparse
import os
from collections import Counter

import cv2
from ultralytics import YOLO


# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "yolov8n.pt"       # auto-downloads on first run
DEFAULT_CONF   = 0.35
DEFAULT_SOURCE = "test.jpg"
DEFAULT_OUTPUT = "output/predictions"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Core detection function ───────────────────────────────────────────────────
def detect_image(model, image_path: str, conf: float, output_dir: str) -> list:
    """
    Run detection on one image. Saves annotated result to output_dir.
    Returns a list of detection dicts.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [SKIP] Could not read: {image_path}")
        return []

    results = model(img, conf=conf, verbose=False)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id   = int(box.cls[0])
        label      = model.names[class_id]

        # Draw green bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label background + text
        text = f"{label}  {confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(img, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        detections.append({
            "image":      os.path.basename(image_path),
            "label":      label,
            "confidence": round(confidence, 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    # Save annotated image
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, img)

    status = f"  ✅ {len(detections)} detection(s)" if detections else "  — no detections"
    print(f"  [{os.path.basename(image_path)}]{status}")

    return detections


def collect_images(source: str) -> list[str]:
    """Return a sorted list of supported image paths from a file or directory."""
    if os.path.isfile(source):
        return [source]
    if os.path.isdir(source):
        return sorted(
            os.path.join(source, f)
            for f in os.listdir(source)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
        )
    return []


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Weapon Detection with YOLOv8")
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Path to image file or folder of images")
    parser.add_argument("--model",  default=DEFAULT_MODEL,
                        help="YOLOv8 model weights (default: yolov8n.pt)")
    parser.add_argument("--conf",   type=float, default=DEFAULT_CONF,
                        help="Confidence threshold 0.0–1.0 (default: 0.35)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Folder to save annotated images")
    args = parser.parse_args()

    print("=" * 55)
    print("  AWDEFS — Automated Weapon Detection System")
    print("=" * 55)
    print(f"  Model  : {args.model}")
    print(f"  Source : {args.source}")
    print(f"  Conf   : {args.conf}")
    print(f"  Output : {args.output}")
    print("-" * 55)

    # Load model
    model = YOLO(args.model)

    # Collect image paths (sorted for deterministic ordering)
    image_paths = collect_images(args.source)

    if not image_paths:
        print(f"  [ERROR] No supported images found at: {args.source}")
        return

    print(f"  Processing {len(image_paths)} image(s)...\n")

    # Run detection on all images
    all_detections = []
    for path in image_paths:
        dets = detect_image(model, path, args.conf, args.output)
        all_detections.extend(dets)

    # Print summary
    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  Images processed  : {len(image_paths)}")
    print(f"  Total detections  : {len(all_detections)}")
    if all_detections:
        counts = Counter(d["label"] for d in all_detections)
        for label, count in counts.most_common():
            print(f"    {label:<20}: {count}")
    print(f"  Annotated images saved to: {args.output}/")
    print("=" * 55)


if __name__ == "__main__":
    main()
