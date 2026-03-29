"""
run_filter.py — Evidence Filtering Pipeline
=============================================
Scans a folder of images, detects weapons, and copies only the
relevant images to an output folder. Also generates a CSV report
and summary statistics.

Usage:
    python run_filter.py
    python run_filter.py --classes knife --annotate
    python run_filter.py --input input/ --output output/ --annotate
"""

import argparse
import csv
import os
import shutil
from datetime import datetime

import cv2
from ultralytics import YOLO

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "yolov8n.pt"
DEFAULT_INPUT_DIR  = "input"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_CONF       = 0.35
# Classes to look for by default (COCO names that relate to weapons)
DEFAULT_CLASSES    = ["knife"]          # extend when using a custom model

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def draw_boxes(img, detections, model_names):
    """Draw bounding boxes and labels onto img (in-place). Returns img."""
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label = det["label"]
        conf  = det["confidence"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        text = f"{label}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 200, 0), -1)
        cv2.putText(img, text, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return img


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_filter(args):
    print("=" * 60)
    print("  AWDEFS — Image Filtering Pipeline")
    print("=" * 60)
    print(f"  Input   : {args.input}")
    print(f"  Output  : {args.output}")
    print(f"  Classes : {', '.join(args.classes)}")
    print(f"  Conf    : {args.conf}")
    print(f"  Annotate: {args.annotate}")
    print("-" * 60)

    model = YOLO(args.model)

    # Collect images
    image_paths = [
        os.path.join(args.input, f)
        for f in os.listdir(args.input)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ] if os.path.isdir(args.input) else []

    if not image_paths:
        print(f"  [ERROR] No images found in: {args.input}")
        return

    # Prepare output dirs
    filtered_dir  = os.path.join(args.output, "filtered_images")
    annotated_dir = os.path.join(args.output, "annotated_images")
    os.makedirs(filtered_dir, exist_ok=True)
    if args.annotate:
        os.makedirs(annotated_dir, exist_ok=True)

    total        = len(image_paths)
    kept         = 0
    all_dets     = []           # for CSV report
    class_counts = {}

    print(f"\n  Scanning {total} image(s)...\n")

    for i, path in enumerate(image_paths, 1):
        fname = os.path.basename(path)
        img   = cv2.imread(path)
        if img is None:
            print(f"  [{i}/{total}] SKIP  {fname}  (unreadable)")
            continue

        results    = model(img, conf=args.conf, verbose=False)[0]
        match_dets = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label  = model.names[cls_id]
            conf   = float(box.conf[0])

            # Keep detection only if it matches a requested class
            if args.classes and label not in args.classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            det = {"image": fname, "label": label, "confidence": round(conf, 3),
                   "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            match_dets.append(det)
            all_dets.append(det)
            class_counts[label] = class_counts.get(label, 0) + 1

        if match_dets:
            kept += 1
            shutil.copy(path, os.path.join(filtered_dir, fname))

            if args.annotate:
                annotated = draw_boxes(img.copy(), match_dets, model.names)
                cv2.imwrite(os.path.join(annotated_dir, fname), annotated)

            labels_str = ", ".join(f"{d['label']} ({d['confidence']:.0%})"
                                   for d in match_dets)
            print(f"  [{i}/{total}] KEEP  {fname}  →  {labels_str}")
        else:
            print(f"  [{i}/{total}] SKIP  {fname}")

    # ── CSV report ────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.output, "detection_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image","label","confidence",
                                               "x1","y1","x2","y2"])
        writer.writeheader()
        writer.writerows(all_dets)

    # ── Text summary ──────────────────────────────────────────────────────────
    summary_path = os.path.join(args.output, "summary_report.txt")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(summary_path, "w") as f:
        lines = [
            "=" * 60,
            "  AWDEFS — Image Filtering Summary Report",
            "=" * 60,
            f"  Generated : {now}",
            f"  Model     : {args.model}",
            "",
            "  STATISTICS",
            "  " + "-" * 30,
            f"  Total images scanned  : {total}",
            f"  Images kept (matched) : {kept}",
            f"  Images skipped        : {total - kept}",
            f"  Filtering rate        : {kept/total*100:.1f}%",
            "",
            "  PER-CLASS DETECTIONS",
            "  " + "-" * 30,
        ]
        for label, count in class_counts.items():
            lines.append(f"  {label:<25}: {count}")
        lines += [
            "",
            "  OUTPUT FILES",
            "  " + "-" * 30,
            f"  Filtered images  → {filtered_dir}",
        ]
        if args.annotate:
            lines.append(f"  Annotated images → {annotated_dir}")
        lines += [
            f"  CSV report       → {csv_path}",
            f"  Summary report   → {summary_path}",
            "=" * 60,
        ]
        f.write("\n".join(lines))

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  Scanned  : {total} images")
    print(f"  Kept     : {kept} images")
    print(f"  Skipped  : {total - kept} images")
    for label, count in class_counts.items():
        print(f"    {label}: {count} detection(s)")
    print(f"\n  Reports saved to: {args.output}/")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AWDEFS Evidence Filtering Pipeline")
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--input",    default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output",   default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conf",     type=float, default=DEFAULT_CONF)
    parser.add_argument("--classes",  nargs="+", default=DEFAULT_CLASSES,
                        help="Object class names to filter for (space-separated)")
    parser.add_argument("--annotate", action="store_true",
                        help="Save annotated images with bounding boxes")
    args = parser.parse_args()

    run_filter(args)
