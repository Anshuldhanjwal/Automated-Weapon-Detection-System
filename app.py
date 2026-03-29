import csv
import io
import zipfile

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AWDEFS – Weapon Detection",
    page_icon="🔫",
    layout="wide",
)

# ── Model loaders (cached — download once, reuse forever) ──────────────────────
@st.cache_resource
def load_gun_model():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    model_path = hf_hub_download(
        repo_id="Subh775/Firearm_Detection_Yolov8n",
        filename="weights/best.pt",
    )
    return YOLO(model_path)

@st.cache_resource
def load_knife_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")   # COCO — used only for knife class

# ── Core helpers ───────────────────────────────────────────────────────────────
def run_detection(image_pil, conf_threshold=0.35):
    """Return (annotated_PIL, detections_list)."""
    img_np = np.array(image_pil.convert("RGB"))
    detections = []

    # Firearm detection
    for box in load_gun_model()(img_np, conf=conf_threshold, verbose=False)[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append({
            "label": "Gun",
            "confidence": round(float(box.conf[0]), 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })

    # Knife detection (COCO, filtered)
    knife_model = load_knife_model()
    for box in knife_model(img_np, conf=conf_threshold, verbose=False)[0].boxes:
        if knife_model.names[int(box.cls[0])] == "knife":
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "label": "Knife",
                "confidence": round(float(box.conf[0]), 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })

    return draw_boxes_pil(image_pil, detections), detections


def draw_boxes_pil(image_pil, detections):
    img  = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        text      = f"{det['label']}  {det['confidence']:.0%}"
        draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3)
        tb = draw.textbbox((x1, y1), text)
        draw.rectangle([x1, y1-(tb[3]-tb[1])-8, x1+(tb[2]-tb[0])+8, y1], fill="#00FF00")
        draw.text((x1+4, y1-(tb[3]-tb[1])-4), text, fill="black")
    return img


def pil_to_bytes(img_pil, fmt="PNG"):
    buf = io.BytesIO()
    img_pil.save(buf, format=fmt)
    return buf.getvalue()


def make_csv(rows):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["image", "label", "confidence", "x1", "y1", "x2", "y2"])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode()


def make_zip(annotated_images: dict[str, bytes]) -> bytes:
    """Pack {filename: png_bytes} into a ZIP and return the bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in annotated_images.items():
            zf.writestr(name, data)
    return buf.getvalue()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    conf_thresh = st.slider(
        "Confidence Threshold", 0.10, 0.95, 0.35, 0.05,
        help="How certain the model must be before flagging a detection.\n"
             "Lower → more detections (risk of false positives).\n"
             "Higher → fewer, more certain detections only.",
    )
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("🔫 Gun model: YOLOv8n fine-tuned — 89% mAP")
    st.markdown("🔪 Knife model: YOLOv8n COCO")
    st.markdown("Built for forensic evidence triage")
    st.markdown("NFSU Delhi — Anshul Dhanjwal")

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("## 🔫 Automated Weapon Detection & Evidence Filtering System")
st.markdown(
    "AI-powered forensic tool using **YOLOv8** to detect firearms and knives "
    "in crime scene images — reducing manual triage time by up to **99.5%**."
)
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Image Detection", "📁 Batch Evidence Filter"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single image
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    uploaded_file = st.file_uploader(
        "📁 Upload a crime scene image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="single",
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            image.verify()
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"⚠️ Could not read image: {e}")
            st.stop()

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.subheader("🖼️ Original Image")
            st.image(image, use_container_width=True)
            st.caption(f"Size: {image.width} × {image.height} px")

        with st.spinner("Running weapon detection…"):
            try:
                annotated_img, detections = run_detection(image, conf_threshold=conf_thresh)
            except Exception as e:
                st.error(f"⚠️ Detection failed: {e}")
                st.stop()

        with col2:
            st.subheader("🔍 Detection Result")
            st.image(annotated_img, use_container_width=True)
            st.caption(f"{len(detections)} weapon(s) detected")

        st.divider()

        if detections:
            st.subheader(f"✅ {len(detections)} Detection(s) Found")
            cols = st.columns(min(len(detections), 4))
            for i, det in enumerate(detections):
                with cols[i % 4]:
                    st.metric(det["label"].upper(), f"{det['confidence']:.0%}",
                              delta="Detected", delta_color="off")
            st.dataframe(detections, use_container_width=True)
        else:
            st.info("No weapons detected. Try lowering the confidence threshold in the sidebar.")

        st.download_button(
            "⬇️ Download Annotated Image",
            data=pil_to_bytes(annotated_img),
            file_name="detection_result.png",
            mime="image/png",
        )
    else:
        st.info("👆 Upload an image above to start detection.")
        st.markdown("""
        **How it works:**
        1. Upload any JPG or PNG image
        2. A fine-tuned YOLOv8 model scans for firearms
        3. A second model scans for knives
        4. Bounding boxes and confidence scores appear on the result
        5. Download the annotated image
        """)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch filter
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📁 Batch Evidence Filter")
    st.markdown(
        "Upload multiple crime scene images at once. "
        "The system will scan every image and show only the ones containing weapons — "
        "exactly like the command-line `run_filter.py` pipeline, but in your browser."
    )

    uploaded_files = st.file_uploader(
        "📂 Upload crime scene images (select multiple)",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="batch",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} image(s) queued for processing.**")
        run_btn = st.button("🚀 Run Batch Detection", type="primary")

        if run_btn:
            all_detections  = []
            flagged_images  = {}
            summary_rows    = []

            progress = st.progress(0, text="Starting…")
            total    = len(uploaded_files)

            for i, f in enumerate(uploaded_files):
                progress.progress((i) / total, text=f"Scanning {f.name}…")
                try:
                    img = Image.open(f)
                    img.verify()
                    f.seek(0)
                    img = Image.open(f)
                except Exception:
                    summary_rows.append({"Image": f.name, "Weapons Found": 0,
                                         "Labels": "—", "Status": "❌ Unreadable"})
                    continue

                annotated, dets = run_detection(img, conf_threshold=conf_thresh)

                for d in dets:
                    all_detections.append({
                        "image": f.name, "label": d["label"],
                        "confidence": d["confidence"],
                        "x1": d["x1"], "y1": d["y1"],
                        "x2": d["x2"], "y2": d["y2"],
                    })

                if dets:
                    flagged_images[f"annotated_{f.name}"] = pil_to_bytes(annotated)
                    label_str = ", ".join(
                        f"{d['label']} ({d['confidence']:.0%})" for d in dets
                    )
                    summary_rows.append({"Image": f.name,
                                         "Weapons Found": len(dets),
                                         "Labels": label_str,
                                         "Status": "🚨 FLAGGED"})
                else:
                    summary_rows.append({"Image": f.name, "Weapons Found": 0,
                                         "Labels": "—", "Status": "✅ Clean"})

            progress.progress(1.0, text="Done!")

            # ── Stats ──────────────────────────────────────────────────────────
            st.divider()
            flagged_count = len(flagged_images)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📸 Images Scanned",   total)
            c2.metric("🚨 Flagged",          flagged_count)
            c3.metric("✅ Clean",            total - flagged_count)
            c4.metric("🔍 Total Detections", len(all_detections))

            # ── Summary table ──────────────────────────────────────────────────
            st.subheader("📋 Results Summary")
            st.dataframe(summary_rows, use_container_width=True)

            # ── Flagged image previews ─────────────────────────────────────────
            if flagged_images:
                st.subheader(f"🚨 {flagged_count} Flagged Image(s)")
                cols = st.columns(min(flagged_count, 3))
                for idx, (fname, img_bytes) in enumerate(flagged_images.items()):
                    with cols[idx % 3]:
                        st.image(img_bytes, caption=fname.replace("annotated_", ""),
                                 use_container_width=True)

            # ── Downloads ──────────────────────────────────────────────────────
            st.divider()
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️ Download CSV Report",
                    data=make_csv(all_detections),
                    file_name="detection_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with dl2:
                if flagged_images:
                    st.download_button(
                        "⬇️ Download Flagged Images (ZIP)",
                        data=make_zip(flagged_images),
                        file_name="flagged_images.zip",
                        mime="application/zip",
                        use_container_width=True,
                    )
    else:
        st.info("👆 Upload multiple images above, then click **Run Batch Detection**.")
        st.markdown("""
        **What this does:**
        - Scans every uploaded image for firearms and knives
        - Shows a summary table — flagged vs clean
        - Displays annotated previews of all flagged images
        - Lets you download a **CSV report** with all detection coordinates
        - Lets you download all **annotated flagged images** as a ZIP
        """)
