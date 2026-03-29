import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AWDEFS – Weapon Detection",
    page_icon="🔫",
    layout="wide",
)

# ── Load model (cached so it only loads once) ─────────────────────────────────
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")   # auto-downloads on first run

# ── Draw boxes using PIL only (no cv2 needed) ─────────────────────────────────
def draw_boxes_pil(image_pil, detections):
    img = image_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label = det["label"]
        conf  = det["confidence"]
        text  = f"{label}  {conf:.0%}"
        draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3)
        text_bbox = draw.textbbox((x1, y1), text)
        draw.rectangle(
            [x1, y1 - (text_bbox[3] - text_bbox[1]) - 8,
             x1 + (text_bbox[2] - text_bbox[0]) + 8, y1],
            fill="#00FF00"
        )
        draw.text((x1 + 4, y1 - (text_bbox[3] - text_bbox[1]) - 4),
                  text, fill="black")
    return img

# ── Run detection ─────────────────────────────────────────────────────────────
def run_detection(image_pil, conf_threshold=0.35):
    model = load_model()
    img_np = np.array(image_pil.convert("RGB"))
    results = model(img_np, conf=conf_threshold, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf   = float(box.conf[0])
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        detections.append({
            "label": label,
            "confidence": round(conf, 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    annotated = draw_boxes_pil(image_pil, detections)
    return annotated, detections

def pil_to_bytes(img_pil, fmt="PNG"):
    buf = io.BytesIO()
    img_pil.save(buf, format=fmt)
    return buf.getvalue()

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("## 🔫 Automated Weapon Detection & Evidence Filtering System")
st.markdown(
    "Upload a crime scene image to detect weapons automatically using **YOLOv8**. "
    "Bounding boxes and confidence scores are drawn on the output."
)
st.divider()

with st.sidebar:
    st.header("⚙️ Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.95, 0.35, 0.05,
        help="Lower = more detections. Higher = only certain detections.")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Model: YOLOv8-nano (COCO pretrained)")
    st.markdown("Built for forensic evidence triage")
    st.markdown("NFSU Delhi — Anshul Dhanjwal")

uploaded_file = st.file_uploader(
    "📁 Upload a crime scene image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("🖼️ Original Image")
        st.image(image, use_column_width=True)
        st.caption(f"Size: {image.width} × {image.height} px")
    with st.spinner("Running weapon detection…"):
        annotated_img, detections = run_detection(image, conf_threshold=conf_thresh)
    with col2:
        st.subheader("🔍 Detection Result")
        st.image(annotated_img, use_column_width=True)
        st.caption(f"{len(detections)} object(s) detected")
    st.divider()
    if detections:
        st.subheader(f"✅ {len(detections)} Detection(s) Found")
        cols = st.columns(min(len(detections), 4))
        for i, det in enumerate(detections):
            with cols[i % 4]:
                st.metric(label=det["label"].upper(),
                          value=f"{det['confidence']:.0%}",
                          delta="Detected", delta_color="off")
        st.markdown("**Detection Details:**")
        st.dataframe(detections, use_container_width=True)
    else:
        st.info("No objects detected. Try lowering the threshold in the sidebar.")
    st.download_button(
        label="⬇️ Download Annotated Image",
        data=pil_to_bytes(annotated_img),
        file_name="detection_result.png",
        mime="image/png",
    )
else:
    st.info("👆 Upload an image above to start detection.")
    st.markdown("""
    **How it works:**
    1. Upload any JPG or PNG image
    2. YOLOv8 scans for weapons and objects
    3. Bounding boxes and confidence scores appear on the result
    4. Download the annotated image
    """)
