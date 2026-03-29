import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AWDEFS – Weapon Detection",
    page_icon="🔫",
    layout="wide",
)

# ── Load model (cached so it only loads once) ────────────────────────────────
@st.cache_resource
def load_model():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")   # downloads automatically on first run

# ── Helper: run detection and draw boxes ─────────────────────────────────────
def run_detection(image_pil, conf_threshold=0.35):
    """
    Run YOLOv8 on a PIL image.
    Returns (annotated PIL image, list of detection dicts).
    """
    model = load_model()

    # Convert PIL → numpy BGR for OpenCV
    img_np = np.array(image_pil.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    results = model(img_bgr, conf=conf_threshold, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf  = float(box.conf[0])
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]

        # Draw bounding box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(img_bgr, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        detections.append({"label": label, "confidence": round(conf, 3),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2})

    # Convert back to PIL for Streamlit
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb), detections


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

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.95, 0.35, 0.05,
                            help="Lower = more detections (more false positives). "
                                 "Higher = fewer, more certain detections.")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("Model: YOLOv8-nano (COCO pretrained)")
    st.markdown("Built for forensic evidence triage")
    st.markdown("NFSU Delhi — Anshul Dhanjwal")

# File uploader
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

    # Detection summary
    if detections:
        st.subheader(f"✅ {len(detections)} Detection(s) Found")
        cols = st.columns(min(len(detections), 4))
        for i, det in enumerate(detections):
            with cols[i % 4]:
                st.metric(
                    label=det["label"].upper(),
                    value=f"{det['confidence']:.0%}",
                    delta="Detected",
                    delta_color="off",
                )
        st.markdown("**Detection Details:**")
        st.dataframe(detections, use_container_width=True)
    else:
        st.info("No objects detected above the confidence threshold. "
                "Try lowering the threshold in the sidebar.")

    # Download button
    st.download_button(
        label="⬇️ Download Annotated Image",
        data=pil_to_bytes(annotated_img),
        file_name="detection_result.png",
        mime="image/png",
    )

else:
    # Placeholder when no image is uploaded
    st.info("👆 Upload an image above to start detection.")
    st.markdown(
        """
        **How it works:**
        1. Upload any JPG or PNG image
        2. The YOLOv8 model scans for weapons and other objects
        3. Bounding boxes and confidence scores are drawn on the result
        4. Download the annotated image or view detection details below
        """
    )
