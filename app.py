import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")

st.title("🔫 Weapon Detection System")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    results = model(img)
    annotated = results[0].plot()

    st.image(annotated, caption="Detection Result", use_column_width=True)
