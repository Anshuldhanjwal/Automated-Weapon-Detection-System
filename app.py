import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO("yolov8n.pt")

st.title("🔫 Weapon Detection System")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
   img = Image.open(uploaded_file)

    results = model(img)
    annotated = results[0].plot()

    st.image(annotated, caption="Detection Result", use_column_width=True)
