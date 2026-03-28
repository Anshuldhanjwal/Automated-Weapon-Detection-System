import streamlit as st
from PIL import Image

st.title("🔫 Weapon Detection System")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Simulated output (replace with your saved result image)
    st.subheader("Detection Result")
    result = Image.open("annotated_result.png")
    st.image(result, caption="Detected Output", use_column_width=True)
