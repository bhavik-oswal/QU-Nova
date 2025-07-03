# app.py
import streamlit as st
import numpy as np
from scripts.model import load_brats_model
from scripts.predict import predict_segmentation
from scripts.utils import display_slices, generate_pdf_report

st.title("BraTS Tumor Segmentation - 3D U-Net")

# Upload input
uploaded_file = st.file_uploader("Upload a .npy file (BraTS Volume)", type=["npy"])
model = load_brats_model("brats_3d.hdf5")

if uploaded_file is not None:
    volume = np.load(uploaded_file)
    st.write(f"Input shape: {volume.shape}")

    # run segmentation exactly once per upload
    prediction = predict_segmentation(model, volume)

    st.subheader("Visualization")
    display_slices(volume, prediction)

    if st.button("Generate PDF Report"):
        pdf_path = generate_pdf_report(volume, prediction)
        with open(pdf_path, "rb") as f:
            st.download_button("Download Report", f, "BraTS_Segmentation_Report.pdf")
