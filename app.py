import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from src.pipelines.predicting_pipeline import Predictor

# Loading model 
CKPT_PATH = "models/checkpoints/brain-tumor-epoch=03-val_f1=0.94.ckpt" 
# IMAGE_TO_TEST = "test/test_image.webp" # Path to an MRI image
IMAGE_TO_TEST = "test/test_no_tumor.webp" # Path to an MRI image
predictor = Predictor(CKPT_PATH)



st.title("Brain Tumor Detection")
st.write("Upload an MRI image to classify the tumor type.")

upload_file = st.file_uploader(
    "Choose an MRI image...", 
    type=["jpg", "jpeg", "png", "webp"],
     accept_multiple_files=False
)

if upload_file is not None:
    image = Image.open(upload_file)
    st.image(image, caption="Uploaded MRI Image", width=400)
    
    if st.button("Predict"):
        with st.spinner("Classifying..."):
            result = predictor.predict_image(image)
        
        st.title(f"{result['prediction']}")
        st.write(f"**Confidence:** {result['confidence']*100:.2f}%")
        st.write("**Probabilities:**")
        for cls, prob in result["probabilities"].items():
            st.write(f"- {cls.upper()}: {prob*100:.2f}%")