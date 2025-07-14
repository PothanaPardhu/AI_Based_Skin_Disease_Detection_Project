import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
from streamlit_lottie import st_lottie

# Page Configuration
st.set_page_config(page_title="Skin Health Analyzer", layout="centered", page_icon="💠")

# Elegant Custom CSS
st.markdown("""
    <style>
        body {
            margin: 0;
            padding: 0;
        }
        .stApp {
            background: linear-gradient(to right, #c9d6ff, #e2e2e2);
            font-family: 'Segoe UI', sans-serif;
            color: #2c3e50;
        }
        h1, h2, h3 {
            color: #2c3e50 !important;
        }
        .box {
            background-color: #ffffffcc;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-top: 1rem;
        }
        .stButton>button {
            background-color: #4e8cff;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
            .thickcolor{
            background-color : #000000;
           }
    </style>
""", unsafe_allow_html=True)

# Load model
model = load_model('skin_health_model.h5')

# Skin condition labels
class_names = ['acne', 'dryness', 'eczema', 'hyperpigmentation', 'healthy']

# Lottie animation loader
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animation
lottie_skin = load_lottieurl("https://lottie.host/9f94a07a-57a4-4b53-85a9-b4dca56f2965/9Qf0drMlBt.json")

# Sidebar
with st.sidebar:
    st.title("📘 About This App")
    st.write("An AI-based skin health analyzer that detects:")
    st.markdown("""
    - Acne  
    - Dryness  
    - Eczema  
    - Hyperpigmentation  
    - Healthy skin  
    """)
    st.write("Upload a skin image to get a prediction and helpful skincare tips.")
    st.markdown("---")
    st.write("💡 Created by *Pothana Pardhu*")

# Title & Intro
st.title("AI-Based Skin Disease Detector")
st.write("Upload a skin image and detect conditions using Deep Learning!")

# Lottie animation
if lottie_skin:
    st_lottie(lottie_skin, height=200, key="skin")

# Image uploader
uploaded_file = st.file_uploader("📤 Upload an image (JPG, JPEG, PNG)", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((224, 224))
    st.image(img, caption="📷 Uploaded Image", use_column_width=True)

    # Image Preprocessing
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    with st.spinner("Analyzing your skin..."):
        preds = model.predict(x)
        predicted_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100

    # Show Prediction
    st.write(f"## Predicted skin Disease: *{predicted_class.capitalize()}*")
    st.markdown(f"📊 Confidence: *{confidence:.2f}%*")
    st.progress(float(min(confidence / 100, 1.0)))  # ✅ Fixed float32 error

    # Show Tips
    st.markdown('<h3 style="color: red;">Detailed skin care tips</h3>', unsafe_allow_html=True)
    if st.toggle(""):
        def show_tips(causes, precautions, medications):
            st.markdown(f"<div class='box'><strong>❗ Causes:</strong><br>{causes}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='box'><strong>🛡 Precautions:</strong><br>{precautions}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='box'><strong>💊 Medications:</strong><br>{medications}</div>", unsafe_allow_html=True)

        if predicted_class == 'acne':
            show_tips(
                causes="Hormonal imbalance, clogged pores, excess sebum, bacteria.",
                precautions="Cleanse gently, avoid harsh scrubs, stay hydrated.",
                medications="Salicylic acid, benzoyl peroxide, retinoids."
            )
        elif predicted_class == 'dryness':
            show_tips(
                causes="Cold weather, dehydration, harsh soaps, over-cleansing.",
                precautions="Use gentle moisturizer, avoid hot water, drink water.",
                medications="Urea-based creams, ceramides, petroleum jelly."
            )
        elif predicted_class == 'eczema':
            show_tips(
                causes="Overactive immune system, allergens, genetics.",
                precautions="Avoid irritants, moisturize frequently, reduce stress.",
                medications="Steroid creams, antihistamines, emollients."
            )
        elif predicted_class == 'hyperpigmentation':
            show_tips(
                causes="Sun exposure, acne scarring, hormonal changes.",
                precautions="Use sunscreen daily, avoid picking skin.",
                medications="Vitamin C, niacinamide, hydroquinone (under supervision)."
            )
        elif predicted_class == 'healthy':
            show_tips(
                causes="No visible skin issues detected!",
                precautions="Stay hydrated, maintain skincare routine.",
                medications="None needed — keep your skin glowing!"
            )