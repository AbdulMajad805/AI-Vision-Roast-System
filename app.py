import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import os

# 🔐 Secure API Key
HF_API_KEY = os.getenv("HF_API_KEY")

API_URL = "https://api-inference.huggingface.co/models/gpt2"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}


# 🤖 AI Roast Function
def ai_roast(caption):
    prompt = f'''
You are a funny and savage internet roaster.
Roast this person based on the description below in one short line.

Description: {caption}

Roast:
'''

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 60,
            "temperature": 0.9,
            "do_sample": True
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    result = response.json()

    if isinstance(result, dict) and "error" in result:
        return "Model is loading... try again in a few seconds 😅"

    return result[0]['generated_text']


# 📦 Load Model
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()


# 🎨 UI CONFIG
st.set_page_config(page_title="AI Roast System", layout="centered")

# 🎨 Custom CSS (Card Style)
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa;
    }
    .title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        margin-bottom: 25px;
    }
    .card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }
    .roast-box {
        background-color: #f1f3f6;
        padding: 15px;
        border-radius: 10px;
        font-size: 18px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 🔥 Title
st.markdown('<div class="title">AI Vision Roast System 😂🔥</div>', unsafe_allow_html=True)

# 📦 Card Start
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("🔥 Generate Roast"):
        with st.spinner("Analyzing image..."):
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

        st.markdown(f"**Caption:** {caption}")

        with st.spinner("Cooking roast 🔥..."):
            roast = ai_roast(caption)

        st.markdown("### 🔥 Roast Output")
        st.markdown(f'<div class="roast-box">{roast}</div>', unsafe_allow_html=True)

# 📦 Card End
st.markdown('</div>', unsafe_allow_html=True)
