import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
from groq import Groq

# 🔐 API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ Groq API key not found. Add it in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


# 🤖 AI Roast Function
def ai_roast(caption):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a funny and savage internet roaster. Keep it short, witty, and sarcastic."
                },
                {
                    "role": "user",
                    "content": f"Roast this: {caption}"
                }
            ],
            temperature=0.9,
            max_tokens=50
        )

        return response.choices[0].message.content

    except Exception:
        return "Something went wrong 😭"


# 📦 Load Caption Model
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


processor, model = load_model()


# 🎨 UI CONFIG
st.set_page_config(page_title="AI Roast System", layout="centered")

# 🎨 NOTEBOOK STYLE CSS
st.markdown("""
<style>
/* Background like paper */
body {
    background-color:#fdf3d7;
}

/* Main title */
.title {
    font-family: 'Courier New', monospace;
    font-size: 34px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 14px;
    margin-bottom: 15px;
}

/* Sections */
.section {
    font-weight: bold;
    margin-top: 15px;
    border-top: 2px dashed #999;
    padding-top: 8px;
}

/* Upload + camera layout */
.upload-box {
    border: 2px dashed #999;
    padding: 10px;
    text-align: center;
    margin-bottom: 10px;
}

/* Image preview */
.preview-box {
    border: 2px solid #333;
    height: 300px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #555;
}

/* Button style */
.stButton>button {
    width: 100%;
    background: linear-gradient(to right, #3b2f2f, #6b4f4f);
    color: white;
    font-size: 18px;
    border-radius: 5px;
    height: 50px;
}

/* Roast box */
.roast-box {
    border: 2px solid #333;
    padding: 15px;
    font-size: 18px;
    background-color: #fff;
    color: #111;
}

/* Copy button */
.copy-btn {
    display: flex;
    gap: 10px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


# 🧾 TITLE
st.markdown('<div class="title">AI Vision Roast System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">by Abdul Majid</div>', unsafe_allow_html=True)


# 📁 SECTION 1: Upload
st.markdown('<div class="section">1. Upload section</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

image = None

with col1:
    uploaded_file = st.file_uploader("Upload image (png / jpg / webp)", label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

with col2:
    camera_photo = st.camera_input("Take photo")
    if camera_photo:
        image = Image.open(camera_photo).convert("RGB")


# 🖼️ SECTION 2: Preview
st.markdown('<div class="section">2. Preview section</div>', unsafe_allow_html=True)

if image:
    st.image(image, use_column_width=True)
else:
    st.markdown('<div class="preview-box">image preview here<br>640 x 480 px</div>', unsafe_allow_html=True)


# 🔘 SECTION 3: Button
st.markdown('<div class="section">3. Generate button</div>', unsafe_allow_html=True)

generate = st.button("Generate Roast →")


# 🔥 SECTION 4: OUTPUT
st.markdown('<div class="section">4. Roast output</div>', unsafe_allow_html=True)

if generate and image:
    with st.spinner("Analyzing..."):
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

    roast = ai_roast(caption)

    st.markdown(f'<div class="roast-box">"{roast}"</div>', unsafe_allow_html=True)

    # Copy + Download
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Copy roast", roast)
    with col2:
        st.download_button("Download", roast, file_name="roast.txt")
