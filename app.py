import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import os




# 🤖 AI Roast Function (ROBUST)
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ai_roast(caption):
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "You are a funny and savage roaster. Give short, witty roasts."
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
        return "Roast machine broke 😭 try again"
# 📦 Load Caption Model
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


processor, model = load_model()


# 🎨 UI CONFIG
st.set_page_config(page_title="AI Roast System", layout="centered")

# 🎨 Custom CSS
st.markdown("""
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
    color: #1a1a1a;
}
</style>
""", unsafe_allow_html=True)

# 🔥 Title
st.markdown('<div class="title">AI Vision Roast System 😂🔥</div>', unsafe_allow_html=True)

# 📦 Card
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

st.markdown('</div>', unsafe_allow_html=True)
