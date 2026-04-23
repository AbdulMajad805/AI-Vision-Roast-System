import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import google.generativeai as genai
import os

# ============================================================
# 🤖 AI Roast Function — Google Gemini (Free via AI Studio)
# ============================================================
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def ai_roast(caption):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            f"You are a funny Pakistani desi roaster who roasts in simple, native English that everyone understands. "
            f"Think like a witty guy from Lahore or Karachi who cracks jokes. "
            f"Keep the roast SHORT — 2 to 3 sentences max. "
            f"Make it funny, relatable, and easy to understand — like talking to a friend. "
            f"No difficult words. No flowery English. Just savage desi humor in plain English. "
            f"Roast this image description: {caption}"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Yaar kuch gadbad ho gayi: {str(e)}"


# ============================================================
# 📦 Load BLIP Caption Model
# ============================================================
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model


processor, model = load_model()


# ============================================================
# 🎨 Page Config
# ============================================================
st.set_page_config(page_title="AI Roast System", layout="centered")


# ============================================================
# 🎨 Antique / Vintage CSS Theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=Crimson+Text:ital@0;1&display=swap');

/* ── Root Variables ── */
:root {
    --parchment:    #f5ecd7;
    --dark-ink:     #2b1d0e;
    --sepia-mid:    #8b5e3c;
    --gold:         #c9973b;
    --gold-light:   #e8c97a;
    --burnt:        #6b3a1f;
    --paper:        #fdf6e3;
    --shadow:       rgba(43, 29, 14, 0.25);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Crimson Text', Georgia, serif !important;
    background-color: var(--parchment) !important;
    color: var(--dark-ink) !important;
}

/* Noise / grain texture overlay */
body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='300'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='300' height='300' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 9999;
}

/* ── Hide Streamlit Branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Main Container ── */
.block-container {
    max-width: 720px !important;
    padding-top: 2rem !important;
}

/* ── Banner / Header ── */
.antique-banner {
    text-align: center;
    padding: 38px 30px 28px;
    background: linear-gradient(160deg, #3b1f0a 0%, #6b3a1f 50%, #3b1f0a 100%);
    border-radius: 4px;
    border: 3px double var(--gold);
    box-shadow:
        0 0 0 6px var(--parchment),
        0 0 0 8px var(--sepia-mid),
        0 6px 30px var(--shadow);
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.antique-banner::before {
    content: '❧';
    position: absolute;
    top: 8px; left: 16px;
    font-size: 22px;
    color: var(--gold-light);
    opacity: 0.6;
}
.antique-banner::after {
    content: '❧';
    position: absolute;
    top: 8px; right: 16px;
    font-size: 22px;
    color: var(--gold-light);
    opacity: 0.6;
    transform: scaleX(-1);
}

.antique-banner h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    color: var(--gold-light) !important;
    letter-spacing: 2px;
    margin: 0 0 6px 0 !important;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.5);
}

.antique-banner p {
    font-family: 'Crimson Text', serif !important;
    font-style: italic;
    font-size: 1.05rem !important;
    color: #d4b896 !important;
    margin: 0 !important;
    letter-spacing: 1px;
}

.ornament {
    color: var(--gold);
    font-size: 1.3rem;
    display: block;
    margin: 6px 0 2px;
    letter-spacing: 8px;
}

/* ── Upload Card ── */
.upload-card {
    background: var(--paper);
    border: 2px solid var(--sepia-mid);
    border-radius: 4px;
    padding: 28px 30px;
    box-shadow: inset 0 0 40px rgba(139,94,60,0.08), 0 4px 20px var(--shadow);
    position: relative;
    margin-bottom: 20px;
}

.upload-card::before {
    content: '';
    position: absolute;
    inset: 6px;
    border: 1px solid var(--gold-light);
    border-radius: 2px;
    opacity: 0.35;
    pointer-events: none;
}

/* ── File Uploader Styling ── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--sepia-mid) !important;
    border-radius: 4px !important;
    background: rgba(201,151,59,0.05) !important;
    padding: 16px !important;
}

[data-testid="stFileUploader"] label {
    font-family: 'Playfair Display', serif !important;
    color: var(--burnt) !important;
    font-size: 1rem !important;
}

/* ── Button ── */
.stButton > button {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    color: var(--parchment) !important;
    background: linear-gradient(135deg, #6b3a1f 0%, #c9973b 50%, #6b3a1f 100%) !important;
    border: 2px solid var(--gold) !important;
    border-radius: 3px !important;
    padding: 12px 36px !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(107,58,31,0.4) !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #8b5e3c 0%, #e8c97a 50%, #8b5e3c 100%) !important;
    box-shadow: 0 6px 24px rgba(107,58,31,0.6) !important;
    transform: translateY(-1px) !important;
}

/* ── Caption Box ── */
.caption-box {
    background: rgba(201,151,59,0.10);
    border-left: 4px solid var(--gold);
    border-radius: 0 4px 4px 0;
    padding: 14px 18px;
    margin: 18px 0 10px;
    font-style: italic;
    font-size: 1.05rem;
    color: var(--burnt);
}

.caption-label {
    font-family: 'Playfair Display', serif;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: var(--sepia-mid);
    margin-bottom: 4px;
}

/* ── Roast Output Box ── */
.roast-section-title {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.15rem !important;
    color: var(--burnt) !important;
    text-align: center;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 22px 0 10px;
    display: flex;
    align-items: center;
    gap: 10px;
    justify-content: center;
}

.roast-box {
    background: linear-gradient(135deg, #3b1f0a 0%, #5a2e10 100%);
    border: 2px solid var(--gold);
    border-radius: 4px;
    padding: 24px 26px;
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 1.18rem;
    line-height: 1.7;
    color: var(--gold-light);
    text-align: center;
    box-shadow:
        inset 0 0 30px rgba(0,0,0,0.3),
        0 6px 24px rgba(43,29,14,0.4);
    position: relative;
}

.roast-box::before {
    content: '\201C';
    font-size: 5rem;
    color: var(--gold);
    opacity: 0.2;
    position: absolute;
    top: -10px; left: 10px;
    line-height: 1;
    font-family: Georgia, serif;
}

.roast-box::after {
    content: '\201D';
    font-size: 5rem;
    color: var(--gold);
    opacity: 0.2;
    position: absolute;
    bottom: -30px; right: 10px;
    line-height: 1;
    font-family: Georgia, serif;
}

/* ── Divider ── */
.antique-divider {
    text-align: center;
    color: var(--gold);
    font-size: 1.1rem;
    letter-spacing: 10px;
    margin: 20px 0;
    opacity: 0.7;
}

/* ── Spinner ── */
[data-testid="stSpinner"] p {
    font-family: 'Crimson Text', serif !important;
    font-style: italic !important;
    color: var(--sepia-mid) !important;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border: 3px solid var(--sepia-mid) !important;
    border-radius: 3px !important;
    box-shadow: 0 4px 20px var(--shadow) !important;
}

/* ── Footer ── */
.antique-footer {
    text-align: center;
    font-family: 'Crimson Text', serif;
    font-style: italic;
    color: var(--sepia-mid);
    font-size: 0.85rem;
    margin-top: 30px;
    opacity: 0.7;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 🔥 Banner
# ============================================================
st.markdown("""
<div class="antique-banner">
    <h1>🔥 AI Vision Roaster 🔥</h1>
    <span class="ornament">⸻ ✦ ⸻</span>
    <p>Upload thine image &amp; receive a desi roasting of legendary proportions</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 📦 Upload Card
# ============================================================
st.markdown('<div class="upload-card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📜  Choose your image (JPG / PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    st.markdown('<div class="antique-divider">· · · ✦ · · ·</div>', unsafe_allow_html=True)

    if st.button("🔥 Generate Roast — If You Dare"):
        with st.spinner("📖  Analysing the visual evidence..."):
            inputs = processor(image, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

        st.markdown(f"""
        <div class="caption-box">
            <div class="caption-label">What the AI Sees</div>
            {caption}
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("🌶️  The Roast Master is preparing..."):
            roast = ai_roast(caption)

        st.markdown("""
        <div class="roast-section-title">
            🔥 &nbsp; The Roast &nbsp; 🔥
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="roast-box">{roast}</div>', unsafe_allow_html=True)

        st.markdown('<div class="antique-divider" style="margin-top:40px;">· · · ✦ · · ·</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# Footer
# ============================================================
st.markdown("""
<div class="antique-footer">
    Powered by BLIP Vision · Google Gemini · Desi Humor Since Forever
</div>
""", unsafe_allow_html=True)
