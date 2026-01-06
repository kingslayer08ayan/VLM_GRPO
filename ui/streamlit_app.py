import streamlit as st
import requests
from PIL import Image
from io import BytesIO

API_URL = "http://api:8000/caption"

st.set_page_config(page_title="VLM-GRPO Demo", layout="centered")

st.title("🖼️ Vision-Language Model (GRPO)")
st.write("Upload an image to compare **Base vs GRPO captions + rewards**")

uploaded = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)
st.sidebar.header("Generation Controls")

max_new_tokens = st.sidebar.slider(
    "Max new tokens", min_value=10, max_value=60, value=30, step=5
)

num_beams = st.sidebar.slider(
    "Beam size", min_value=1, max_value=5, value=3
)

repetition_penalty = st.sidebar.slider(
    "Repetition penalty", min_value=1.0, max_value=2.0, value=1.2, step=0.1
)

length_penalty = st.sidebar.slider(
    "Length penalty", min_value=0.6, max_value=1.4, value=1.0, step=0.1
)
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", width=400)

    with st.spinner("Running inference on GPU..."):
        res = requests.post(
            API_URL,
            files={
                "file": (uploaded.name, uploaded.getvalue(), uploaded.type)
            },
            data={
                "max_new_tokens": max_new_tokens,
                "num_beams": num_beams,
                "repetition_penalty": repetition_penalty,
                "length_penalty": length_penalty,
            }
        )

    if res.status_code == 200:
        data = res.json()

        st.subheader("📄 Captions")
        st.markdown(f"**Base caption:** {data['base_caption']}")
        st.markdown(f"**GRPO caption:** {data['grpo_caption']}")

        st.subheader("📊 Rewards")
        st.metric("Base reward", data["base_reward"])
        st.metric("GRPO reward", data["grpo_reward"])
        st.metric("Δ Improvement", data["delta"])
    else:
        st.error("API error")

