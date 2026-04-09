import streamlit as st
from PIL import Image
import base64
from openai import OpenAI
from gtts import gTTS
import tempfile, os

# -------------------------
# SETUP
# -------------------------
st.set_page_config(page_title="Full GenAI Food Analyzer", layout="wide")
st.title("🍔 Full GenAI Food & Health Analyzer")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------
# USER INPUT
# -------------------------
st.sidebar.header("👤 User Profile")

age = st.sidebar.number_input("Age", 5, 80)
weight = st.sidebar.number_input("Weight (kg)", 20, 150)
height = st.sidebar.number_input("Height (cm)", 100, 200)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
goal = st.sidebar.selectbox("Goal", ["Weight Loss", "Muscle Gain", "Maintain"])

# -------------------------
# IMAGE TO BASE64
# -------------------------
def encode_image(image):
    return base64.b64encode(image.read()).decode("utf-8")

# -------------------------
# GENAI ANALYSIS
# -------------------------
def analyze_with_genai(image, age, weight, height, gender, goal):

    img_base64 = encode_image(image)

    prompt = f"""
    Analyze the food image and provide:

    1. Detected food items
    2. Estimated calories, protein, fat, carbs
    3. Calculate BMI, BMR, and daily nutrient needs
    4. Compare intake vs required
    5. Give health advice
    6. Suggest improvements based on:
       Age: {age}, Weight: {weight}, Height: {height}, Gender: {gender}, Goal: {goal}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}
            ]}
        ]
    )

    return response.choices[0].message.content

# -------------------------
# AUDIO
# -------------------------
def text_to_audio(text):
    tts = gTTS(text)
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(file.name)
    return file.name

# -------------------------
# MAIN
# -------------------------
uploaded_file = st.file_uploader("📸 Upload Food Image")

if uploaded_file:
    st.image(uploaded_file)

    st.subheader("🤖 GenAI Analysis")

    result = analyze_with_genai(
        uploaded_file,
        age, weight, height, gender, goal
    )

    st.write(result)

    # AUDIO OUTPUT
    st.subheader("🔊 Audio Advice")
    audio = text_to_audio(result[:500])  # limit text
    st.audio(audio)

    if os.path.exists(audio):
        os.remove(audio)
