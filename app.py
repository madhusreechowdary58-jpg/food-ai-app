import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
from gtts import gTTS
import tempfile
import os
import requests
import time

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="AI Food Analyzer", layout="centered")

st.title("🍔 AI Food Nutrition Analyzer & Health Advisor")
st.markdown("---")

# -------------------------
# USER INPUT
# -------------------------
st.sidebar.header("👤 User Details")

age = st.sidebar.number_input("Age", 10, 80)
weight = st.sidebar.number_input("Weight (kg)", 30, 120)
goal = st.sidebar.selectbox("Goal", ["Weight Loss", "Muscle Gain", "Maintain"])

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="nateraw/food")

classifier = load_model()

# -------------------------
# API KEYS
# -------------------------
HF_API_KEY = st.secrets["HF_API_KEY"]
USDA_API_KEY = st.secrets["USDA_API_KEY"]

# -------------------------
# CLEAN FOOD NAME
# -------------------------
def clean_food(food):
    mapping = {
        "macaroni and cheese": "macaroni",
        "bread pudding": "bread",
        "cheeseburger": "burger",
        "french fries": "fries"
    }
    return mapping.get(food.lower(), food)

# -------------------------
# USDA NUTRITION
# -------------------------
def get_nutrition(food):
    try:
        food = clean_food(food)

        url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food}&api_key={USDA_API_KEY}"
        res = requests.get(url).json()

        nutrients = {"calories": 0, "protein": 0, "fat": 0, "carbs": 0}

        foods = res.get("foods", [])
        if not foods:
            return nutrients

        for item in foods[0]["foodNutrients"]:
            name = item["nutrientName"].lower()

            if "energy" in name:
                nutrients["calories"] = item["value"]
            elif "protein" in name:
                nutrients["protein"] = item["value"]
            elif "fat" in name:
                nutrients["fat"] = item["value"]
            elif "carbohydrate" in name:
                nutrients["carbs"] = item["value"]

        return nutrients

    except:
        return {"calories": 0, "protein": 0, "fat": 0, "carbs": 0}

# -------------------------
# TOTAL
# -------------------------
def calculate_total(foods):
    total = {"calories": 0, "protein": 0, "fat": 0, "carbs": 0}
    details = []

    for food in foods:
        data = get_nutrition(food)
        details.append((food, data))

        for key in total:
            total[key] += data[key]

    return total, details

# -------------------------
# GEN AI (FOR MAIN ANALYSIS)
# -------------------------
def generate_ai_response(foods, total):

    prompt = f"""
    Act as a professional nutritionist.

    Foods eaten: {foods}
    Nutrition: {total}
    Goal: {goal}

    Provide:
    - Health analysis
    - Missing nutrients
    - Food recommendations
    """

    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=15)
        output = response.json()

        if isinstance(output, list):
            return output[0]["generated_text"]

    except:
        pass

    return "Basic advice: Maintain balanced diet with protein, fruits and vegetables."

# -------------------------
# CHATBOT FUNCTION (FIXED)
# -------------------------
def generate_chatbot_response(user_input, foods, total):

    prompt = f"""
    You are a helpful nutrition assistant.

    User ate: {foods}
    Nutrition: {total}

    User question: {user_input}

    Give a clear and direct answer.
    """

    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=15)
        output = response.json()

        if isinstance(output, list):
            return output[0]["generated_text"]

    except:
        pass

    return "Try asking: Is my diet healthy? What should I eat next?"

# -------------------------
# AUDIO
# -------------------------
def text_to_audio(text):
    tts = gTTS(text=text)
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(file.name)
    return file.name

# -------------------------
# UPLOAD
# -------------------------
uploaded_files = st.file_uploader("📸 Upload Food Images", accept_multiple_files=True)

if uploaded_files:
    all_foods = []

    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Image {i+1}")

        results = classifier(image)
        food = results[0]["label"].replace("_", " ")
        all_foods.append(food)

        st.success(f"Detected: {food}")

    total, details = calculate_total(all_foods)

    st.subheader("🍽 Nutrition per Food")
    for food, data in details:
        st.write(f"### {food}")
        st.write(data)
        st.markdown("---")

    st.subheader("📊 Total Nutrition")
    st.write(total)

    df = pd.DataFrame.from_dict(total, orient='index', columns=['Value'])
    st.bar_chart(df)

    # AI ANALYSIS
    st.subheader("🤖 Health Analysis")
    ai_response = generate_ai_response(all_foods, total)
    st.write(ai_response)

    # AUDIO
    st.subheader("🔊 Audio Advice")
    summary = f"You consumed {len(all_foods)} foods. Calories {total['calories']}."
    audio_file = text_to_audio(summary)
    st.audio(audio_file)

# -------------------------
# CHATBOT UI (FIXED)
# -------------------------
st.subheader("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about your diet...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    response = generate_chatbot_response(
        user_input,
        all_foods if 'all_foods' in locals() else [],
        total if 'total' in locals() else {"calories":0,"protein":0,"fat":0,"carbs":0}
    )

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
