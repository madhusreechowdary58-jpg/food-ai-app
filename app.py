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
# USDA NUTRITION (ACCURATE)
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
# TOTAL CALCULATION
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
# GEN AI RESPONSE (STABLE)
# -------------------------
def generate_ai_response(foods, total):

    prompt = f"""
    Act as a nutrition expert.

    Foods eaten: {foods}
    Nutrition: {total}
    Goal: {goal}

    Give:
    - Health analysis
    - Missing nutrients
    - What foods to eat
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

    # fallback (IMPORTANT)
    return f"""
You consumed {foods}.
Calories: {total['calories']}

Your diet may lack protein and fiber.
Eat eggs, dal, fruits, vegetables.
"""

# -------------------------
# DEFICIENCY ANALYSIS (AUDIO)
# -------------------------
def analyze_nutrition(total):
    advice = []

    if total["protein"] < 50:
        advice.append("Protein is low. Eat eggs, chicken, paneer, or dal.")

    if total["carbs"] < 130:
        advice.append("Carbohydrates are low. Add rice, fruits, or oats.")

    if total["fat"] < 20:
        advice.append("Healthy fats are low. Add nuts and seeds.")

    if total["calories"] < 500:
        advice.append("Calories are low. Increase balanced meals.")

    if not advice:
        advice.append("Your diet is balanced. Keep it up.")

    return advice

# -------------------------
# AUDIO
# -------------------------
def text_to_audio(text):
    tts = gTTS(text=text)
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(file.name)
    return file.name

# -------------------------
# IMAGE UPLOAD
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

    # -------------------------
    # NUTRITION
    # -------------------------
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

    # -------------------------
    # AI ANALYSIS
    # -------------------------
    st.subheader("🤖 Health Analysis")
    ai_response = generate_ai_response(all_foods, total)
    st.write(ai_response)

    # -------------------------
    # AUDIO ADVICE
    # -------------------------
    st.subheader("🔊 Smart Audio Advice")

    tips = analyze_nutrition(total)

    summary = f"You consumed {len(all_foods)} foods. "

    for tip in tips:
        summary += tip + " "

    audio_file = text_to_audio(summary)
    st.audio(audio_file)

    if os.path.exists(audio_file):
        os.remove(audio_file)

# -------------------------
# CHATBOT (FIXED)
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

    response = generate_ai_response(
        all_foods if 'all_foods' in locals() else [],
        total if 'total' in locals() else {"calories":0,"protein":0,"fat":0,"carbs":0}
    )

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
