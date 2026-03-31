import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
from gtts import gTTS
import tempfile
import os
import pandas as pd

# -------------------------
# PAGE CONFIG
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
# NUTRITION (Fallback)
# -------------------------
def get_nutrition(food):
    # Simple fallback values (no API needed)
    sample_data = {
        "pizza": {"calories": 285, "protein": 12, "fat": 10, "carbs": 36},
        "burger": {"calories": 295, "protein": 17, "fat": 12, "carbs": 30},
        "french fries": {"calories": 365, "protein": 4, "fat": 17, "carbs": 48},
        "hot dog": {"calories": 250, "protein": 10, "fat": 15, "carbs": 20},
        "sandwich": {"calories": 200, "protein": 8, "fat": 6, "carbs": 28}
    }
    return sample_data.get(food.lower(), {"calories": 150, "protein": 5, "fat": 5, "carbs": 20})

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
# AI RESPONSE
# -------------------------
def generate_ai_response(foods, total):
    return f"""
You consumed: {', '.join(foods)}

Total Calories: {total['calories']} kcal
Protein: {total['protein']} g
Fat: {total['fat']} g
Carbohydrates: {total['carbs']} g

Health Analysis:
- This meal is {'high' if total['fat'] > 30 else 'moderate'} in fat.
- {'High calorie intake' if total['calories'] > 700 else 'Balanced calorie level'}.

Personalized Advice:
- Based on your goal ({goal}), adjust portion size.
- Add vegetables and fiber-rich foods.
- Reduce fried or processed foods.
"""

# -------------------------
# TEXT TO AUDIO
# -------------------------
def text_to_audio(text):
    tts = gTTS(text=text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# -------------------------
# IMAGE INPUT
# -------------------------
uploaded_file = st.file_uploader("📸 Upload Food Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.info("🔍 Analyzing food...")

    # -------------------------
    # FOOD CLASSIFICATION
    # -------------------------
    results = classifier(image)
    foods = [res["label"].replace("_", " ") for res in results[:3]]

    st.success(f"Detected Foods: {foods}")

    # -------------------------
    # NUTRITION
    # -------------------------
    st.subheader("🍽 Nutrition Breakdown")

    total, details = calculate_total(foods)

    for food, data in details:
        st.write(f"**{food}** → {data}")

    # -------------------------
    # TOTAL
    # -------------------------
    st.subheader("📊 Total Nutrition")
    st.write(total)

    # Chart
    df = pd.DataFrame.from_dict(total, orient='index', columns=['Value'])
    st.bar_chart(df)

    # -------------------------
    # ALERTS
    # -------------------------
    if total["fat"] > 30:
        st.error("⚠️ High fat intake detected!")

    if total["calories"] > 700:
        st.warning("⚠️ High calorie meal!")

    # -------------------------
    # AI ANALYSIS
    # -------------------------
    st.subheader("🤖 Health Analysis")
    ai_response = generate_ai_response(foods, total)
    st.write(ai_response)

    # -------------------------
    # AUDIO
    # -------------------------
    st.subheader("🔊 Audio Summary")

    summary = f"Your meal has {total['calories']} calories. It is {'high' if total['fat'] > 30 else 'moderate'} in fat."

    audio_file = text_to_audio(summary)
    st.audio(audio_file)

    if os.path.exists(audio_file):
        os.remove(audio_file)
