import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
from gtts import gTTS
import tempfile
import os

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
# NUTRITION DATA
# -------------------------
def get_nutrition(food):
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
Carbs: {total['carbs']} g

Health Analysis:
- This meal is {'high' if total['fat'] > 30 else 'moderate'} in fat.
- {'High calorie intake' if total['calories'] > 700 else 'Balanced calorie level'}.

Advice:
- Based on your goal ({goal}), adjust portion size.
- Add vegetables and fiber-rich foods.
"""

# -------------------------
# AUDIO
# -------------------------
def text_to_audio(text):
    tts = gTTS(text=text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# -------------------------
# MULTIPLE IMAGE UPLOAD
# -------------------------
uploaded_files = st.file_uploader(
    "📸 Upload Food Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    all_foods = []

    for i, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)

        st.image(image, caption=f"Image {i+1}", use_column_width=True)

        st.info("🔍 Analyzing...")

        results = classifier(image)

        # ✅ ONLY TOP-1 FOOD
        food = results[0]["label"].replace("_", " ")
        all_foods.append(food)

        st.success(f"Detected: {food}")

    # -------------------------
    # TOTAL NUTRITION
    # -------------------------
    st.subheader("🍽 Combined Nutrition")

    total, details = calculate_total(all_foods)

    for food, data in details:
        st.write(f"**{food}** → {data}")

    st.subheader("📊 Total Nutrition")
    st.write(total)

    df = pd.DataFrame.from_dict(total, orient='index', columns=['Value'])
    st.bar_chart(df)

    # -------------------------
    # ALERTS
    # -------------------------
    if total["fat"] > 30:
        st.error("⚠️ High fat intake!")

    if total["calories"] > 700:
        st.warning("⚠️ High calorie intake!")

    # -------------------------
    # AI ANALYSIS
    # -------------------------
    st.subheader("🤖 Health Analysis")
    ai_response = generate_ai_response(all_foods, total)
    st.write(ai_response)

    # -------------------------
    # AUDIO OUTPUT
    # -------------------------
    st.subheader("🔊 Audio Summary")

    summary = f"You consumed {len(all_foods)} items with {total['calories']} calories."

    audio_file = text_to_audio(summary)
    st.audio(audio_file)

    if os.path.exists(audio_file):
        os.remove(audio_file)
