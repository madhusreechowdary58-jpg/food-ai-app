import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
from gtts import gTTS
import tempfile
import os
import requests

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="GenAI Health Analyzer", layout="wide")

st.title("🍔 GenAI Food & Health Analyzer")
st.markdown("---")

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
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="nateraw/food")

classifier = load_model()

# -------------------------
# API KEYS
# -------------------------
USDA_API_KEY = st.secrets["USDA_API_KEY"]

# -------------------------
# BMI CALCULATION
# -------------------------
def calculate_bmi(weight, height):
    h = height / 100
    bmi = weight / (h * h)

    if bmi < 18.5:
        status = "Underweight"
    elif bmi < 25:
        status = "Normal"
    elif bmi < 30:
        status = "Overweight"
    else:
        status = "Obese"

    return round(bmi, 2), status

# -------------------------
# CLEAN FOOD
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
        for k in total:
            total[k] += data[k]

    return total, details

# -------------------------
# DEFICIENCY
# -------------------------
def analyze_deficiency(total):
    tips = []

    if total["protein"] < 50:
        tips.append("Low protein → eat eggs, chicken, dal.")
    if total["carbs"] < 130:
        tips.append("Low carbs → eat rice, fruits.")
    if total["fat"] < 20:
        tips.append("Low healthy fats → eat nuts.")
    if total["calories"] < 500:
        tips.append("Low calories → increase food intake.")

    if not tips:
        tips.append("Diet is balanced.")

    return tips

# -------------------------
# ADVANCED AI LOGIC (NO FAIL)
# -------------------------
def generate_analysis(foods, total, age, weight, height, gender, goal):

    bmi, status = calculate_bmi(weight, height)
    deficiency = analyze_deficiency(total)

    # Goal validation
    goal_msg = ""
    if status == "Normal" and goal == "Weight Loss":
        goal_msg = "You already have normal weight. Weight loss is not required."
    elif status == "Underweight" and goal == "Weight Loss":
        goal_msg = "Weight loss is dangerous for your condition."
    elif status == "Overweight" and goal == "Muscle Gain":
        goal_msg = "Focus on fat loss before muscle gain."
    else:
        goal_msg = "Your goal is appropriate."

    # Age logic
    if age < 18:
        age_msg = "You are young. Balanced nutrition is very important for growth."
    elif age > 50:
        age_msg = "Focus on low fat and heart-healthy diet."
    else:
        age_msg = "Maintain balanced adult diet."

    return f"""
### 🧠 Personalized Health Analysis

👤 Age: {age} ({age_msg})  
⚖️ BMI: {bmi} → {status}  

🎯 Goal Check: {goal_msg}

🍔 Foods Consumed: {foods}

📊 Nutrition:
Calories: {total['calories']}
Protein: {total['protein']}
Fat: {total['fat']}
Carbs: {total['carbs']}

⚠️ Deficiencies:
{', '.join(deficiency)}

🥗 Recommendations:
- Adjust diet based on BMI status
- Add missing nutrients
- Follow goal-oriented diet
"""

# -------------------------
# AUDIO
# -------------------------
def text_to_audio(text):
    tts = gTTS(text)
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(file.name)
    return file.name

# -------------------------
# UPLOAD
# -------------------------
uploaded_files = st.file_uploader("📸 Upload Food Images", accept_multiple_files=True)

if uploaded_files:
    all_foods = []

    for file in uploaded_files:
        img = Image.open(file)
        st.image(img)

        res = classifier(img)
        food = res[0]["label"].replace("_", " ")
        all_foods.append(food)

    total, details = calculate_total(all_foods)

    st.subheader("🍽 Per Food Nutrition")
    for f, d in details:
        st.write(f"{f} → {d}")

    st.subheader("📊 Total Nutrition")
    st.write(total)

    # -------------------------
    # FINAL ANALYSIS
    # -------------------------
    st.subheader("🤖 Health Analysis")

    analysis = generate_analysis(all_foods, total, age, weight, height, gender, goal)
    st.write(analysis)

    # -------------------------
    # AUDIO
    # -------------------------
    st.subheader("🔊 Audio Advice")

    audio = text_to_audio(analysis[:300])
    st.audio(audio)

    if os.path.exists(audio):
        os.remove(audio)
