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
st.set_page_config(page_title="AI Food Health Analyzer", layout="wide")

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
# API KEY
# -------------------------
USDA_API_KEY = st.secrets["USDA_API_KEY"]

# -------------------------
# BMI
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
# BMR (GENDER IMPACT)
# -------------------------
def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        return 10*weight + 6.25*height - 5*age + 5
    else:
        return 10*weight + 6.25*height - 5*age - 161

# -------------------------
# ICMR RDA
# -------------------------
def get_rda(age, weight, gender):
    protein = round(0.8 * weight, 1)

    if gender == "Male":
        calories = 2500
    else:
        calories = 2000

    carbs = round((0.5 * calories) / 4, 1)
    fat = round((0.25 * calories) / 9, 1)

    return {
        "calories": calories,
        "protein": protein,
        "carbs": carbs,
        "fat": fat
    }

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
# DEFICIENCY (RDA BASED)
# -------------------------
def analyze_deficiency(total, rda):
    tips = []

    if total["protein"] < rda["protein"]:
        tips.append(f"Protein is low. Eat eggs, dal, paneer.")

    if total["carbs"] < rda["carbs"]:
        tips.append(f"Carbs are low. Eat rice, fruits.")

    if total["fat"] < rda["fat"]:
        tips.append(f"Healthy fats are low. Eat nuts.")

    if total["calories"] < rda["calories"]:
        tips.append(f"Calories are low. Increase food intake.")

    if not tips:
        tips.append("Your diet meets recommended levels.")

    return tips

# -------------------------
# ADVANCED ANALYSIS (UNCHANGED STYLE)
# -------------------------
def generate_analysis(foods, total, age, weight, height, gender, goal):

    bmi, status = calculate_bmi(weight, height)
    rda = get_rda(age, weight, gender)
    deficiency = analyze_deficiency(total, rda)

    h = height / 100
    min_w = round(18.5 * (h*h), 1)
    max_w = round(24.9 * (h*h), 1)

    bmr = calculate_bmr(weight, height, age, gender)

    # Goal logic
    if status == "Normal" and goal == "Weight Loss":
        goal_msg = "Weight loss not required."
    elif status == "Underweight":
        goal_msg = "Weight gain recommended."
    elif status == "Overweight":
        goal_msg = "Weight loss recommended."
    else:
        goal_msg = "Your goal is appropriate."

    return f"""
### 🧠 Personalized Health Analysis

👤 Age: {age}  
⚧ Gender: {gender}

⚖️ BMI: {bmi} → {status}  
Ideal Weight: {min_w}-{max_w} kg

🔥 BMR: {round(bmr)} kcal  

📊 Recommended (ICMR):
Calories: {rda['calories']}
Protein: {rda['protein']}
Fat: {rda['fat']}
Carbs: {rda['carbs']}

🍔 Foods: {foods}

📊 Your Intake:
Calories: {total['calories']}
Protein: {total['protein']}
Fat: {total['fat']}
Carbs: {total['carbs']}

🎯 Goal: {goal_msg}

⚠️ Deficiencies:
{', '.join(deficiency)}
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
# MAIN
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

    # ANALYSIS
    st.subheader("🤖 Health Analysis")
    analysis = generate_analysis(all_foods, total, age, weight, height, gender, goal)
    st.write(analysis)

    # AUDIO (ONLY DEFICIENCY BASED)
    st.subheader("🔊 Smart Audio Advice")

    rda = get_rda(age, weight, gender)
    deficiency = analyze_deficiency(total, rda)

    audio_text = "Based on your diet, "

    for d in deficiency:
        audio_text += d + " "

    audio = text_to_audio(audio_text)
    st.audio(audio)

    if os.path.exists(audio):
        os.remove(audio)
