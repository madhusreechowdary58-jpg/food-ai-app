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
height_ft = st.sidebar.number_input("Height (feet)", 3.0, 7.0, step=0.1)
height = round(height_ft * 30.48)

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
# BMR
# -------------------------
def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        return 10*weight + 6.25*height - 5*age + 5
    else:
        return 10*weight + 6.25*height - 5*age - 161

# -------------------------
# PERSONALIZED RDA
# -------------------------
def get_rda(age, weight, height, gender):
    bmr = calculate_bmr(weight, height, age, gender)
    calories = round(bmr * 1.4)

    if age <= 10:
        protein = round(1.0 * weight, 1)
    elif age <= 18:
        protein = round(0.9 * weight, 1)
    else:
        protein = round(0.8 * weight, 1)

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
# DEFICIENCY (SMART)
# -------------------------
def analyze_deficiency(total, rda):
    tips = []

    def check(n):
        if total[n] < 0.8*rda[n]:
            return "low"
        elif total[n] > 1.2*rda[n]:
            return "high"
        return "ok"

    if check("protein") == "low":
        tips.append("Protein is low. Eat eggs, dal, paneer.")
    if check("carbs") == "low":
        tips.append("Carbs are low. Eat rice, fruits.")
    if check("fat") == "low":
        tips.append("Healthy fats are low. Add nuts.")
    if check("calories") == "high":
        tips.append("Calories are high. Reduce junk food.")
    if check("calories") == "low":
        tips.append("Calories are low. Increase intake.")

    return tips if tips else ["Diet is balanced."]

# -------------------------
# FOOD SUITABILITY (NEW)
# -------------------------
def evaluate_food_health(foods, total, bmi, goal, age):
    unhealthy = ["burger", "fries", "pizza", "donut", "hot dog"]
    feedback = []

    if bmi > 25:
        feedback.append("You are overweight. Avoid high calorie foods.")
    elif bmi < 18.5:
        feedback.append("You are underweight. Eat nutrient-rich foods.")
    else:
        feedback.append("BMI is normal. Maintain balance.")

    junk = sum([1 for f in foods if any(j in f.lower() for j in unhealthy)])

    if junk > 0:
        feedback.append(f"{junk} unhealthy food items detected.")
        feedback.append("Frequent intake may cause obesity, heart disease, and poor nutrition.")

        if age > 40:
            feedback.append("Higher risk of cholesterol and heart problems at this age.")
        elif age < 18:
            feedback.append("May affect growth and immunity.")

    if goal == "Weight Loss" and total["calories"] > 600:
        feedback.append("Not suitable for weight loss.")
    if goal == "Muscle Gain" and total["protein"] < 50:
        feedback.append("Protein insufficient for muscle gain.")

    return feedback

# -------------------------
# ANALYSIS
# -------------------------
def generate_analysis(foods, total, age, weight, height, gender, goal):

    bmi, status = calculate_bmi(weight, height)
    rda = get_rda(age, weight, height, gender)
    deficiency = analyze_deficiency(total, rda)
    food_feedback = evaluate_food_health(foods, total, bmi, goal, age)

    h = height/100
    min_w = round(18.5*(h*h),1)
    max_w = round(24.9*(h*h),1)

    bmr = calculate_bmr(weight, height, age, gender)

    return f"""
### 🧠 Personalized Health Analysis

👤 Age: {age}
⚧ Gender: {gender}

📏 Height: {round(height/30.48,1)} ft

⚖️ BMI: {bmi} → {status}
Ideal Weight: {min_w}-{max_w} kg

🔥 BMR: {round(bmr)} kcal

📊 Recommended:
Calories: {rda['calories']}
Protein: {rda['protein']}
Fat: {rda['fat']}
Carbs: {rda['carbs']}

🍔 Foods: {foods}

📊 Your Intake:
Calories: {round(total['calories'],1)}
Protein: {round(total['protein'],1)}
Fat: {round(total['fat'],1)}
Carbs: {round(total['carbs'],1)}

🧾 Food Suitability:
{chr(10).join(food_feedback)}

⚠️ Nutritional Feedback:
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

    st.subheader("📊 Total Nutrition")
    st.write(total)

    st.subheader("🤖 Health Analysis")
    analysis = generate_analysis(all_foods, total, age, weight, height, gender, goal)
    st.write(analysis)

    # AUDIO
    st.subheader("🔊 Smart Audio Advice")

    bmi, _ = calculate_bmi(weight, height)
    feedback = evaluate_food_health(all_foods, total, bmi, goal, age)

    audio_text = f"You are {age} years old. "

    for msg in feedback:
        audio_text += msg + " "

    audio = text_to_audio(audio_text)
    st.audio(audio)

    if os.path.exists(audio):
        os.remove(audio)
