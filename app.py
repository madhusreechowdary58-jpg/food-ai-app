import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile
import os
import requests
import matplotlib.pyplot as plt

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

height_ft = st.sidebar.number_input("Height (feet)", 1.0, 7.0, step=0.1)
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
        return round(bmi, 2), "Underweight"
    elif bmi < 25:
        return round(bmi, 2), "Normal"
    elif bmi < 30:
        return round(bmi, 2), "Overweight"
    else:
        return round(bmi, 2), "Obese"

# -------------------------
# BMR
# -------------------------
def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        return 10*weight + 6.25*height - 5*age + 5
    else:
        return 10*weight + 6.25*height - 5*age - 161

# -------------------------
# RDA
# -------------------------
def get_rda(age, weight, height, gender):
    bmr = calculate_bmr(weight, height, age, gender)
    calories = round(bmr * 1.4)

    protein = round(0.8 * weight, 1)
    carbs = round((0.5 * calories) / 4, 1)
    fat = round((0.25 * calories) / 9, 1)

    return {"calories": calories, "protein": protein, "carbs": carbs, "fat": fat}

# -------------------------
# CLEAN FOOD
# -------------------------
def clean_food(food):
    mapping = {
        "cheeseburger": "burger",
        "french fries": "fries",
        "bread pudding": "bread"
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

        nutrients = {"calories": 0, "protein": 0, "fat": 0, "carbs": 0, "vitamins": 0}

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
            elif "vitamin" in name:
                nutrients["vitamins"] += item["value"]

        return nutrients
    except:
        return {"calories": 0, "protein": 0, "fat": 0, "carbs": 0, "vitamins": 0}

# -------------------------
# TOTAL NUTRITION
# -------------------------
def calculate_total(foods):
    total = {"calories": 0, "protein": 0, "fat": 0, "carbs": 0, "vitamins": 0}
    details = []

    for food in foods:
        data = get_nutrition(food)
        details.append((food, data))

        for k in total:
            total[k] += data[k]

    return total, details

# -------------------------
# DEFICIENCY ANALYSIS
# -------------------------
def analyze_deficiency(total, rda):
    tips = []

    if total["protein"] < 0.8 * rda["protein"]:
        tips.append("Protein is low. Eat eggs, dal, paneer.")
    if total["carbs"] < 0.8 * rda["carbs"]:
        tips.append("Carbs are low. Eat rice, fruits.")
    if total["fat"] < 0.8 * rda["fat"]:
        tips.append("Healthy fats are low. Add nuts.")
    if total["calories"] > 1.2 * rda["calories"]:
        tips.append("Calories are high. Reduce junk food.")
    if total["calories"] < 0.8 * rda["calories"]:
        tips.append("Calories are low. Increase intake.")

    return tips if tips else ["Diet is balanced."]

# -------------------------
# FOOD HEALTH EVALUATION
# -------------------------
def evaluate_food_health(foods, total, bmi, goal, age):

    unhealthy = ["burger", "fries", "pizza", "donut", "hot dog"]
    feedback = []

    junk_items = [f for f in foods if any(j in f.lower() for j in unhealthy)]

    if bmi > 25:
        feedback.append("You are overweight. Avoid high-calorie foods.")
    elif bmi < 18.5:
        feedback.append("You are underweight. Eat nutritious food.")
    else:
        feedback.append("BMI is normal. Maintain healthy diet.")

    if junk_items:
        feedback.append(f"Unhealthy foods detected: {', '.join(junk_items)}")
        feedback.append("Avoid fried and processed foods.")

    else:
        feedback.append("Foods are healthy.")

    return feedback

# -------------------------
# GOAL EVALUATION
# -------------------------
def evaluate_goal(bmi, goal):

    if bmi < 18.5:
        if goal == "Weight Loss":
            return "❌ Underweight. Weight loss not recommended."
        else:
            return "✅ Goal is appropriate."

    elif bmi > 25:
        if goal == "Muscle Gain":
            return "⚠️ Focus on fat loss first."
        elif goal == "Maintain":
            return "⚠️ Consider weight loss."
        else:
            return "✅ Good goal."

    else:
        return "✅ Goal is suitable."

# -------------------------
# AI AGENT
# -------------------------
def ai_agent(foods, age, weight, height, gender, goal):

    bmi, status = calculate_bmi(weight, height)
    rda = get_rda(age, weight, height, gender)
    total, details = calculate_total(foods)
    deficiency = analyze_deficiency(total, rda)
    feedback = evaluate_food_health(foods, total, bmi, goal, age)
    goal_check = evaluate_goal(bmi, goal)

    return {
        "BMI": bmi,
        "Status": status,
        "Total": total,
        "Deficiency": deficiency,
        "Feedback": feedback,
        "Goal": goal_check
    }

# -------------------------
# MAIN APP
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

    result = ai_agent(all_foods, age, weight, height, gender, goal)

    st.subheader("🍔 Detected Foods")
    st.write(all_foods)

    st.subheader("📊 AI Agent Output")
    st.json(result)
