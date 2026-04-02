import streamlit as st
from transformers import pipeline
from PIL import Image
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

# ✅ Height in feet (min 1)
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
# RDA (PERSONALIZED)
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
    for food in foods:
        data = get_nutrition(food)
        for k in total:
            total[k] += data[k]
    return total

# -------------------------
# FOOD ANALYSIS (FIXED 🔥)
# -------------------------
def evaluate_food_health(foods, total, bmi, goal, age):

    unhealthy = ["burger", "fries", "pizza", "donut", "hot dog"]
    feedback = []

    junk_items = [f for f in foods if any(j in f.lower() for j in unhealthy)]

    # BMI reasoning
    if bmi > 25:
        feedback.append("You are overweight. High-calorie foods increase fat storage and may lead to heart disease.")
    elif bmi < 18.5:
        feedback.append("You are underweight. Low-nutrient foods will not help in healthy weight gain.")
    else:
        feedback.append("Your BMI is normal, but maintaining diet quality is important.")

    # Food reasoning
    if junk_items:
        feedback.append(f"The foods {', '.join(junk_items)} are unhealthy.")

        feedback.append("These foods contain high unhealthy fats, sugar, and low essential nutrients.")

        if age < 18:
            feedback.append("At your age, this may affect growth and immunity.")
        elif age <= 40:
            feedback.append("This may lead to weight gain and poor metabolism.")
        else:
            feedback.append("This increases risk of heart disease, cholesterol, and diabetes.")

        if goal == "Weight Loss":
            feedback.append("This meal is not suitable for weight loss due to high calories.")
        if goal == "Muscle Gain":
            feedback.append("This meal does not support muscle growth due to low protein.")

        feedback.append("Avoid fried and processed foods.")
        feedback.append("Instead, eat vegetables, fruits, whole grains, and protein-rich foods.")

    else:
        feedback.append("The detected foods are generally healthy if consumed in moderation.")

    return feedback

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
    foods = []

    for file in uploaded_files:
        img = Image.open(file)
        st.image(img)

        result = classifier(img)
        food = result[0]["label"].replace("_", " ")
        foods.append(food)

    total = calculate_total(foods)

    st.subheader("📊 Total Nutrition")
    st.write(total)

    bmi, status = calculate_bmi(weight, height)

    st.subheader("🤖 Health Analysis")
    feedback = evaluate_food_health(foods, total, bmi, goal, age)

    for f in feedback:
        st.write(f"- {f}")

    # AUDIO
    st.subheader("🔊 Smart Audio Advice")

    audio_text = "Health advice: "
    for f in feedback:
        audio_text += f + " "

    audio = text_to_audio(audio_text)
    st.audio(audio)

    if os.path.exists(audio):
        os.remove(audio)
