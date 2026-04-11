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
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="nateraw/food")

@st.cache_resource
def load_text_model():
    return pipeline("text-generation", model="gpt2")

classifier = load_model()
generator = load_text_model()

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
# TOTAL
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
# DEFICIENCY
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
# FOOD SUITABILITY
# -------------------------
def evaluate_food_health(foods, total, bmi, goal, age):
    unhealthy = ["burger", "fries", "pizza", "donut", "hot dog"]
    feedback = []

    junk_items = [f for f in foods if any(j in f.lower() for j in unhealthy)]

    if bmi > 25:
        feedback.append("You are overweight.")
    elif bmi < 18.5:
        feedback.append("You are underweight.")
    else:
        feedback.append("Your BMI is normal.")

    if junk_items:
        feedback.append(f"Unhealthy foods: {', '.join(junk_items)}")
        feedback.append("Avoid fried and processed foods.")
    else:
        feedback.append("Foods are healthy in moderation.")

    return feedback

# -------------------------
# ANALYSIS
# -------------------------
def generate_analysis(foods, total, age, weight, height, gender, goal):
    bmi, status = calculate_bmi(weight, height)
    rda = get_rda(age, weight, height, gender)
    deficiency = analyze_deficiency(total, rda)
    feedback = evaluate_food_health(foods, total, bmi, goal, age)

    return f"""
BMI: {bmi} ({status})
Foods: {foods}
Calories: {round(total['calories'],1)}

Feedback:
{chr(10).join(feedback)}

Deficiency:
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

    st.subheader("🍔 Detected Food Items")
    for f in all_foods:
        st.success(f)

    st.subheader("📊 Total Nutrients")
    st.write(total)

    st.subheader("🤖 Health Analysis")
    analysis = generate_analysis(all_foods, total, age, weight, height, gender, goal)
    st.write(analysis)

    audio = text_to_audio(analysis)
    st.audio(audio)

# =========================================================
# ✅ NEW FEATURE: AI ASSISTANT (ADDED - DOES NOT DISTURB)
# =========================================================

st.markdown("---")
st.subheader("🤖 AI Health Assistant")

user_query = st.text_input("Ask anything (diet / tips / summary / health advice):")

def ai_agent(query, context=""):
    query = query.lower()

    if "diet" in query:
        prompt = f"Give short diet advice for {context}"
    elif "tips" in query:
        prompt = f"Give 3 health tips for {context}"
    elif "summary" in query:
        prompt = f"Summarize this:\n{context}"
    else:
        prompt = query

    result = generator(prompt, max_new_tokens=80)
    return result[0]['generated_text']

if st.button("Run AI Assistant"):
    context = analysis if 'analysis' in locals() else "general health"
    response = ai_agent(user_query, context)
    st.write(response)
