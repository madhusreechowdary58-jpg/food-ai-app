import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile
import os
import requests
import matplotlib.pyplot as plt
from openai import OpenAI

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI Food Health Analyzer", layout="wide")
st.title("🍔 GenAI Food & Health Analyzer")
st.markdown("---")

# -------------------------
# OPENAI CLIENT
# -------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

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
# AUDIO
# -------------------------
def text_to_audio(text):
    tts = gTTS(text)
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(file.name)
    return file.name

# -------------------------
# AI DIET AGENT (WESTERN)
# -------------------------
def diet_agent(age, weight, height, gender, goal, foods, bmi):

    system_prompt = "You are a Western nutrition expert."

    user_prompt = f"""
    Age: {age}, Weight: {weight}, Height: {height}
    Gender: {gender}, Goal: {goal}, BMI: {bmi}
    Foods: {foods}

    Generate:
    - Diet analysis
    - Full-day Western diet plan
    - Improvements
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content

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

    # DETECTED FOODS
    st.subheader("🍔 Detected Food Items")
    for f in all_foods:
        st.success(f)

    # PER FOOD
    st.subheader("📊 Per Food Nutrients")
    for f, d in details:
        st.write(f"{f} → Carbs:{round(d['carbs'],1)}, Protein:{round(d['protein'],1)}, Fat:{round(d['fat'],1)}, Vitamins:{round(d['vitamins'],1)}")

    # BMI
    bmi, status = calculate_bmi(weight, height)

    st.subheader("🤖 Health Analysis")
    st.write(f"BMI: {bmi} → {status}")

    # AUDIO
    st.subheader("🔊 Smart Audio Advice")
    audio = text_to_audio(f"Your BMI is {bmi}")
    st.audio(audio)

    if os.path.exists(audio):
        os.remove(audio)

    # -------------------------
    # ✅ AI DIET PLANNER
    # -------------------------
    st.subheader("🤖 AI Diet Planner")

    if st.button("🥗 Generate Diet Plan"):

        with st.spinner("Generating diet plan..."):
            try:
                plan = diet_agent(age, weight, height, gender, goal, all_foods, bmi)
                st.write(plan)

            except Exception as e:
                st.warning("⚠️ API limit reached. Showing basic plan.")

                st.write("""
### 🥗 Basic Western Diet Plan

Breakfast:
- Oatmeal + milk + fruits
- Boiled eggs

Lunch:
- Grilled chicken + brown rice + vegetables

Snacks:
- Apple + almonds

Dinner:
- Whole wheat bread + salad + soup

Tip:
Avoid fried and processed foods. Drink more water.
""")
