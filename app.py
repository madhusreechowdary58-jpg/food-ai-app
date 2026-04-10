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
def load_food_model():
    return pipeline("image-classification", model="nateraw/food")

@st.cache_resource
def load_llm():
    return pipeline("text-generation", model="gpt2")

classifier = load_food_model()
llm = load_llm()

# -------------------------
# USDA API
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
# USDA NUTRITION
# -------------------------
def clean_food(food):
    mapping = {
        "cheeseburger": "burger",
        "french fries": "fries",
        "bread pudding": "bread"
    }
    return mapping.get(food.lower(), food)

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
# AUDIO
# -------------------------
def text_to_audio(text):
    tts = gTTS(text)
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(file.name)
    return file.name

# -------------------------
# AI DIET AGENT (FINAL FIX)
# -------------------------
def diet_agent(age, weight, height, gender, goal, foods, bmi):

    prompt = f"""
    Create a simple Western diet plan.

    Age: {age}, Weight: {weight}, Height: {height}
    Goal: {goal}, BMI: {bmi}

    Give:
    Breakfast:
    Lunch:
    Dinner:
    Snacks:
    Tips:
    """

    result = llm(
        prompt,
        max_new_tokens=120,
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.2
    )

    text = result[0]['generated_text']

    # ❌ Remove prompt echo
    text = text.replace(prompt, "")

    # ✅ Clean lines
    lines = text.split("\n")
    clean = []
    for l in lines:
        if l.strip() and l not in clean:
            clean.append(l)

    return "\n".join(clean[:6])

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

    st.subheader("📊 Nutrients")
    for f, d in details:
        st.write(f"{f} → Carbs:{round(d['carbs'],1)}, Protein:{round(d['protein'],1)}, Fat:{round(d['fat'],1)}")

    bmi, status = calculate_bmi(weight, height)

    st.subheader("🤖 Health Analysis")
    st.write(f"BMI: {bmi} → {status}")

    st.subheader("🔊 Audio")
    audio = text_to_audio(f"Your BMI is {bmi}")
    st.audio(audio)

    if os.path.exists(audio):
        os.remove(audio)

    # -------------------------
    # AI DIET PLANNER
    # -------------------------
    st.subheader("🤖 Free AI Diet Planner")

    if st.button("🥗 Generate Diet Plan"):
        with st.spinner("Generating..."):
            plan = diet_agent(age, weight, height, gender, goal, all_foods, bmi)
            st.text(plan)
