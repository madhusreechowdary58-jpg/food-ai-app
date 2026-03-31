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
st.set_page_config(page_title="GenAI Food Analyzer", layout="wide")

st.markdown("""
<style>
.main {background-color: #f5f7fa;}
h1 {text-align:center;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🍔 GenAI Food Nutrition Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered food detection & smart health advisor</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# USER INPUT
# -------------------------
st.sidebar.title("👤 User Profile")
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
# API KEYS
# -------------------------
HF_API_KEY = st.secrets["HF_API_KEY"]
USDA_API_KEY = st.secrets["USDA_API_KEY"]

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

        for key in total:
            total[key] += data[key]

    return total, details

# -------------------------
# DEFICIENCY ANALYSIS
# -------------------------
def analyze_deficiency(total):
    advice = []

    if total["protein"] < 50:
        advice.append("Protein is low. Eat eggs, chicken, dal or paneer.")
    if total["carbs"] < 130:
        advice.append("Carbohydrates are low. Add rice, fruits or bread.")
    if total["fat"] < 20:
        advice.append("Healthy fats are low. Add nuts and seeds.")
    if total["calories"] < 500:
        advice.append("Calories are low. Increase balanced meals.")

    if not advice:
        advice.append("Your diet looks balanced.")

    return advice

# -------------------------
# GEN AI (PERSONALIZED)
# -------------------------
def generate_ai_response(foods, total, age, weight, goal):

    deficiency = analyze_deficiency(total)

    prompt = f"""
    You are a professional nutritionist.

    User Details:
    Age: {age}
    Weight: {weight}
    Goal: {goal}

    Foods consumed: {foods}
    Total nutrition: {total}

    Nutrient deficiencies: {deficiency}

    Provide:
    1. Health analysis
    2. Is diet suitable for goal?
    3. Risks
    4. What nutrients are lacking
    5. What foods to eat next
    """

    API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=20)
        output = response.json()

        if isinstance(output, list):
            return output[0]["generated_text"]

    except:
        pass

    # fallback
    return f"""
    Based on your goal ({goal}), your diet needs improvement.
    Add protein, vegetables, and balanced meals.
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

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Images")
        for i, file in enumerate(uploaded_files):
            img = Image.open(file)
            st.image(img, caption=f"Image {i+1}")

            result = classifier(img)
            food = result[0]["label"].replace("_", " ")
            all_foods.append(food)

    with col2:
        st.subheader("🍔 Detected Foods")
        for f in all_foods:
            st.success(f)

    # -------------------------
    # NUTRITION
    # -------------------------
    total, details = calculate_total(all_foods)

    st.markdown("## 🍽 Nutrition per Food")
    for food, data in details:
        st.write(f"### {food}")
        st.write(data)
        st.markdown("---")

    st.markdown("## 📊 Total Nutrition")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calories", total["calories"])
    c2.metric("Protein", total["protein"])
    c3.metric("Fat", total["fat"])
    c4.metric("Carbs", total["carbs"])

    df = pd.DataFrame.from_dict(total, orient='index', columns=['Value'])
    st.bar_chart(df)

    # -------------------------
    # GEN AI
    # -------------------------
    st.markdown("## 🤖 Health Analysis")

    with st.spinner("Generating AI insights..."):
        ai_response = generate_ai_response(all_foods, total, age, weight, goal)

    st.write(ai_response)

    # -------------------------
    # AUDIO
    # -------------------------
    st.markdown("## 🔊 Smart Audio Advice")

    deficiency = analyze_deficiency(total)

    summary = f"Based on your goal {goal}, "

    for tip in deficiency:
        summary += tip + " "

    audio = text_to_audio(summary)
    st.audio(audio)

    if os.path.exists(audio):
        os.remove(audio)
