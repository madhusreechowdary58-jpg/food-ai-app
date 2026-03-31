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
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🍔 GenAI Food Nutrition Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered food detection & health advisor</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------
# SIDEBAR
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

        for item in res["foods"][0]["foodNutrients"]:
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

    for f in foods:
        data = get_nutrition(f)
        details.append((f, data))
        for k in total:
            total[k] += data[k]

    return total, details

# -------------------------
# HUGGING FACE GENAI (MISTRAL)
# -------------------------
def generate_ai_response(foods, total, age, weight, goal):

    prompt = f"""
    You are a nutrition expert.

    Foods eaten: {foods}
    Nutrition: {total}
    Age: {age}, Weight: {weight}, Goal: {goal}

    Provide:
    1. Health analysis
    2. Nutritional issues
    3. Risks
    4. Personalized diet suggestions
    """

    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}

    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    output = response.json()

    if isinstance(output, list):
        return output[0]["generated_text"]
    else:
        return "AI response not available"

# -------------------------
# DEFICIENCY
# -------------------------
def analyze_nutrition(total):
    tips = []

    if total["protein"] < 50:
        tips.append("Protein low. Eat eggs, chicken, dal.")
    if total["carbs"] < 130:
        tips.append("Carbs low. Add rice or fruits.")
    if total["fat"] < 20:
        tips.append("Healthy fats low. Add nuts.")
    if total["calories"] < 500:
        tips.append("Calories low. Increase meal size.")

    if not tips:
        tips.append("Diet is balanced.")

    return tips

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
        for file in uploaded_files:
            img = Image.open(file)
            st.image(img)

            res = classifier(img)
            food = res[0]["label"].replace("_", " ")
            all_foods.append(food)

    with col2:
        for f in all_foods:
            st.success(f)

    total, details = calculate_total(all_foods)

    st.markdown("## 📊 Nutrition")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calories", total["calories"])
    c2.metric("Protein", total["protein"])
    c3.metric("Fat", total["fat"])
    c4.metric("Carbs", total["carbs"])

    df = pd.DataFrame.from_dict(total, orient='index', columns=['Value'])
    st.bar_chart(df)

    # AI
    st.markdown("## 🤖 AI Analysis")
    ai_text = generate_ai_response(all_foods, total, age, weight, goal)
    st.write(ai_text)

    # Audio
    st.markdown("## 🔊 Audio Advice")
    tips = analyze_nutrition(total)

    summary = f"You ate {len(all_foods)} foods. Calories {total['calories']}. "
    for t in tips:
        summary += t + " "

    audio = text_to_audio(summary)
    st.audio(audio)

    if os.path.exists(audio):
        os.remove(audio)
