import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile
import os
import json
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
def load_classifier():
    return pipeline("image-classification", model="nateraw/food")

@st.cache_resource
def load_chatbot():
    return pipeline("text-generation", model="gpt2")

classifier = load_classifier()
chatbot = load_chatbot()

# -------------------------
# LOCAL NUTRITION DATA
# -------------------------
nutrition_data = {
    "rice": {"calories":130,"protein":2.7,"fat":0.3,"carbs":28},
    "egg": {"calories":155,"protein":13,"fat":11,"carbs":1.1},
    "burger": {"calories":295,"protein":17,"fat":12,"carbs":30},
    "pizza": {"calories":266,"protein":11,"fat":10,"carbs":33},
    "fries": {"calories":312,"protein":3.4,"fat":15,"carbs":41},
    "apple": {"calories":52,"protein":0.3,"fat":0.2,"carbs":14},
    "banana": {"calories":89,"protein":1.1,"fat":0.3,"carbs":23}
}

# -------------------------
# FUNCTIONS
# -------------------------
def get_nutrition(food):
    food = food.lower()
    return nutrition_data.get(food, {"calories":100,"protein":2,"fat":2,"carbs":10})

def calculate_bmi(weight, height):
    h = height / 100
    bmi = weight / (h*h)
    if bmi < 18.5:
        return round(bmi,2),"Underweight"
    elif bmi < 25:
        return round(bmi,2),"Normal"
    elif bmi < 30:
        return round(bmi,2),"Overweight"
    else:
        return round(bmi,2),"Obese"

def calculate_total(foods):
    total = {"calories":0,"protein":0,"fat":0,"carbs":0}
    for f in foods:
        data = get_nutrition(f)
        for k in total:
            total[k] += data[k]
    return total

# -------------------------
# IMAGE UPLOAD
# -------------------------
uploaded_files = st.file_uploader("📸 Upload Food Images", accept_multiple_files=True)

if uploaded_files:
    all_foods = []

    for file in uploaded_files:
        img = Image.open(file)
        st.image(img)

        res = classifier(img)
        food = res[0]["label"].replace("_"," ")
        all_foods.append(food)

    total = calculate_total(all_foods)

    st.subheader("🍔 Detected Foods")
    for f in all_foods:
        st.success(f)

    st.subheader("📊 Total Nutrients")
    st.write(total)

    # -------------------------
    # BMI
    # -------------------------
    bmi, status = calculate_bmi(weight, height)
    st.subheader("⚖️ BMI")
    st.write(f"{bmi} → {status}")

    # -------------------------
    # GRAPH
    # -------------------------
    st.subheader("📈 Nutrient Graph")

    labels = list(total.keys())
    values = list(total.values())

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    st.pyplot(fig)

    # -------------------------
    # AUDIO
    # -------------------------
    text = f"Your BMI is {bmi}. You are {status}"
    tts = gTTS(text)
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(file.name)
    st.audio(file.name)

    if os.path.exists(file.name):
        os.remove(file.name)

# -------------------------
# AI CHATBOT
# -------------------------
st.subheader("🤖 AI Nutrition Chatbot")

user_query = st.text_input("Ask about food, diet, health:")

if st.button("Ask AI"):
    if user_query:
        prompt = f"You are a health expert. Give simple diet advice.\nUser: {user_query}\nAI:"

        response = chatbot(
            prompt,
            max_length=100,
            do_sample=True
        )

        answer = response[0]['generated_text'].replace(prompt, "")
        st.success(answer)
