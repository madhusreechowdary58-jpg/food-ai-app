import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
from gtts import gTTS
import tempfile
import os
import requests
import openai

# -------------------------
# PAGE CONFIG + STYLE
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
# API KEY (Streamlit Secrets)
# -------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

# -------------------------
# NUTRITION (Fallback)
# -------------------------
def get_nutrition(food):
    sample_data = {
        "pizza": {"calories": 285, "protein": 12, "fat": 10, "carbs": 36},
        "burger": {"calories": 295, "protein": 17, "fat": 12, "carbs": 30},
        "french fries": {"calories": 365, "protein": 4, "fat": 17, "carbs": 48},
        "sandwich": {"calories": 200, "protein": 8, "fat": 6, "carbs": 28}
    }
    return sample_data.get(food.lower(), {"calories": 150, "protein": 5, "fat": 5, "carbs": 20})

# -------------------------
# TOTAL CALCULATION
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
# GEN AI RESPONSE
# -------------------------
def generate_ai_response(foods, total, age, weight, goal):
    prompt = f"""
    Foods eaten: {foods}
    Nutrition: {total}
    Age: {age}, Weight: {weight}, Goal: {goal}

    Give health analysis and diet suggestions.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response['choices'][0]['message']['content']

# -------------------------
# NUTRIENT DEFICIENCY ANALYSIS
# -------------------------
def analyze_nutrition(total):
    advice = []

    if total["protein"] < 50:
        advice.append("Protein is low. Eat eggs, chicken, lentils or paneer.")

    if total["carbs"] < 130:
        advice.append("Carbohydrates are low. Add rice, bread, fruits or oats.")

    if total["fat"] < 20:
        advice.append("Healthy fats are low. Add nuts, seeds or avocado.")

    if total["calories"] < 500:
        advice.append("Calories are low. Increase portion size or balanced meals.")

    if not advice:
        advice.append("Your diet looks balanced.")

    return advice

# -------------------------
# AUDIO FUNCTION
# -------------------------
def text_to_audio(text):
    tts = gTTS(text=text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

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
            st.image(img, caption=f"Image {i+1}", use_column_width=True)

            results = classifier(img)
            food = results[0]["label"].replace("_", " ")
            all_foods.append(food)

    with col2:
        st.subheader("🍔 Detected Foods")
        for f in all_foods:
            st.success(f)

    # Nutrition
    total, details = calculate_total(all_foods)

    st.markdown("## 📊 Nutrition Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calories", total["calories"])
    c2.metric("Protein", total["protein"])
    c3.metric("Fat", total["fat"])
    c4.metric("Carbs", total["carbs"])

    df = pd.DataFrame.from_dict(total, orient='index', columns=['Value'])
    st.bar_chart(df)

    # AI
    st.markdown("## 🤖 GenAI Analysis")
    ai_response = generate_ai_response(all_foods, total, age, weight, goal)
    st.write(ai_response)

    # Audio Advice
    st.markdown("## 🔊 Smart Audio Advice")

    advice = analyze_nutrition(total)

    summary = f"You consumed {len(all_foods)} items. Calories are {total['calories']}. "

    for tip in advice:
        summary += tip + " "

    if goal == "Muscle Gain":
        summary += "Increase protein intake."
    elif goal == "Weight Loss":
        summary += "Reduce calorie intake."

    audio_file = text_to_audio(summary)
    st.audio(audio_file)

    if os.path.exists(audio_file):
        os.remove(audio_file)

# -------------------------
# CHATBOT
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("## 💬 Chat with AI")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about your diet...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    prompt = f"User foods: {all_foods if 'all_foods' in locals() else []}, question: {user_input}"

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    reply = response['choices'][0]['message']['content']

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply)
