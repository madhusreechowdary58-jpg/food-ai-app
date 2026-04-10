import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile
import os
import matplotlib.pyplot as plt
from langchain_community.llms import Ollama
import threading
import time

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI Food Health Analyzer", page_icon="🍔", layout="wide")

st.title("🍔 GenAI Food & Health Analyzer")
st.markdown("---")

# -------------------------
# SIDEBAR - USER INPUT
# -------------------------
st.sidebar.header("👤 User Profile")

age = st.sidebar.number_input("Age", 5, 80, value=25)
weight = st.sidebar.number_input("Weight (kg)", 20, 150, value=60)

height_ft = st.sidebar.number_input("Height (feet)", 1.0, 7.0, value=5.5, step=0.1)
height = round(height_ft * 30.48)

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
goal = st.sidebar.selectbox("Goal", ["Weight Loss", "Muscle Gain", "Maintain"])

# -------------------------
# INITIALIZE OLLAMA
# -------------------------
@st.cache_resource
def load_ollama():
    return Ollama(model="mistral", temperature=0.7)

try:
    llm = load_ollama()
    ollama_status = "✅ Ollama Connected"
except Exception as e:
    llm = None
    ollama_status = f"❌ Ollama Error"

st.sidebar.markdown("---")
st.sidebar.markdown(f"**🤖 AI Status:** {ollama_status}")

# -------------------------
# LOAD FOOD CLASSIFIER
# -------------------------
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="nateraw/food")

try:
    classifier = load_model()
    classifier_status = "✅ Model Loaded"
except:
    classifier = None
    classifier_status = "❌ Not Loaded"

st.sidebar.markdown(f"**📷 Classifier:** {classifier_status}")

# -------------------------
# BMI CALCULATION
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
# BMR CALCULATION
# -------------------------
def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        return 10*weight + 6.25*height - 5*age + 5
    else:
        return 10*weight + 6.25*height - 5*age - 161

# -------------------------
# RDA CALCULATION
# -------------------------
def get_rda(age, weight, height, gender):
    bmr = calculate_bmr(weight, height, age, gender)
    calories = round(bmr * 1.4)
    protein = round(0.8 * weight, 1)
    carbs = round((0.5 * calories) / 4, 1)
    fat = round((0.25 * calories) / 9, 1)
    return {"calories": calories, "protein": protein, "carbs": carbs, "fat": fat}

# -------------------------
# ESTIMATE NUTRITION
# -------------------------
def estimate_nutrition(food):
    estimates = {
        "burger": {"calories": 300, "protein": 15, "fat": 12, "carbs": 35, "vitamins": 10},
        "fries": {"calories": 250, "protein": 3, "fat": 12, "carbs": 33, "vitamins": 5},
        "pizza": {"calories": 270, "protein": 11, "fat": 10, "carbs": 36, "vitamins": 15},
        "salad": {"calories": 50, "protein": 2, "fat": 0, "carbs": 10, "vitamins": 40},
        "rice": {"calories": 130, "protein": 2, "fat": 0, "carbs": 28, "vitamins": 5},
        "chicken": {"calories": 165, "protein": 31, "fat": 3.6, "carbs": 0, "vitamins": 5},
        "apple": {"calories": 95, "protein": 0.5, "fat": 0.3, "carbs": 25, "vitamins": 20},
        "banana": {"calories": 105, "protein": 1.3, "fat": 0.4, "carbs": 27, "vitamins": 15},
        "egg": {"calories": 78, "protein": 6, "fat": 5, "carbs": 0.6, "vitamins": 10},
        "milk": {"calories": 103, "protein": 8, "fat": 2.4, "carbs": 12, "vitamins": 30},
        "dal": {"calories": 116, "protein": 9, "fat": 0.5, "carbs": 20, "vitamins": 15},
        "paneer": {"calories": 265, "protein": 14, "fat": 22, "carbs": 3, "vitamins": 8},
        "bread": {"calories": 80, "protein": 3, "fat": 1, "carbs": 15, "vitamins": 5},
    }
    food_lower = food.lower()
    for key, value in estimates.items():
        if key in food_lower:
            return value
    return {"calories": 100, "protein": 5, "fat": 3, "carbs": 15, "vitamins": 10}

# -------------------------
# CALCULATE TOTAL
# -------------------------
def calculate_total(foods):
    total = {"calories": 0, "protein": 0, "fat": 0, "carbs": 0, "vitamins": 0}
    details = []
    for food in foods:
        data = estimate_nutrition(food)
        details.append((food, data))
        for k in total:
            total[k] += data[k]
    return total, details

# -------------------------
# AI HEALTH AGENT
# -------------------------
def ai_health_agent(user_question, context):
    prompt = f"""You are a helpful health and nutrition AI assistant.
User Profile:
- Age: {context['age']}
- Weight: {context['weight']} kg
- Height: {context['height']} cm
- Gender: {context['gender']}
- Goal: {context['goal']}
- BMI: {context['bmi']} ({context['bmi_status']})
- Foods: {', '.join(context['foods'])}

Question: {user_question}

Provide a helpful, concise answer."""
    try:
        if llm is None:
            return "❌ Ollama not connected"
        return llm.invoke(prompt)
    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------
# TEXT TO AUDIO
# -------------------------
def text_to_audio(text):
    try:
        tts = gTTS(text)
        file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(file.name)
        return file.name
    except:
        return None

# -------------------------
# MAIN APP
# -------------------------
st.header("📸 Upload Your Food Images")

uploaded_files = st.file_uploader("Choose food images...", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files and classifier is not None:
    all_foods = []
    
    for file in uploaded_files:
        img = Image.open(file)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, width=150)
        with col2:
            with st.spinner(f"🔍 Analyzing {file.name}..."):
                res = classifier(img)
                food = res[0]["label"].replace("_", " ")
                all_foods.append(food)
                st.success(f"🍔 Detected: {food.title()}")

    if all_foods:
        total, details = calculate_total(all_foods)
        bmi, status = calculate_bmi(weight, height)
        rda = get_rda(age, weight, height, gender)
        
        ai_context = {
            "age": age, "weight": weight, "height": height,
            "gender": gender, "goal": goal, "bmi": bmi,
            "bmi_status": status, "foods": all_foods,
            "total_nutrients": total, "rda": rda
        }

        # NUTRIENT SUMMARY
        st.markdown("---")
        st.header("📊 Nutrient Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("🔥 Calories", f"{round(total['calories'])}/{rda['calories']}")
        with col2:
            st.metric("🥩 Protein", f"{round(total['protein'],1)}g/{rda['protein']}g")
        with col3:
            st.metric("🍞 Carbs", f"{round(total['carbs'],1)}g/{rda['carbs']}g")
        with col4:
            st.metric("🥑 Fat", f"{round(total['fat'],1)}g/{rda['fat']}g")
        with col5:
            st.metric("💊 Vitamins", f"{round(total['vitamins'])}/100")

        # NUTRIENT RINGS
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.subheader("📈 Nutrient Progress")
            carbs_p = min(total["carbs"] / rda["carbs"], 1) if rda["carbs"] > 0 else 0
            protein_p = min(total["protein"] / rda["protein"], 1) if rda["protein"] > 0 else 0
            fat_p = min(total["fat"] / rda["fat"], 1) if rda["fat"] > 0 else 0
            vit_p = min(total["vitamins"] / 100, 1)

            fig, ax = plt.subplots(figsize=(6,6))
            fig.patch.set_facecolor("#0E1117")
            ax.set_facecolor("#0E1117")
            progress = [carbs_p, protein_p, fat_p, vit_p]
            colors = ["#00C2FF", "#00FF7F", "#FF7F50", "#FFD700"]

            for i, p in enumerate(progress):
                ax.pie([p, 1-p], radius=1 - i*0.18, startangle=90, counterclock=False,
                       colors=[colors[i], "#1f1f1f"], wedgeprops=dict(width=0.13, edgecolor="none"))

            centre_circle = plt.Circle((0, 0), 0.35, color="#0E1117")
            ax.add_artist(centre_circle)
            ax.set(aspect="equal")
            ax.axis('off')
            st.pyplot(fig)
            st.markdown("🔵 Carbs | 🟢 Protein | 🟠 Fat | 🟡 Vitamins")

        # HEALTH PROFILE
        with col_right:
            st.subheader("🏥 Health Profile")
            h = height / 100
            min_w = round(18.5*(h*h),1)
            max_w = round(24.9*(h*h),1)
            bmr = calculate_bmr(weight, height, age, gender)
            st.write(f"**BMI:** {bmi} ({status})")
            st.write(f"**Ideal Weight:** {min_w}-{max_w} kg")
            st.write(f"**BMR:** {round(bmr)} kcal/day")
            st.write(f"**Goal:** {goal}")

        # AI CHAT
        st.markdown("---")
        st.header("🤖 AI Health Assistant")
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        user_q = st.chat_input("💬 Ask AI about your diet, nutrition, or health...")

        if user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.spinner("🤔 AI is thinking..."):
                ai_response = ai_health_agent(user_q, ai_context)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            st.rerun()

        # AUDIO
        st.subheader("🔊 Audio Health Advice")
        advice_text = f"Your BMI is {bmi}, which is {status}. "
        if bmi > 25:
            advice_text += "Consider reducing high-calorie foods. "
        else:
            advice_text += "Your weight is healthy. "
        advice_text += f"Today you ate {', '.join(all_foods)}."
        
        audio_file = text_to_audio(advice_text)
        if audio_file:
            st.audio(audio_file)
            def cleanup():
                time.sleep(2)
                try:
                    os.remove(audio_file)
                except:
                    pass
            threading.Thread(target=cleanup, daemon=True).start()

else:
    st.info("""
    ### 👋 Welcome to GenAI Food & Health Analyzer!
    
    **How to use:**
    1. Fill in your profile in the sidebar
    2. Upload food images above
    3. Chat with AI about your diet!
    
    **Requirements:**
    - Ollama must be running in background
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("📷 Food Classifier: Ready" if classifier else "❌ Not Loaded")
    with col2:
        st.success("🤖 AI Agent: Ready" if llm else "⚠️ Check Ollama")
