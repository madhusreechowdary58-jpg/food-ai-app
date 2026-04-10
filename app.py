import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile
import os
import matplotlib.pyplot as plt
import requests
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
# GROQ AI (Free Cloud AI)
# -------------------------
def get_ai_response(user_question, context):
    """Using Groq API - free and fast, with smart fallback"""
    
    # Get API key from secrets
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, 'secrets') else os.environ.get("GROQ_API_KEY", "")
    
    # If no API key, use smart fallback
    if not GROQ_API_KEY:
        return smart_fallback(context, user_question)
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    prompt = f"""You are a friendly health and nutrition AI assistant. Give practical, personalized advice.

USER PROFILE:
- Age: {context['age']}
- Weight: {context['weight']} kg
- Height: {context['height']} cm  
- Gender: {context['gender']}
- Goal: {context['goal']}
- BMI: {context['bmi']} ({context['bmi_status']})
- Today's Foods: {', '.join(context['foods'])}

QUESTION: {user_question}

Respond in 2-3 sentences, be helpful and specific to their profile."""

    data = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": 200
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            return smart_fallback(context, user_question)
    except:
        return smart_fallback(context, user_question)

def smart_fallback(context, question):
    """Smart fallback when no API available - gives different responses based on question"""
    
    bmi = context['bmi']
    status = context['bmi_status']
    goal = context['goal']
    foods = context['foods']
    foods_str = ', '.join(foods).lower()
    weight = context['weight']
    
    q = question.lower()
    
    # Weight loss questions
    if any(word in q for word in ['weight', 'lose', 'loss', 'slim', 'diet']):
        if bmi > 25:
            return f"For weight loss with your BMI of {bmi}, create a 300-500 calorie deficit. Avoid {foods_str} and add more vegetables, lean protein, and water. Walking 30 mins daily helps!"
        elif bmi < 18.5:
            return f"Your BMI is {bmi}, which is underweight. Focus on nutrient-dense foods like nuts, avocados, and protein. Don't restrict calories - eat more than you burn!"
        else:
            return f"Your BMI of {bmi} is healthy! For maintenance, balance calories with activity. Eat whole foods and limit processed items."
    
    # Protein questions
    if any(word in q for word in ['protein', 'muscle', 'gym', 'workout']):
        if goal == "Muscle Gain":
            return f"For muscle gain at {weight}kg, eat 1.6-2g protein per kg body weight. Good sources: eggs, chicken, dal, paneer. Combine with strength training!"
        else:
            return f"Protein is essential! Aim for {int(weight * 0.8)}g daily. Best sources: eggs, chicken, fish, dal, paneer, and legumes."
    
    # Food questions
    if any(word in q for word in ['food', 'eat', 'meal', 'healthy', 'unhealthy']):
        junk = [f for f in foods if any(x in f.lower() for x in ['burger', 'fries', 'pizza', 'donut', 'hotdog'])]
        if junk:
            return f"Foods like {', '.join(junk)} are high in unhealthy fats and calories. Replace with home-cooked meals, more vegetables, fruits, and lean proteins!"
        else:
            return f"Your food choices are reasonable! Keep eating balanced meals with proteins, complex carbs, and plenty of vegetables."
    
    # Exercise questions
    if any(word in q for word in ['exercise', 'workout', 'gym', 'run', 'walk', 'sport']):
        if goal == "Weight Loss":
            return "For weight loss, combine cardio (walking, running, cycling) with strength training. Start with 30 mins daily and gradually increase!"
        elif goal == "Muscle Gain":
            return "For muscle gain, focus on strength training - weight lifting, pushups, squats. Aim for 4-5 sessions weekly with protein-rich diet!"
        else:
            return "Regular exercise is great! Mix cardio and strength training. 150 mins weekly of moderate activity is recommended for maintaining health."
    
    # Calories questions
    if any(word in q for word in ['calorie', 'calories', 'energy']):
        return f"Your daily calorie needs depend on activity. At {weight}kg with your goal of {goal}, focus on whole foods rather than counting - quality matters!"
    
    # General health
    if any(word in q for word in ['health', 'good', 'better', 'improve']):
        return f"Your BMI is {bmi} ({status}). To improve health: eat balanced meals, stay active, sleep 7-8 hours, and drink plenty of water daily!"
    
    # Goal specific
    if any(word in q for word in ['goal', 'target', 'achieve']):
        if goal == "Weight Loss":
            return f"Your goal is {goal}. Create calorie deficit, eat protein-rich foods, avoid sugar, and exercise regularly. You got this! 💪"
        elif goal == "Muscle Gain":
            return f"Your goal is {goal}. Eat protein-rich diet, do strength training, rest properly. Progressive overload is key!"
        else:
            return f"Your goal is {goal}. Maintain balanced diet, stay active, and monitor your weight weekly. Consistency is key!"
    
    # Default - personalized based on their profile
    return f"Based on your profile (BMI: {bmi}, {status}, Goal: {goal}): Focus on whole foods, stay active, and maintain consistency. Small steps lead to big changes!"

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

st.sidebar.markdown("---")
st.sidebar.markdown(f"**📷 Classifier:** {classifier_status}")
st.sidebar.markdown("**🤖 AI:** Groq + Smart Fallback")

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
                ai_response = get_ai_response(user_q, ai_context)
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
    
    **Try asking:**
    - "How can I lose weight?"
    - "How much protein do I need?"
    - "What exercises should I do?"
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("📷 Food Classifier: Ready" if classifier else "❌ Not Loaded")
    with col2:
        st.info("🤖 AI: Smart Responses")
