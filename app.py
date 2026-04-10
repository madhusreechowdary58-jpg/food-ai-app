import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile
import os
import matplotlib.pyplot as plt

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
# LOAD FOOD CLASSIFIER
# -------------------------
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="nateraw/food")

classifier = load_model()

st.sidebar.markdown("---")
st.sidebar.success("📷 Classifier: Ready")

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
        "burger": {"calories": 300, "protein": 15, "fat": 12, "carbs": 35},
        "fries": {"calories": 250, "protein": 3, "fat": 12, "carbs": 33},
        "pizza": {"calories": 270, "protein": 11, "fat": 10, "carbs": 36},
        "salad": {"calories": 50, "protein": 2, "fat": 0, "carbs": 10},
        "rice": {"calories": 130, "protein": 2, "fat": 0, "carbs": 28},
        "chicken": {"calories": 165, "protein": 31, "fat": 3.6, "carbs": 0},
        "apple": {"calories": 95, "protein": 0.5, "fat": 0.3, "carbs": 25},
        "banana": {"calories": 105, "protein": 1.3, "fat": 0.4, "carbs": 27},
        "egg": {"calories": 78, "protein": 6, "fat": 5, "carbs": 0.6},
        "milk": {"calories": 103, "protein": 8, "fat": 2.4, "carbs": 12},
        "dal": {"calories": 116, "protein": 9, "fat": 0.5, "carbs": 20},
        "paneer": {"calories": 265, "protein": 14, "fat": 22, "carbs": 3},
        "bread": {"calories": 80, "protein": 3, "fat": 1, "carbs": 15},
    }
    food_lower = food.lower()
    for key, value in estimates.items():
        if key in food_lower:
            return value
    return {"calories": 100, "protein": 5, "fat": 3, "carbs": 15}

# -------------------------
# SMART AI RESPONSE
# -------------------------
def get_ai_response(question, context):
    """Generate personalized AI response based on question"""
    
    bmi = context['bmi']
    status = context['bmi_status']
    goal = context['goal']
    weight = context['weight']
    age = context['age']
    foods = context['foods']
    foods_str = ', '.join(foods).lower()
    
    q = question.lower()
    
    # Weight loss
    if any(w in q for w in ['lose', 'loss', 'weight', 'slim', 'fat']):
        if bmi > 25:
            return f"🔥 For weight loss with BMI {bmi}: Create 300-500 cal deficit. Skip {foods_str}. Eat protein-rich foods, vegetables, and drink 3L water daily. Walk 10,000 steps!"
        elif bmi < 18.5:
            return f"⚠️ You're underweight (BMI {bmi}). Don't lose weight! Eat calorie-dense foods: nuts, peanut butter, bananas, full-fat milk. Lift weights to build muscle."
        else:
            return f"✅ Your BMI {bmi} is normal. For maintenance: balanced meals, exercise 3x/week, avoid processed foods. You're doing great!"
    
    # Protein / muscle
    if any(w in q for w in ['protein', 'muscle', 'gym', 'workout', 'strength']):
        protein_needed = int(weight * 1.6)
        return f"💪 For {goal}: Eat {protein_needed}g protein daily! Sources: 4 eggs, 200g chicken, 100g paneer, 1 cup dal. Workout: strength training 4x/week. Sleep 8 hours!"
    
    # Food questions
    if any(w in q for w in ['eat', 'food', 'meal', 'healthy', 'unhealthy', 'good']):
        junk = [f for f in foods if any(x in f.lower() for x in ['burger', 'fries', 'pizza', 'donut', 'hotdog'])]
        if junk:
            return f"🍔 {', '.join(junk.title())} aren't ideal. They have lots of unhealthy fats and sugar. Replace with: grilled chicken, brown rice, vegetables, fruits, and home-cooked meals!"
        else:
            return f"🥗 Your food choices look good! Keep eating whole foods: proteins, complex carbs, and plenty of vegetables. Variety is key for nutrients!"
    
    # Exercise
    if any(w in q for w in ['exercise', 'run', 'walk', 'cardio', 'yoga']):
        if goal == "Weight Loss":
            return "🏃 For weight loss: Walk 30 mins daily + strength training 3x/week. Start slow, gradually increase intensity. Swimming and cycling are great too!"
        elif goal == "Muscle Gain":
            return "🏋️ For muscle gain: Weight lifting 4-5x/week. Focus on compound movements: squats, deadlifts, bench press. Rest muscles 48 hours between workouts."
        else:
            return "🏃 For maintenance: Mix of cardio + strength. 150 mins moderate exercise weekly. Walking, swimming, cycling - pick what you enjoy!"
    
    # Calories
    if any(w in q for w in ['calorie', 'energy']):
        return f"⚡ At {weight}kg with goal '{goal}': Focus on food QUALITY not just calories. Whole foods keep you full longer. Avoid empty calories from sugar and fried foods!"
    
    # General health
    if any(w in q for w in ['health', 'better', 'improve', 'tip', 'advice']):
        return f"💡 Health tips for you: 1) Drink 3L water daily 2) Sleep 7-8 hours 3) Eat protein with every meal 4) Walk 10,000 steps 5) Limit sugar and processed foods!"
    
    # Goal specific
    if any(w in q for w in ['goal', 'target']):
        if goal == "Weight Loss":
            return f"🎯 Your goal: Weight Loss. Action plan: Eat 300-500 cal less than burn, protein-rich diet, cardio + strength training. You CAN do it! 💪"
        elif goal == "Muscle Gain":
            return f"🎯 Your goal: Muscle Gain. Action plan: Eat 200-300 cal surplus, high protein (1.6g/kg), strength train hard. Progressive overload = gains! 💪"
        else:
            return f"🎯 Your goal: Maintain. Action plan: Balance intake with activity, eat whole foods, exercise regularly, monitor weekly. Consistency wins! 💪"
    
    # Default - make it dynamic
    responses = [
        f"Based on your profile: BMI {bmi} ({status}) and goal '{goal}'. Focus on whole foods, stay active, and be consistent! Small changes = big results!",
        f"💡 Tip: With BMI {bmi}, prioritize {('protein-rich foods and cardio' if bmi > 25 else 'balanced nutrition and regular exercise')}. You've got this!",
        f"🌟 Remember: Consistency over perfection! Even small steps daily lead to big changes over time. Start with one healthy habit today!"
    ]
    
    import hashlib
    # Different response based on question length
    return responses[len(q) % 3]

# -------------------------
# MAIN APP
# -------------------------
st.header("📸 Upload Your Food Images")

uploaded_files = st.file_uploader("Choose food images...", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if uploaded_files:
    all_foods = []
    
    for file in uploaded_files:
        img = Image.open(file)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img, width=150)
        with col2:
            with st.spinner(f"🔍 Analyzing..."):
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
            "bmi_status": status, "foods": all_foods
        }

        # NUTRIENT SUMMARY
        st.markdown("---")
        st.header("📊 Nutrient Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔥 Calories", f"{round(total['calories'])}/{rda['calories']}")
        with col2:
            st.metric("🥩 Protein", f"{round(total['protein'],1)}g/{rda['protein']}g")
        with col3:
            st.metric("🍞 Carbs", f"{round(total['carbs'],1)}g/{rda['carbs']}g")
        with col4:
            st.metric("🥑 Fat", f"{round(total['fat'],1)}g/{rda['fat']}g")

        # HEALTH PROFILE
        st.subheader("🏥 Your Health Profile")
        h = height / 100
        min_w = round(18.5*(h*h),1)
        max_w = round(24.9*(h*h),1)
        bmr = calculate_bmr(weight, height, age, gender)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"**BMI:** {bmi}\n({status})")
        with col2:
            st.info(f"**Ideal:**\n{min_w}-{max_w} kg")
        with col3:
            st.info(f"**BMR:**\n{round(bmr)} kcal")
        with col4:
            st.info(f"**Goal:**\n{goal}")

        # AI CHAT
        st.markdown("---")
        st.header("🤖 AI Health Assistant")
        
        # Clear chat when new images uploaded
        if 'current_foods' not in st.session_state or st.session_state.current_foods != all_foods:
            st.session_state.messages = []
            st.session_state.current_foods = all_foods
        
        # Show messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat input
        if prompt := st.chat_input("💬 Ask me anything about your diet, health, or nutrition!"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate AI response
            ai_reply = get_ai_response(prompt, ai_context)
            
            # Add AI response
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})
            
            # Force refresh
            st.rerun()

else:
    st.info("👆 Upload food images to get started!")
    st.markdown("""
    ### Features:
    - 📸 Food Image Recognition
    - 📊 Nutrient Analysis  
    - 🏥 Health Metrics (BMI, BMR)
    - 🤖 AI Health Assistant
    
    ### Try asking:
    - "How can I lose weight?"
    - "How much protein do I need?"
    - "What exercises should I do?"
    - "Are my foods healthy?"
    """)
