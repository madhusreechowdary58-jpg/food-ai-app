import streamlit as st
from transformers import pipeline
from PIL import Image
from gtts import gTTS
import tempfile
import os
import requests
import matplotlib.pyplot as plt
import google.generativeai as genai
import json

# -------------------------
# CACHE DIR
# -------------------------
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf_cache"

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="AI Food Health Analyzer", layout="wide")

st.title("🍔 GenAI Food & Health Analyzer")
st.markdown("---")

# -------------------------
# API KEYS
# -------------------------
USDA_API_KEY = st.secrets["USDA_API_KEY"]

# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_food_model():
    return pipeline("image-classification", model="nateraw/food")

@st.cache_resource
def load_text_model():
    return pipeline("text2text-generation", model="google/flan-t5-base")

@st.cache_resource
def load_zs_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_food_model()
text_model = load_text_model()
zs_classifier = load_zs_model()

# -------------------------
# USER INPUT SIDEBAR
# -------------------------
st.sidebar.header("👤 User Profile")

st.sidebar.markdown("### 📷 Auto-detect from Photo (optional)")
person_file = st.sidebar.file_uploader(
    "Upload your photo", type=["jpg", "jpeg", "png"], key="person_img"
)

if person_file and "person_estimates" not in st.session_state:
    with st.spinner("Analyzing your photo with AI..."):
        try:
            img = Image.open(person_file)

            # Zero-shot classify age group
            zs_age = zs_classifier(
                "person in photo",
                candidate_labels=["child under 18", "young adult 18-30", "adult 30-50", "senior above 50"]
            )
            age_label = zs_age["labels"][0]

            if "child" in age_label:
                est_age = 12
            elif "young" in age_label:
                est_age = 24
            elif "adult" in age_label:
                est_age = 38
            else:
                est_age = 55

            st.session_state["person_estimates"] = {
                "age": est_age,
                "weight_kg": 70,
                "height_ft": 5.5,
                "gender": "Male",
                "confidence": "low"
            }
            st.sidebar.info("⚠️ Age group estimated by AI. Please adjust weight, height and gender manually.")

        except Exception as e:
            st.sidebar.error(f"Could not analyze image: {e}")

if st.sidebar.button("🔄 Clear detected values"):
    st.session_state.pop("person_estimates", None)

_est = st.session_state.get("person_estimates", {})

age = st.sidebar.number_input("Age", 5, 80, value=int(_est.get("age", 25)))
weight = st.sidebar.number_input("Weight (kg)", 20, 150, value=int(_est.get("weight_kg", 70)))
height_ft = st.sidebar.number_input(
    "Height (feet)", 1.0, 7.0,
    value=float(_est.get("height_ft", 5.5)), step=0.1
)
height = round(height_ft * 30.48)
gender = st.sidebar.selectbox(
    "Gender", ["Male", "Female"],
    index=0 if _est.get("gender", "Male") == "Male" else 1
)
goal = st.sidebar.selectbox("Goal", ["Weight Loss", "Muscle Gain", "Maintain"])

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
        feedback.append("You are overweight. High-calorie foods increase fat storage and may lead to heart disease.")
    elif bmi < 18.5:
        feedback.append("You are underweight. Low-nutrient foods will not help in healthy weight gain.")
    else:
        feedback.append("Your BMI is normal. Maintaining diet quality is important.")

    if junk_items:
        feedback.append(f"The foods {', '.join(junk_items)} are unhealthy.")
        feedback.append("These foods contain high unhealthy fats, sugar, and low nutrients.")
        if age < 18:
            feedback.append("May affect growth and immunity.")
        elif age <= 40:
            feedback.append("May lead to weight gain and poor metabolism.")
        else:
            feedback.append("Increases heart and cholesterol risks.")
        if goal == "Weight Loss":
            feedback.append("Not suitable for weight loss.")
        if goal == "Muscle Gain":
            feedback.append("Does not support muscle growth.")
        feedback.append("Avoid fried and processed foods.")
        feedback.append("Eat vegetables, fruits, whole grains, and proteins.")
    else:
        feedback.append("Foods are healthy in moderation.")

    return feedback

# -------------------------
# GOAL SUITABILITY
# -------------------------
def evaluate_goal(bmi, goal):
    if bmi < 18.5:
        if goal == "Weight Loss":
            return "❌ You are underweight. Weight loss is not recommended."
        else:
            return "✅ Your goal is appropriate."
    elif bmi > 25:
        if goal == "Muscle Gain":
            return "⚠️ You are overweight. Focus on fat loss first."
        elif goal == "Maintain":
            return "⚠️ Consider weight loss instead."
        else:
            return "✅ Your goal aligns with your health."
    else:
        return "✅ Your goal is suitable."

# -------------------------
# ANALYSIS
# -------------------------
def generate_analysis(foods, total, age, weight, height, gender, goal):
    bmi, status = calculate_bmi(weight, height)
    rda = get_rda(age, weight, height, gender)
    deficiency = analyze_deficiency(total, rda)
    feedback = evaluate_food_health(foods, total, bmi, goal, age)
    h = height / 100
    min_w = round(18.5*(h*h), 1)
    max_w = round(24.9*(h*h), 1)
    bmr = calculate_bmr(weight, height, age, gender)

    return f"""
### 🧠 Personalized Health Analysis

👤 Age: {age}  
⚧ Gender: {gender}

📏 Height: {round(height/30.48, 1)} ft

⚖️ BMI: {bmi} → {status}  
Ideal Weight: {min_w}-{max_w} kg

🔥 BMR: {round(bmr)} kcal  

📊 Recommended:
Calories: {rda['calories']}
Protein: {rda['protein']}
Fat: {rda['fat']}
Carbs: {rda['carbs']}

🍔 Foods: {foods}

📊 Your Intake:
Calories: {round(total['calories'], 1)}
Protein: {round(total['protein'], 1)}
Fat: {round(total['fat'], 1)}
Carbs: {round(total['carbs'], 1)}

🧾 Food Suitability & Health Impact:
{chr(10).join(feedback)}

⚠️ Nutritional Feedback:
{', '.join(deficiency)}
"""

# -------------------------
# AI MEAL PLAN - Flan-T5
# -------------------------
def generate_meal_plan_ai(goal, bmi_status, age):
    prompt = f"Suggest a healthy one day meal plan for a {age} year old with BMI status {bmi_status} and goal {goal}. Include breakfast, lunch, snack and dinner."
    result = text_model(prompt, max_length=200)
    return result[0]["generated_text"]

# -------------------------
# AI HEALTH ADVICE - Flan-T5
# -------------------------
def generate_health_advice_ai(bmi_status, goal, foods):
    prompt = f"Give health advice for a person who is {bmi_status}, wants to {goal} and ate {', '.join(foods)} today."
    result = text_model(prompt, max_length=200)
    return result[0]["generated_text"]

# -------------------------
# AI FOOD RISK - Zero Shot
# -------------------------
def classify_food_risk(foods):
    food_str = ", ".join(foods)
    labels = ["very healthy", "moderately healthy", "unhealthy", "very unhealthy"]
    result = zs_classifier(food_str, candidate_labels=labels)
    return result["labels"][0], round(result["scores"][0] * 100, 1)

# -------------------------
# AI DIET SENTIMENT - Zero Shot
# -------------------------
def analyze_diet_sentiment(total, rda):
    summary = (
        f"calories {round(total['calories'])} out of {rda['calories']} recommended, "
        f"protein {round(total['protein'])} out of {rda['protein']}, "
        f"fat {round(total['fat'])} out of {rda['fat']}"
    )
    labels = ["balanced diet", "poor diet", "excessive diet", "deficient diet"]
    result = zs_classifier(summary, candidate_labels=labels)
    return result["labels"][0], round(result["scores"][0] * 100, 1)

# -------------------------
# AUDIO
# -------------------------
def text_to_audio(text):
    tts = gTTS(text)
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(file.name)
    return file.name

# -------------------------
# MAIN - FOOD IMAGE UPLOAD
# -------------------------
st.markdown("## 🍽️ Food Analysis")
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
        st.write(
            f"{f} → Carbs:{round(d['carbs'],1)}, "
            f"Protein:{round(d['protein'],1)}, "
            f"Fat:{round(d['fat'],1)}, "
            f"Vitamins:{round(d['vitamins'],1)}"
        )

    # TOTAL + RINGS
    st.subheader("📊 Total Nutrient Consumption")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Carbs", round(total["carbs"], 1))
        st.metric("Protein", round(total["protein"], 1))
        st.metric("Fat", round(total["fat"], 1))
        st.metric("Vitamins", round(total["vitamins"], 1))

    with col2:
        st.markdown("### 🟢 Nutrient Activity Rings")
        rda = get_rda(age, weight, height, gender)

        carbs_p = min(total["carbs"] / rda["carbs"], 1)
        protein_p = min(total["protein"] / rda["protein"], 1)
        fat_p = min(total["fat"] / rda["fat"], 1)
        vit_p = min(total["vitamins"] / 100, 1)

        progress = [carbs_p, protein_p, fat_p, vit_p]
        colors = ["#00C2FF", "#00FF7F", "#FF7F50", "#FFD700"]

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor("#0E1117")
        ax.set_facecolor("#0E1117")

        for i, p in enumerate(progress):
            ax.pie(
                [p, 1-p],
                radius=1 - i*0.18,
                startangle=90,
                counterclock=False,
                colors=[colors[i], "#1f1f1f"],
                wedgeprops=dict(width=0.13, edgecolor="none")
            )

        centre_circle = plt.Circle((0, 0), 0.35, color="#0E1117")
        ax.add_artist(centre_circle)
        ax.set(aspect="equal")
        ax.axis('off')
        st.pyplot(fig)

        st.markdown("### 🏷️ Nutrient Legend")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("🔵 **Carbs**")
        with c2:
            st.markdown("🟢 **Protein**")
        with c3:
            st.markdown("🟠 **Fat**")
        with c4:
            st.markdown("🟡 **Vitamins**")

    # ANALYSIS
    st.subheader("🤖 Health Analysis")
    bmi, status = calculate_bmi(weight, height)
    analysis = generate_analysis(all_foods, total, age, weight, height, gender, goal)
    st.write(analysis)

    # GOAL
    st.subheader("🎯 Goal Suitability Analysis")
    st.info(evaluate_goal(bmi, goal))

    # -------------------------
    # AI MEAL PLAN
    # -------------------------
    st.subheader("🗓️ AI Meal Plan (Flan-T5)")
    with st.spinner("Generating personalized meal plan..."):
        meal = generate_meal_plan_ai(goal, status, age)
        st.info(meal)

    # -------------------------
    # AI HEALTH ADVICE
    # -------------------------
    st.subheader("💡 AI Health Advice (Flan-T5)")
    with st.spinner("Generating health advice..."):
        advice = generate_health_advice_ai(status, goal, all_foods)
        st.success(advice)

    # -------------------------
    # AI FOOD RISK
    # -------------------------
    st.subheader("⚠️ AI Food Risk Classification (Zero-Shot)")
    with st.spinner("Classifying food risk..."):
        risk_label, risk_score = classify_food_risk(all_foods)
        if "very healthy" in risk_label:
            st.success(f"Your meal is: **{risk_label}** ({risk_score}% confidence)")
        elif "moderately" in risk_label:
            st.warning(f"Your meal is: **{risk_label}** ({risk_score}% confidence)")
        else:
            st.error(f"Your meal is: **{risk_label}** ({risk_score}% confidence)")

    # -------------------------
    # AI DIET SENTIMENT
    # -------------------------
    st.subheader("📊 AI Diet Sentiment (Zero-Shot)")
    with st.spinner("Analyzing diet balance..."):
        sentiment, conf = analyze_diet_sentiment(total, rda)
        if "balanced" in sentiment:
            st.success(f"Diet Analysis: **{sentiment}** ({conf}% confidence)")
        elif "poor" in sentiment or "deficient" in sentiment:
            st.warning(f"Diet Analysis: **{sentiment}** ({conf}% confidence)")
        else:
            st.error(f"Diet Analysis: **{sentiment}** ({conf}% confidence)")

    # AUDIO
    st.subheader("🔊 Smart Audio Advice")
    feedback = evaluate_food_health(all_foods, total, bmi, goal, age)
    audio_text = "Health advice: "
    for f in feedback:
        audio_text += f + " "

    audio = text_to_audio(audio_text)
    st.audio(audio)

    if os.path.exists(audio):
        os.remove(audio)
