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
st.set_page_config(page_title="AI Food Health Analyzer", layout="wide")

st.title("🍔 AI Food & Health Analyzer")
st.markdown("---")

# -------------------------
# USER INPUT
# -------------------------
st.sidebar.header("👤 User Details")

age = st.sidebar.number_input("Age", 5, 80)
weight = st.sidebar.number_input("Weight (kg)", 20, 150)
height = st.sidebar.number_input("Height (cm)", 100, 200)
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
# CLEAN FOOD NAMES
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
# GET NUTRITION (USDA)
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
# TOTAL CALCULATION
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
# -------------------------
# ADVANCED AI LOGIC (STRONG + REALISTIC)
# -------------------------
def generate_analysis(foods, total, age, weight, height, gender, goal):

    bmi, status = calculate_bmi(weight, height)
    deficiency = analyze_deficiency(total)

    # -------------------------
    # IDEAL WEIGHT RANGE (approx)
    # -------------------------
    h = height / 100
    min_w = round(18.5 * (h*h), 1)
    max_w = round(24.9 * (h*h), 1)

    # -------------------------
    # GOAL VALIDATION
    # -------------------------
    if status == "Normal":
        if goal == "Weight Loss":
            goal_msg = "You already have a healthy weight. Weight loss is not necessary."
        elif goal == "Muscle Gain":
            goal_msg = "Muscle gain is appropriate if you focus on strength and protein intake."
        else:
            goal_msg = "Maintaining your current weight is a good choice."
    elif status == "Underweight":
        if goal == "Weight Loss":
            goal_msg = "Weight loss is dangerous for your condition."
        else:
            goal_msg = "You should focus on healthy weight gain."
    elif status == "Overweight":
        if goal == "Muscle Gain":
            goal_msg = "You should reduce fat first before focusing on muscle gain."
        else:
            goal_msg = "Weight loss is recommended for better health."
    else:
        goal_msg = "Strict weight reduction is required for health improvement."

    # -------------------------
    # AGE BASED LOGIC
    # -------------------------
    if age < 18:
        age_msg = "You are in a growth stage. Balanced nutrition is very important."
    elif age <= 40:
        age_msg = "You are in an active age group. Maintain a balanced diet."
    else:
        age_msg = "Focus on heart health and controlled diet."

    # -------------------------
    # GENDER BASED NOTE
    # -------------------------
    if gender == "Male":
        gender_msg = "Males generally require higher calories and protein."
    else:
        gender_msg = "Females require balanced nutrition with proper iron intake."

    # -------------------------
    # CALORIE INTERPRETATION
    # -------------------------
    if total["calories"] < 500:
        calorie_msg = "Your calorie intake is too low."
    elif total["calories"] > 900:
        calorie_msg = "Your calorie intake is high."
    else:
        calorie_msg = "Your calorie intake is moderate."

    # -------------------------
    # FINAL OUTPUT
    # -------------------------
    return f"""
### 🧠 Personalized Health Analysis

👤 Age: {age}  
➡️ {age_msg}

⚧ Gender: {gender}  
➡️ {gender_msg}

⚖️ BMI: {bmi} → {status}  
➡️ Ideal weight range: {min_w} kg to {max_w} kg

🎯 Goal Evaluation:  
➡️ {goal_msg}

🍔 Foods Consumed: {foods}

📊 Nutrition Summary:
- Calories: {total['calories']} → {calorie_msg}
- Protein: {total['protein']} g
- Fat: {total['fat']} g
- Carbohydrates: {total['carbs']} g

⚠️ Nutrient Deficiencies:
{', '.join(deficiency)}

🥗 Personalized Recommendations:
- Adjust diet according to BMI status
- Include missing nutrients in daily meals
- Follow a goal-oriented balanced diet
- Maintain proper meal timing and hydration
"""
# -------------------------
# DEFICIENCY ANALYSIS
# -------------------------
def analyze_deficiency(total):
    deficiencies = []

    if total["protein"] < 50:
        deficiencies.append("Your protein intake is low. Eat eggs, chicken, dal or paneer.")

    if total["carbs"] < 130:
        deficiencies.append("Your carbohydrate intake is low. Eat rice, bread or fruits.")

    if total["fat"] < 20:
        deficiencies.append("Healthy fats are low. Eat nuts or seeds.")

    if total["calories"] < 500:
        deficiencies.append("Your calorie intake is low. Increase balanced meals.")

    if not deficiencies:
        deficiencies.append("Your diet looks balanced. Keep maintaining it.")

    return deficiencies

# -------------------------
# TEXT TO AUDIO
# -------------------------
def text_to_audio(text):
    tts = gTTS(text)
    file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(file.name)
    return file.name

# -------------------------
# MAIN INPUT
# -------------------------
uploaded_files = st.file_uploader("📸 Upload Food Images", accept_multiple_files=True)

if uploaded_files:
    all_foods = []

    col1, col2 = st.columns(2)

    # Images
    with col1:
        st.subheader("📸 Uploaded Images")
        for i, file in enumerate(uploaded_files):
            img = Image.open(file)
            st.image(img, caption=f"Image {i+1}")

            result = classifier(img)
            food = result[0]["label"].replace("_", " ")
            all_foods.append(food)

    # Detected Foods
    with col2:
        st.subheader("🍔 Detected Foods")
        for f in all_foods:
            st.success(f)

    # -------------------------
    # NUTRITION
    # -------------------------
    total, details = calculate_total(all_foods)

    st.subheader("🍽 Nutrition per Food")
    for f, d in details:
        st.write(f"{f} → {d}")

    st.subheader("📊 Total Nutrition")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Calories", total["calories"])
    c2.metric("Protein", total["protein"])
    c3.metric("Fat", total["fat"])
    c4.metric("Carbs", total["carbs"])

    df = pd.DataFrame.from_dict(total, orient='index', columns=['Value'])
    st.bar_chart(df)

    # -------------------------
    # HEALTH ANALYSIS
    # -------------------------
    bmi, status = calculate_bmi(weight, height)

    st.subheader("🤖 Health Analysis")

    st.write(f"BMI: {bmi} ({status})")

    if status == "Underweight":
        st.warning("You are underweight. Consider increasing calorie intake.")
    elif status == "Overweight":
        st.warning("You are overweight. Consider reducing calories.")
    elif status == "Obese":
        st.error("You are obese. Strict diet control needed.")
    else:
        st.success("You have a normal weight.")

    # -------------------------
    # AUDIO (SMART ADVICE)
    # -------------------------
    st.subheader("🔊 Smart Audio Advice")

    deficiency = analyze_deficiency(total)

    audio_text = f"Based on your diet analysis, "

    for d in deficiency:
        audio_text += d + " "

    if goal == "Muscle Gain":
        audio_text += "Focus on high protein foods."
    elif goal == "Weight Loss":
        audio_text += "Reduce calories and eat more fiber."

    audio_file = text_to_audio(audio_text)
    st.audio(audio_file)

    if os.path.exists(audio_file):
        os.remove(audio_file)
