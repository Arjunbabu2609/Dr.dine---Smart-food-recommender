import streamlit as st
import joblib
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import random
from streamlit_chat import message

# Load ML components
model = joblib.load("C:/Users/tonys/OneDrive/Desktop/python.py/food_recommendation_model.pkl")
vectorizer = joblib.load("C:/Users/tonys/OneDrive/Desktop/python.py/vectorizer.pkl")
label_encoder = joblib.load("C:/Users/tonys/OneDrive/Desktop/python.py/label_encoder.pkl")

# Constants
SUPPORTED_CONDITIONS = [
    "Diabetes", "Hypertension", "Obesity", "High Cholesterol", "Heart Disease",
    "PCOD", "Gout", "Liver Disease Irritable", "Lactose Intolerance",
    "Anxiety", "Cancer", "Asthma", "Allergy"
]

# Helper functions
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return "".join([page.get_text() for page in doc])
    else:
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)

def extract_conditions_from_text(text):
    return list({cond for cond in SUPPORTED_CONDITIONS if cond.lower() in text.lower()})

def extract_food_items_from_image(image):
    text = pytesseract.image_to_string(image)
    return [item.strip() for line in text.splitlines() for item in line.split(',') if item.strip()]

def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)

def get_bmi_category(bmi):
    if bmi < 18.5: return "Underweight"
    elif bmi < 25: return "Normal"
    elif bmi < 30: return "Overweight"
    else: return "Obese"

def is_suitable_for_all_conditions(food_item, conditions):
    for cond in conditions:
        input_text = f"{food_item} for {cond}"
        input_vector = vectorizer.transform([input_text])
        prediction = model.predict(input_vector)[0]
        if label_encoder.inverse_transform([prediction])[0] != "Yes":
            return False
    return True

def get_best_foods(food_list, conditions, top_n=5):
    suitable_foods = [food for food in food_list if is_suitable_for_all_conditions(food, conditions)]
    random.shuffle(suitable_foods)
    return suitable_foods[:top_n]

# Persistent state setup
if 'page' not in st.session_state:
    st.session_state.page = 'Food Input'
if 'food_items' not in st.session_state:
    st.session_state.food_items = ""
if 'user_data' not in st.session_state:
    st.session_state.user_data = [{} for _ in range(3)]
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [{"role": "assistant", "content": "Hi! I'm Dr. Dine 🤖. Ask me anything!"}]
if 'chat_text' not in st.session_state:
    st.session_state.chat_text = ""

# Layout & Navigation
st.set_page_config(page_title="Dr.Dine", page_icon="Group 13.png")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.pinimg.com/736x/b2/24/ba/b224bae0222627f98fb20ae546fe9c85.jpg");
    background-size: cover;
}
.stButton>button {
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🍽 Dr.Dine - Smart Food Recommender")

# Horizontal Navigation Buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    if st.button("📋 Food Input"):
        st.session_state.page = "Food Input"
with col2:
    if st.button("📤 Upload Reports"):
        st.session_state.page = "Upload Reports"
with col3:
    if st.button("💬 Chatbot"):
        st.session_state.page = "Chatbot"

# ---------------------- FOOD INPUT PAGE ----------------------
if st.session_state.page == "Food Input":
    st.header("📋 Food Input")
    upload_option = st.radio("Choose food input method:", ("Manual entry", "Upload menu image"))
    if upload_option == "Manual entry":
        st.session_state.food_items = st.text_area("Enter food items (comma separated):", st.session_state.food_items)
    else:
        menu_image = st.file_uploader("Upload menu image", type=["png", "jpg", "jpeg"], key="menu_upload")
        if menu_image:
            image = Image.open(menu_image)
            st.image(image, caption="Uploaded Menu", use_column_width=True)
            extracted = extract_food_items_from_image(image)
            st.session_state.food_items = ", ".join(extracted)
        st.session_state.food_items = st.text_area("Edit food items:", st.session_state.food_items)

# ---------------------- UPLOAD REPORTS PAGE ----------------------
elif st.session_state.page == "Upload Reports":
    st.header("👥 Upload User Reports & Details")
    cols = st.columns(3)
    for i in range(3):
        with cols[i]:
            st.subheader(f"User {i+1}")
            weight = st.number_input(f"Weight (kg)", 10.0, 300.0, value=st.session_state.user_data[i].get("weight", 70.0), key=f"w{i}")
            height = st.number_input(f"Height (cm)", 50.0, 250.0, value=st.session_state.user_data[i].get("height", 170.0), key=f"h{i}")
            bmi = calculate_bmi(weight, height)
            bmi_cat = get_bmi_category(bmi)
            st.markdown(f"*BMI:* {bmi:.2f} ({bmi_cat})")

            file = st.file_uploader(f"Upload Report", type=["pdf", "png", "jpg", "jpeg"], key=f"rep_{i}")
            conditions = []
            if file:
                try:
                    text = extract_text_from_file(file)
                    conditions = extract_conditions_from_text(text)
                except:
                    st.error("Error reading file")
            if conditions:
                st.success(f"Conditions: {', '.join(conditions)}")
            else:
                st.info("No known conditions found." if file else "Upload a file to detect conditions.")

            st.session_state.user_data[i] = {
                "weight": weight,
                "height": height,
                "bmi": bmi,
                "bmi_cat": bmi_cat,
                "conditions": conditions
            }

    if st.button("🍛 Get Recommendations"):
        if not st.session_state.food_items.strip():
            st.warning("Please provide food items.")
        else:
            foods = [f.strip() for f in st.session_state.food_items.split(",") if f.strip()]
            st.subheader("✅ Recommendations")
            cols = st.columns(3)
            for i in range(3):
                with cols[i]:
                    st.markdown(f"**User {i+1}**")
                    conds = st.session_state.user_data[i].get("conditions", [])
                    if not conds:
                        st.info("No conditions.")
                    else:
                        rec = get_best_foods(foods, conds)
                        if rec:
                            for food in rec:
                                st.success(food)
                        else:
                            st.warning("No suitable foods.")

# ---------------------- CHATBOT PAGE ----------------------
elif st.session_state.page == "Chatbot":
    st.header("💬 Chat with Dr. Dine")
    chat_file = st.file_uploader("📤 Upload a medical report or food menu", type=["pdf", "png", "jpg", "jpeg"], key="chat_report")
    if chat_file:
        try:
            st.session_state.chat_text = extract_text_from_file(chat_file)
            st.success("Text extracted!")
            st.text_area("Extracted Text", st.session_state.chat_text[:1000], height=150)
        except:
            st.error("Unable to read file")

    for msg in st.session_state.chat_messages:
        message(msg["content"], is_user=(msg["role"] == "user"))

    user_input = st.text_input("You:", key="chat_input")

    def get_bot_response(msg, extracted_text=""):
        msg = msg.lower()
        if "bmi" in msg:
            return "Your BMI indicates your body fat. Refer to your section for BMI insights."
        elif "report" in msg or "condition" in msg:
            return extracted_text[:500] + "..." if extracted_text else "Please upload a report first."
        elif "food" in msg or "recommend" in msg:
            return "Upload your menu and go to the Upload Reports section to get recommendations!"
        elif "hello" in msg or "hi" in msg:
            return "Hey there! 👋 How can I help today?"
        else:
            return "I'm still learning! Try asking about food, reports, or BMI."

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        reply = get_bot_response(user_input, st.session_state.chat_text)
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()
