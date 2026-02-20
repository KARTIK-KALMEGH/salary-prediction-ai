import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AI Salary Predictor", page_icon="💼", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #0e1117;}
    h1 {color: #4CAF50;}
    </style>
""", unsafe_allow_html=True)

st.title("💼 AI Salary Prediction System")
st.write("An intelligent HR salary recommendation tool")

# Load dataset
df = pd.read_csv("expected_ctc.csv")

# Handle missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

X = df.drop("Expected_CTC", axis=1)
y = df["Expected_CTC"]

model = RandomForestRegressor(n_estimators=200)
model.fit(X, y) 

# ---- ADD THIS BELOW ----
from sklearn.metrics import r2_score

y_pred = model.predict(X)
score = r2_score(y, y_pred)

st.write(f"📈 Model Accuracy (R² Score): {round(score,2)}")

st.subheader("📋 Enter Candidate Details")

col1, col2 = st.columns(2)

inputs = []

for i, col in enumerate(X.columns):
    if i % 2 == 0:
        val = col1.number_input(col)
    else:
        val = col2.number_input(col)
    inputs.append(val)

if st.button("🚀 Predict Salary"):
    prediction = model.predict([inputs])
    st.success(f"💰 Recommended Salary: ₹ {round(prediction[0],2)}")

