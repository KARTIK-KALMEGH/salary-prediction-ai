import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.title("💼 AI Salary Prediction System")

# Load dataset
df = pd.read_csv("expected_ctc.csv")

# ---------- DATA CLEANING ----------
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# ---------- ENCODING TEXT TO NUMBER ----------
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# ---------- SELECT FEATURES ----------
X = df.drop("Expected_CTC", axis=1)
y = df["Expected_CTC"]

# ---------- TRAIN MODEL ----------
model = RandomForestRegressor()
model.fit(X, y)

st.write("Enter candidate details")

inputs = []

for col in X.columns:
    val = st.number_input(f"{col}")
    inputs.append(val)

if st.button("Predict Salary"):
    pred = model.predict([inputs])
    st.success(f"Predicted Salary: ₹ {round(pred[0],2)}")