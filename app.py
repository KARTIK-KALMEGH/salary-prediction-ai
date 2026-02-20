import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.title("💼 Employee Salary Prediction System")

# Load dataset
df = pd.read_csv("expected_ctc.csv")

# Data Cleaning
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Encoding categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# Feature Selection
X = df.drop("Expected_CTC", axis=1)
y = df["Expected_CTC"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("📊 Model Performance")
st.write(f"R² Score: {round(r2,2)}")
st.write(f"Mean Absolute Error: {round(mae,2)}")

# Prediction Section
st.subheader("📋 Enter Candidate Details")

inputs = []
for col in X.columns:
    val = st.number_input(f"{col}")
    inputs.append(val)

if st.button("Predict Salary"):
    prediction = model.predict([inputs])
    st.success(f"Predicted Salary: ₹ {round(prediction[0],2)}")
