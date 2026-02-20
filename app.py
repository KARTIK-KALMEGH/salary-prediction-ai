import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("expected_ctc.csv")

# Features
X = df[['Total_Experience','Current_CTC','Last_Appraisal_Rating']]
y = df['Expected_CTC']

# Train model
model = RandomForestRegressor()
model.fit(X,y)

# UI
st.title("💼 AI Salary Prediction System")
st.write("Enter candidate details")

exp = st.number_input("Total Experience (years)")
current = st.number_input("Current CTC")
rating = st.number_input("Last Appraisal Rating")

if st.button("Predict Salary"):
    pred = model.predict([[exp,current,rating]])
    st.success(f"Predicted Expected Salary: ₹ {round(pred[0],2)}")