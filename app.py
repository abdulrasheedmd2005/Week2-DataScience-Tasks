import streamlit as st
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Student_Performance_Dataset.csv")

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("Student Performance Dashboard")

# =========================
# Metrics
# =========================

st.subheader("Dataset Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(df))

col2.metric("Average Percentage",
            round(df['Final_Percentage'].mean(), 2))

col3.metric("Highest Percentage",
            round(df['Final_Percentage'].max(), 2))

# =========================
# Dataset Preview
# =========================

st.subheader("Dataset Preview")

st.dataframe(df.head())

# =========================
# Visualization 1
# =========================

st.subheader("Study Hours vs Final Percentage")

fig1, ax1 = plt.subplots()

sns.scatterplot(x='Study_Hours_Per_Day',
                y='Final_Percentage',
                data=df,
                ax=ax1)

st.pyplot(fig1)

# =========================
# Visualization 2
# =========================

st.subheader("Attendance Distribution")

fig2, ax2 = plt.subplots()

sns.histplot(df['Attendance_Percentage'],
             bins=20,
             ax=ax2)

st.pyplot(fig2)

# =========================
# Visualization 3
# =========================

st.subheader("Average Subject Scores")

subjects = ['Math_Score',
            'Science_Score',
            'English_Score']

fig3, ax3 = plt.subplots()

df[subjects].mean().plot(kind='bar',
                         ax=ax3)

st.pyplot(fig3)

# =========================
# Prediction Section
# =========================

st.subheader("Predict Final Percentage")

study_hours = st.number_input("Study Hours")

attendance = st.number_input("Attendance")

math_score = st.number_input("Math Score")

science_score = st.number_input("Science Score")

english_score = st.number_input("English Score")

previous_score = st.number_input("Previous Score")

age = st.number_input("Age")

student_class = st.number_input("Class")

if st.button("Predict Final Percentage"):

    features = np.array([[age,
                          student_class,
                          study_hours,
                          attendance,
                          math_score,
                          science_score,
                          english_score,
                          previous_score]])

    prediction = model.predict(features)

    st.success(
        f"Predicted Final Percentage: {prediction[0]:.2f}"
    )