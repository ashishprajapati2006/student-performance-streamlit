import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Streamlit page configuration
st.set_page_config(page_title="ML Project - Student Exam Performance", layout="centered")

# --- Main Page Design ---
st.markdown("<h1 style='text-align: center; color: #ff6600;'>🎓 Machine Learning Project</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>This project is created by <b>Ashish Prajapati</b></h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:16px;'>Welcome! Enter the student details below to predict the exam performance.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Input Section ---
st.markdown("<h3 style='text-align:center;'>🧠 Enter Student Information</h3>", unsafe_allow_html=True)

# Center form using columns
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    with st.form("prediction_form"):
        gender = st.selectbox("Gender", ["", "male", "female"])
        race_ethnicity = st.selectbox("Race/Ethnicity", ["", "group A", "group B", "group C", "group D", "group E"])
        parental_level_of_education = st.selectbox(
            "Parental Level of Education",
            ["", "some high school", "high school", "associate's degree", "bachelor's degree", "master's degree"]
        )
        lunch = st.selectbox("Lunch Type", ["", "standard", "free/reduced"])
        test_preparation_course = st.selectbox("Test Preparation Course", ["", "none", "completed"])
        reading_score = st.text_input("Reading Score (0-100)")
        writing_score = st.text_input("Writing Score (0-100)")
        
        submitted = st.form_submit_button("🔮 Predict Performance")

# --- Prediction Logic ---
if submitted:
    try:
        # Convert input values safely
        reading_score_val = float(reading_score)
        writing_score_val = float(writing_score)

        # Create input dataframe
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score_val,
            writing_score=writing_score_val
        )

        pred_df = data.get_data_as_data_frame()

        # Display input data
        st.markdown("### 📋 Input Data")
        st.dataframe(pred_df)

        # Predict using model
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Display result
        st.success(f"🎯 **Predicted Math Score:** {results[0]:.2f}")

    except ValueError:
        st.error("⚠️ Please enter valid numeric values for Reading and Writing Scores.")
    except Exception as e:
        st.error(f"❌ Error during prediction:\n{e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Developed with ❤️ by Ashish Prajapati</p>", unsafe_allow_html=True)
