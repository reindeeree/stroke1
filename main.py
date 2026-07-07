import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration MUST be the first Streamlit command
st.set_page_config(page_title="Stroke Risk Prediction", layout="wide", initial_sidebar_state="expanded")

# =============================================================================
# BACKEND FUNCTIONS
# =============================================================================
@st.cache_resource
def load_files():
    try:
        model = joblib.load("stroke_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return model, scaler, feature_columns

    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

model, scaler, feature_columns = load_files()

# =============================================================================
# SIDEBAR (FOR INPUTS)
# =============================================================================
st.sidebar.title("Data Input Panel")
st.sidebar.markdown("Enter the values for each medical attribute below.")

categorical_cols = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

feature_descriptions = {
    "gender": "The gender of the patient (Male or Female).",
    "age": "Age of the patient in years.",
    "hypertension": "Whether the patient has hypertension (1 = Yes, 0 = No).",
    "heart_disease": "Whether the patient has heart disease (1 = Yes, 0 = No).",
    "ever_married": "Whether the patient has ever been married.",
    "work_type": "Patient's occupation.",
    "Residence_type": "Whether the patient lives in an Urban or Rural area.",
    "avg_glucose_level": "Average blood glucose level.",
    "bmi": "Body Mass Index.",
    "smoking_status": "Smoking status."
}

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.sidebar.number_input(
    "Age",
    min_value=0,
    max_value=120,
    value=30
)

hypertension = st.sidebar.selectbox(
    "Hypertension",
    [0,1]
)

heart_disease = st.sidebar.selectbox(
    "Heart Disease",
    [0,1]
)

ever_married = st.sidebar.selectbox(
    "Ever Married",
    ["Yes","No"]
)

work_type = st.sidebar.selectbox(
    "Work Type",
    [
        "Private",
        "Self-employed",
        "Govt_job",
        "children",
        "Never_worked"
    ]
)

Residence_type = st.sidebar.selectbox(
    "Residence Type",
    [
        "Urban",
        "Rural"
    ]
)

avg_glucose_level = st.sidebar.number_input(
    "Average Glucose Level",
    value=100.0
)

bmi = st.sidebar.number_input(
    "BMI",
    value=25.0
)

smoking_status = st.sidebar.selectbox(
    "Smoking Status",
    [
        "formerly smoked",
        "never smoked",
        "smokes",
        "Unknown"
    ]
)


# Prediction button is placed in the sidebar
predict_button = st.sidebar.button('**Get Prediction**', type="primary", use_container_width=True)

# =============================================================================
# MAIN PAGE DISPLAY
# =============================================================================
st.title("Stroke Risk Classification Prediction")
st.warning("**Disclaimer:** Please consult a doctor for an accurate diagnosis. This tool is intended to serve as a second opinion for doctors to improve accuracy and efficiency in detecting Stroke Risks.", icon="⚠️")

# Creating TABS
tab_prediction, tab_attributes, tab_project = st.tabs(["📈 **Prediction Result**", "📚 **Attribute Descriptions**", "ℹ️ **About the Project**"])

with tab_prediction:

    st.header("Prediction Result from the AI Model")

    if predict_button:

        user_inputs = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": Residence_type,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status
        }

        numerical_features = [
            "age",
            "avg_glucose_level",
            "bmi"
        ]

        empty_columns = [
            feature
            for feature, value in user_inputs.items()
            if feature in numerical_features and value == 0
        ]

        if empty_columns:

            st.error("Please fill in all required numerical values.")

            for col in empty_columns:
                st.write(f"• {col}")

        else:

            if model is None or scaler is None or feature_columns is None:
                st.error("Model files could not be loaded.")
            else:

                input_df = pd.DataFrame([user_inputs])

                input_df = pd.get_dummies(
                    input_df,
                    columns=categorical_cols,
                    drop_first=True
                )

                input_df = input_df.reindex(
                    columns=feature_columns,
                    fill_value=0
                )

                input_scaled = scaler.transform(input_df)

                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)

                if prediction[0] == 1:
                    st.error("## Stroke Risk Detected")
                    confidence = prediction_proba[0][1] * 100
                else:
                    st.success("## No Stroke Risk Detected")
                    confidence = prediction_proba[0][0] * 100

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

                st.progress(int(confidence))

                with st.expander("Processed Input"):

                    st.write("Input after preprocessing")

                    st.dataframe(input_df)

                    st.write("Scaled input")

                    st.dataframe(
                        pd.DataFrame(
                            input_scaled,
                            columns=feature_columns
                        )
                    )

    else:

        st.info("Fill in the form on the left and press **Get Prediction**.")

with tab_attributes:
    st.header("Attribute Glossary")
    st.write("Here are simple descriptions for each attribute used by the model:")
    for feature, description in feature_descriptions.items():
        st.info(f"**{feature.replace('_',' ').title()}**: {description}")

with tab_project:
    st.header("About")
    st.markdown("""
    This website is an implementation of the research project **[COMPARISON OF LOGISTIC REGRESSION, SUPPORT VECTOR MACHINE, AND K-NEAREST NEIGHBOR USING ENSEMBLE METHOD IN Stroke Risk CLASSIFICATION]**.
    Its purpose is to assist in Stroke Risk classification based on 13 features extracted from medical imagery.

    ### Model Used
    This application uses an **Ensemble Stacking Classifier** for stroke risk prediction. Stacking is an ensemble learning technique that combines the strengths of multiple machine learning algorithms through a meta-classifier.

    The ensemble consists of three base classifiers:
    1. **K-Nearest Neighbors (KNN)** with **n_neighbors = 7**
    2. **Support Vector Machine (SVM)** using a **Linear kernel**
    3. **Decision Tree (CART)** with **max_depth = 14**

    The predictions generated by these three base classifiers are then combined using **Logistic Regression** as the **meta-classifier (final estimator)** to produce the final stroke risk prediction.

    ### Created by:
    **Jessica Joseph Sen**
    
    **Rosalinda Amelia Hizkia Manurung**
                
    **Sania Rahma Dwita Manoppo**
                
    **Vannesa Reinesya** 
                
    **Diah Suci**
                
    **Novita Manullang** """)