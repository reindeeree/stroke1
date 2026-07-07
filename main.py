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
    except FileNotFoundError:
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
help_texts = [
    "The average pixel intensity level.", "The variation or spread of intensity levels.",
    "The square root of Variance, also measuring the spread.", "The degree of randomness or complexity in the image.",
    "The degree of asymmetry in the intensity distribution.", "The 'peakedness' of the intensity distribution.",
    "A measure of the local variations between pixels.", "A measure of the uniformity or regularity of a pattern.",
    "Another name for Energy, measuring uniformity.", "A measure of the texture's smoothness.",
    "A measure of texture roughness (opposite of Homogeneity).", "A measure of the presence of linear patterns.",
    "A measure of the scale or 'graininess' of the texture."
]

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
        # --- VALIDATION REVISION STARTS HERE ---
        # Step 1: Create a list of inputs that are still 0.0, EXCEPT for 'Coarseness'
        empty_columns = [
            feature_name for feature_name, value in user_inputs.items() 
            if value == 0.0 and feature_name != 'Coarseness'
        ]
        
        # Step 2: Check if any column (other than Coarseness) is empty
        if len(empty_columns) > 0:
            # Step 3: If so, display a warning
            st.error(f"**Warning:** Please fill in all attributes (except Coarseness) for an accurate prediction. The following attributes are still set to 0.0:", icon="❗")
            for column in empty_columns:
                st.markdown(f"- **{column}**")
        else:
            # Step 4: If all other columns are filled, run the prediction
            if model is not None and scaler is not None:
                input_df = pd.DataFrame({
                    "gender":[gender],
                    "age":[age],
                    "hypertension":[hypertension],
                    "heart_disease":[heart_disease],
                    "ever_married":[ever_married],
                    "work_type":[work_type],
                    "Residence_type":[Residence_type],
                    "avg_glucose_level":[avg_glucose_level],
                    "bmi":[bmi],
                    "smoking_status":[smoking_status]

                })

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
                    st.error("### Result: Stroke Risk Indicated", icon="⚠️")
                    confidence = prediction_proba[0][1] * 100
                else:
                    st.success("### Result: No Stroke Risk Indicated", icon="✅")
                    confidence = prediction_proba[0][0] * 100

                st.metric(label="Model Confidence Level", value=f"{confidence:.2f}%")
                st.progress(int(confidence))

                with st.expander("View Processed Data Details"):
                    st.write("Initial Input Data:")
                    st.dataframe(input_df)
                    st.write("Data After Scaling (as seen by the model):")
                    st.dataframe(pd.DataFrame(input_scaled, columns=feature_names))
            else:
                st.error("Model could not be loaded. Please check your .pkl files.")
        # --- END OF VALIDATION REVISION ---
    else:
        st.info("Please enter all attribute values in the left panel and click the 'Get Prediction' button.")

with tab_attributes:
    st.header("Attribute Glossary")
    st.write("Here are simple descriptions for each attribute used by the model:")
    for feature, help_text in zip(feature_names, help_texts):
        st.info(f"**{feature}**: {help_text}")

with tab_project:
    st.header("About")
    st.markdown("""
    This website is an implementation of the research project **[COMPARISON OF LOGISTIC REGRESSION, SUPPORT VECTOR MACHINE, AND K-NEAREST NEIGHBOR USING ENSEMBLE METHOD IN Stroke Risk CLASSIFICATION]**.
    Its purpose is to assist in Stroke Risk classification based on 13 features extracted from medical imagery.

    ### Model Used
    The "brain" of this website is an **Ensemble Stacking** model, an advanced technique that combines the strengths of three base models to enhance prediction accuracy and reliability:
    1.  **K-Nearest Neighbors (KNN)**: With n_neighbors=11 and 'manhattan' metric.
    2.  **Logistic Regression (ALR)**: With C=0.000082.
    3.  **Support Vector Machine (SVM)**: With a 'poly' kernel and degree=3.

    ### Created by:
    **Jessica Joseph Sen**
    
    **Rosalinda Amelia Hizkia Manurung**
                
    **Sania Rahma Dwita Manoppo**
                
    **Nannesa Reinesya** 
                
    **Diah Suci**
                
    **Novita Manullang** """)