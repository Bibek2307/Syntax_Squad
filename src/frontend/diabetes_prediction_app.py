import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from PIL import Image

# Load the trained model and related objects
@st.cache_resource
def load_model_and_objects():
    try:
        with open('diabetes_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('diabetes_feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('diabetes_model_metrics.pkl', 'rb') as f:
            model_metrics = pickle.load(f)
        return model, feature_names, model_metrics
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None

model, feature_names, model_metrics = load_model_and_objects()

# Display header
st.title("🩺 Diabetes Prediction System")
st.markdown("### Enter the patient's information to predict diabetes risk")

# Display model metrics in sidebar
if model_metrics:
    st.sidebar.title("Model Information")
    st.sidebar.markdown(f"Model Type: {model_metrics.get('model_name', 'Unknown')}")
    st.sidebar.markdown(f"Accuracy: {model_metrics.get('testing_accuracy', 0):.2%}")
    st.sidebar.markdown(f"AUC: {model_metrics.get('roc_auc', 0):.2%}")
    
    # Show top features as text-based visualization
    st.sidebar.title("Key Factors in Diabetes Risk")
    try:
        top_features = pd.DataFrame(model_metrics['feature_importance'])
        top_features = top_features.sort_values('importance', ascending=False).head(5)
        
        # Removed the "Top 5 Important Features" heading
        
        # Get max importance for scaling
        max_importance = top_features['importance'].max()
        
        # Feature descriptions
        feature_descriptions = {
            'Glucose': "Blood glucose level",
            'BMI': "Body Mass Index",
            'Age': "Patient's age",
            'Insulin': "Insulin level in blood",
            'DiabetesPedigreeFunction': "Diabetes hereditary factor",
            'BloodPressure': "Blood pressure measurement",
            'Pregnancies': "Number of pregnancies",
            'SkinThickness': "Skin fold thickness"
        }
        
        # Add custom CSS for better styling with light blue bars
        st.markdown("""
        <style>
        .feature-importance {
            margin-bottom: 12px;
            background-color: rgba(49, 51, 63, 0.7);
            border-radius: 5px;
            padding: 10px;
        }
        .feature-name {
            font-weight: bold;
            margin-bottom: 2px;
            display: flex;
            justify-content: space-between;
        }
        .feature-description {
            font-size: 0.85em;
            color: #9e9e9e;
            margin-bottom: 5px;
        }
        .importance-bar {
            height: 8px;
            background-color: #4682B4;  /* Changed to Steel Blue - a light blue shade */
            border-radius: 4px;
            margin-top: 5px;
        }
        .importance-value {
            color: #9e9e9e;
            font-size: 0.9em;
            font-weight: normal;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create text-based visualization for each feature
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            percentage = int((importance / max_importance) * 100)
            
            # Get feature description or use a default
            description = feature_descriptions.get(feature, "")
            
            # Format feature name for display
            display_name = feature.replace('_', ' ').title()
            
            # Special case for DiabetesPedigreeFunction
            if feature == 'DiabetesPedigreeFunction':
                display_name = "Diabetes Pedigree Function"
            
            st.sidebar.markdown(f"""
            <div class="feature-importance">
                <div class="feature-name">{display_name} <span class="importance-value">{importance:.3f}</span></div>
                <div class="feature-description">{description}</div>
                <div class="importance-bar" style="width: {percentage}%;"></div>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.sidebar.write("Could not display feature importance information.")

# Create input form
with st.form("diabetes_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, 
                                           help="A function that scores likelihood of diabetes based on family history")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
    
    submit_button = st.form_submit_button("Predict Diabetes Risk")

if submit_button and model is not None:
    # Create input dataframe with the exact same features used during training
    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    
    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Ensure all required features are present
    if feature_names:
        # Add any missing features with default values
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder columns to match training data
        input_df = input_df[feature_names]
    
    # Make prediction
    try:
        # Use the model's predict method directly with the DataFrame
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Display result
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.markdown(f"""
            <div class="error-box">
                <h3>⚠️ High Risk of Diabetes (Confidence: {prediction_proba[1]:.2%})</h3>
                <p>Based on the patient's profile, our model predicts a high risk of diabetes.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                <h3>✅ Low Risk of Diabetes (Confidence: {prediction_proba[0]:.2%})</h3>
                <p>Based on the patient's profile, our model predicts a low risk of diabetes.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display risk factors
        st.subheader("Risk Factor Analysis")
        
        # Create a radar chart for risk factors
        categories = ['Glucose', 'BMI', 'Age', 'Blood Pressure', 'Insulin', 'Family History']
        
        # Define reference ranges for each category
        reference_ranges = {
            'Glucose': {'low': 70, 'normal': 99, 'high': 126, 'max': 300},
            'BMI': {'low': 18.5, 'normal': 24.9, 'high': 30, 'max': 50},
            'Age': {'low': 0, 'normal': 40, 'high': 60, 'max': 100},
            'Blood Pressure': {'low': 60, 'normal': 80, 'high': 90, 'max': 200},
            'Insulin': {'low': 0, 'normal': 100, 'high': 200, 'max': 1000},
            'Family History': {'low': 0, 'normal': 0.5, 'high': 1, 'max': 3}
        }
        
        # Normalize values to 0-1 scale for radar chart
        normalized_values = [
            min(1, max(0, (glucose - reference_ranges['Glucose']['low']) / 
                      (reference_ranges['Glucose']['max'] - reference_ranges['Glucose']['low']))),
            min(1, max(0, (bmi - reference_ranges['BMI']['low']) / 
                      (reference_ranges['BMI']['max'] - reference_ranges['BMI']['low']))),
            min(1, max(0, (age - reference_ranges['Age']['low']) / 
                      (reference_ranges['Age']['max'] - reference_ranges['Age']['low']))),
            min(1, max(0, (blood_pressure - reference_ranges['Blood Pressure']['low']) / 
                      (reference_ranges['Blood Pressure']['max'] - reference_ranges['Blood Pressure']['low']))),
            min(1, max(0, (insulin - reference_ranges['Insulin']['low']) / 
                      (reference_ranges['Insulin']['max'] - reference_ranges['Insulin']['low']))),
            min(1, max(0, (diabetes_pedigree - reference_ranges['Family History']['low']) / 
                      (reference_ranges['Family History']['max'] - reference_ranges['Family History']['low'])))
        ]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=categories,
            fill='toself',
            name='Patient Values',
            line_color='indianred'
        ))
        
        # Add normal range reference
        normal_values = [
            (reference_ranges['Glucose']['normal'] - reference_ranges['Glucose']['low']) / 
            (reference_ranges['Glucose']['max'] - reference_ranges['Glucose']['low']),
            (reference_ranges['BMI']['normal'] - reference_ranges['BMI']['low']) / 
            (reference_ranges['BMI']['max'] - reference_ranges['BMI']['low']),
            (reference_ranges['Age']['normal'] - reference_ranges['Age']['low']) / 
            (reference_ranges['Age']['max'] - reference_ranges['Age']['low']),
            (reference_ranges['Blood Pressure']['normal'] - reference_ranges['Blood Pressure']['low']) / 
            (reference_ranges['Blood Pressure']['max'] - reference_ranges['Blood Pressure']['low']),
            (reference_ranges['Insulin']['normal'] - reference_ranges['Insulin']['low']) / 
            (reference_ranges['Insulin']['max'] - reference_ranges['Insulin']['low']),
            (reference_ranges['Family History']['normal'] - reference_ranges['Family History']['low']) / 
            (reference_ranges['Family History']['max'] - reference_ranges['Family History']['low'])
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=normal_values,
            theta=categories,
            fill='toself',
            name='Normal Range',
            line_color='lightseagreen',
            opacity=0.3
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Risk Factor Analysis"
        )
        
        st.plotly_chart(fig)
        
        # Display detailed metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Key Metrics Analysis")
            
            # Glucose analysis
            if glucose < 70:
                glucose_status = "Low (Hypoglycemia)"
                glucose_color = "blue"
            elif glucose < 100:
                glucose_status = "Normal"
                glucose_color = "green"
            elif glucose < 126:
                glucose_status = "Prediabetes"
                glucose_color = "orange"
            else:
                glucose_status = "Diabetes range"
                glucose_color = "red"
                
            st.markdown(f"Glucose: {glucose} mg/dL - <span style='color:{glucose_color}'>{glucose_status}</span>", unsafe_allow_html=True)
            
            # BMI analysis
            if bmi < 18.5:
                bmi_status = "Underweight"
                bmi_color = "blue"
            elif bmi < 25:
                bmi_status = "Normal weight"
                bmi_color = "green"
            elif bmi < 30:
                bmi_status = "Overweight"
                bmi_color = "orange"
            else:
                bmi_status = "Obese"
                bmi_color = "red"
                
            st.markdown(f"BMI: {bmi:.1f} - <span style='color:{bmi_color}'>{bmi_status}</span>", unsafe_allow_html=True)
            
            # Blood pressure analysis
            if blood_pressure < 60:
                bp_status = "Low (Hypotension)"
                bp_color = "blue"
            elif blood_pressure < 80:
                bp_status = "Normal"
                bp_color = "green"
            elif blood_pressure < 90:
                bp_status = "Elevated"
                bp_color = "orange"
            else:
                bp_status = "High (Hypertension)"
                bp_color = "red"
                
            st.markdown(f"Blood Pressure: {blood_pressure} mm Hg - <span style='color:{bp_color}'>{bp_status}</span>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Recommendations")
            
            if prediction == 1:
                st.markdown("""
                - Consult with a healthcare provider for a comprehensive diabetes assessment
                - Consider glucose tolerance testing
                - Monitor blood glucose levels regularly
                - Maintain a healthy diet and regular exercise routine
                - Limit sugar and refined carbohydrate intake
                """)
            else:
                st.markdown("""
                - Continue maintaining a healthy lifestyle
                - Regular check-ups to monitor glucose levels
                - Stay physically active
                - Maintain a balanced diet
                - Consider annual diabetes screening, especially if risk factors are present
                """)
    
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.write("Please try again with different values or check if the model is loaded correctly.")
else:
    if not model:
        st.warning("⚠️ Model not loaded correctly. Please check if all model files are present.")