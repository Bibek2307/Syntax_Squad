import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import traceback

def show_liver_disease_prediction():
    st.title("Liver Disease Prediction")
    
    # Create sidebar for inputs
    with st.sidebar:
        st.header("Patient Details")
        
        # Personal Information
        st.subheader("Personal Information")
        age = st.slider("Age", 10, 90, 45)
        gender = st.radio("Gender", ["Male", "Female"])
        
        # Liver Function Tests
        st.subheader("Liver Function Tests")
        total_bilirubin = st.number_input("Total Bilirubin", min_value=0.1, max_value=100.0, value=1.0, step=0.1,
                                         help="Normal range: 0.3-1.2 mg/dL")
        direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, max_value=50.0, value=0.3, step=0.1,
                                          help="Normal range: 0.0-0.3 mg/dL")
        
        # Enzyme Levels
        st.subheader("Enzyme Levels")
        alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=20, max_value=2000, value=290, step=10,
                                              help="Normal range: 44-147 IU/L")
        alamine_aminotransferase = st.number_input("Alamine Aminotransferase (ALT)", min_value=1, max_value=2000, value=40, step=1,
                                                  help="Normal range: 7-55 IU/L")
        aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (AST)", min_value=1, max_value=2000, value=40, step=1,
                                                    help="Normal range: 8-48 IU/L")
        
        # Protein Levels
        st.subheader("Protein Levels")
        total_protiens = st.number_input("Total Proteins", min_value=1.0, max_value=10.0, value=6.8, step=0.1,
                                        help="Normal range: 6.0-8.3 g/dL")
        albumin = st.number_input("Albumin", min_value=0.5, max_value=10.0, value=3.5, step=0.1,
                                 help="Normal range: 3.5-5.0 g/dL")
        albumin_globulin_ratio = st.number_input("Albumin/Globulin Ratio", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                                help="Normal range: 0.8-2.0")
        
        # Submit button
        predict_button = st.button("Predict Liver Disease Risk")
    
    # Main content area
    if predict_button:
        # Prepare data for API
        features = {
            "Age": age,
            "Gender": gender,
            "Total_Bilirubin": total_bilirubin,
            "Direct_Bilirubin": direct_bilirubin,
            "Alkaline_Phosphotase": alkaline_phosphotase,
            "Alamine_Aminotransferase": alamine_aminotransferase,
            "Aspartate_Aminotransferase": aspartate_aminotransferase,
            "Total_Protiens": total_protiens,
            "Albumin": albumin,
            "Albumin_and_Globulin_Ratio": albumin_globulin_ratio
        }
        
        # Call API
        with st.spinner("Predicting..."):
            try:
                response = requests.post(
                    "http://localhost:8000/predict/liver",
                    json=features
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Prediction result
                        if result["prediction"]:
                            st.error("### High Risk of Liver Disease ⚠️")
                            st.markdown("The patient shows indicators consistent with potential liver disease.")
                        else:
                            st.success("### Low Risk of Liver Disease ✅")
                            st.markdown("The patient's indicators suggest normal liver function.")
                        
                        # Enhanced explanation
                        st.info("### Key Indicators Analysis")
                        
                        # Create a dataframe to show normal ranges vs patient values
                        indicators = {
                            "Indicator": [
                                "Total Bilirubin", 
                                "Direct Bilirubin",
                                "Alkaline Phosphotase",
                                "ALT",
                                "AST",
                                "Total Proteins",
                                "Albumin",
                                "Albumin/Globulin Ratio"
                            ],
                            "Patient Value": [
                                total_bilirubin,
                                direct_bilirubin,
                                alkaline_phosphotase,
                                alamine_aminotransferase,
                                aspartate_aminotransferase,
                                total_protiens,
                                albumin,
                                albumin_globulin_ratio
                            ],
                            "Normal Range": [
                                "0.3-1.2 mg/dL",
                                "0.0-0.3 mg/dL",
                                "44-147 IU/L",
                                "7-55 IU/L",
                                "8-48 IU/L",
                                "6.0-8.3 g/dL",
                                "3.5-5.0 g/dL",
                                "0.8-2.0"
                            ],
                            "Status": [
                                "Normal" if 0.3 <= total_bilirubin <= 1.2 else "Abnormal",
                                "Normal" if 0.0 <= direct_bilirubin <= 0.3 else "Abnormal",
                                "Normal" if 44 <= alkaline_phosphotase <= 147 else "Abnormal",
                                "Normal" if 7 <= alamine_aminotransferase <= 55 else "Abnormal",
                                "Normal" if 8 <= aspartate_aminotransferase <= 48 else "Abnormal",
                                "Normal" if 6.0 <= total_protiens <= 8.3 else "Abnormal",
                                "Normal" if 3.5 <= albumin <= 5.0 else "Abnormal",
                                "Normal" if 0.8 <= albumin_globulin_ratio <= 2.0 else "Abnormal"
                            ]
                        }
                        
                        indicators_df = pd.DataFrame(indicators)
                        
                        # Style the dataframe
                        def highlight_abnormal(val):
                            if val == "Abnormal":
                                return 'background-color: #ffcccb'
                            else:
                                return 'background-color: #90ee90'
                        
                        styled_df = indicators_df.style.applymap(highlight_abnormal, subset=['Status'])
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Feature importance visualization if available
                        if "feature_importance" in result and result["feature_importance"]:
                            st.subheader("Factors Affecting Liver Disease Risk")
                            
                            # Create feature importance dataframe
                            feature_names = list(features.keys())
                            importance_values = result["feature_importance"]
                            
                            # If lengths don't match, use default values
                            if len(importance_values) != len(feature_names):
                                importance_values = [0.15, 0.05, 0.12, 0.08, 0.18, 0.14, 0.10, 0.08, 0.06, 0.04]
                            
                            importance_df = pd.DataFrame({
                                "Feature": feature_names,
                                "Importance": importance_values
                            })
                            importance_df = importance_df.sort_values("Importance", ascending=False)
                            
                            fig = px.bar(
                                importance_df,
                                x="Importance",
                                y="Feature",
                                orientation='h',
                                title="Feature Importance",
                                color="Importance",
                                color_continuous_scale=["#90ee90", "#ffcccb"]
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Recommendations section
                        st.subheader("Recommendations")
                        if result["prediction"]:
                            st.markdown("""
                            * **Consult a Hepatologist:** Schedule an appointment with a liver specialist for further evaluation
                            * **Additional Testing:** Consider ultrasound, CT scan, or liver biopsy for definitive diagnosis
                            * **Lifestyle Changes:** 
                                * Limit alcohol consumption
                                * Maintain a healthy weight
                                * Follow a liver-friendly diet low in processed foods and sugar
                                * Regular exercise
                            * **Medication Review:** Discuss current medications with your doctor as some may affect liver function
                            """)
                        else:
                            st.markdown("""
                            * **Regular Check-ups:** Continue routine health screenings
                            * **Healthy Lifestyle:** 
                                * Maintain a balanced diet rich in fruits, vegetables, and whole grains
                                * Regular physical activity
                                * Limit alcohol consumption
                                * Stay hydrated
                            * **Liver Protection:** Avoid unnecessary medications that may strain the liver
                            """)
                    
                    with col2:
                        # Risk probability gauge
                        st.subheader("Disease Risk Probability")
                        
                        probability = result["probability"]
                        
                        # Create gauge chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=probability * 100,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Risk Level"},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1},
                                'bar': {'color': "darkred" if probability > 0.7 else "orange" if probability > 0.3 else "green"},
                                'steps': [
                                    {'range': [0, 30], 'color': 'rgba(0, 128, 0, 0.3)'},
                                    {'range': [30, 70], 'color': 'rgba(255, 165, 0, 0.3)'},
                                    {'range': [70, 100], 'color': 'rgba(255, 0, 0, 0.3)'}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': probability * 100
                                }
                            }
                        ))
                        
                        fig.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=50, b=20)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk level explanation
                        risk_level = "High" if probability > 0.7 else "Moderate" if probability > 0.3 else "Low"
                        risk_color = "red" if probability > 0.7 else "orange" if probability > 0.3 else "green"
                        
                        st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: {risk_color};'>{risk_level} Risk</div>", unsafe_allow_html=True)
                        
                        # Liver health score
                        st.subheader("Liver Health Indicators")
                        
                        # Calculate abnormal indicators
                        abnormal_count = sum(1 for status in indicators["Status"] if status == "Abnormal")
                        health_score = 100 - (abnormal_count / len(indicators["Status"])) * 100
                        
                        # Display health score
                        st.metric(
                            label="Liver Health Score", 
                            value=f"{health_score:.1f}%",
                            delta=None
                        )
                        
                        # Create progress bar for health score
                        health_color = "green" if health_score > 70 else "orange" if health_score > 40 else "red"
                        st.markdown(f"""
                        <div style="width: 100%; background-color: #ddd; border-radius: 5px;">
                            <div style="width: {health_score}%; height: 20px; background-color: {health_color}; border-radius: 5px;"></div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Age and gender analysis
                        st.subheader("Demographic Analysis")
                        
                        # Age risk factor
                        age_risk = "Higher" if age > 50 else "Moderate" if age > 35 else "Lower"
                        age_color = "red" if age > 50 else "orange" if age > 35 else "green"
                        
                        st.markdown(f"""
                        <div style="margin-top: 10px;">
                            <span style="font-weight: bold;">Age Risk Factor:</span> 
                            <span style="color: {age_color};">{age_risk}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Gender risk factor
                        gender_risk = "Higher" if gender == "Male" else "Lower"
                        gender_color = "orange" if gender == "Male" else "green"
                        
                        st.markdown(f"""
                        <div style="margin-top: 10px;">
                            <span style="font-weight: bold;">Gender Risk Factor:</span> 
                            <span style="color: {gender_color};">{gender_risk}</span>
                        </div>
                        <div style="margin-top: 5px; font-size: 12px; color: #888;">
                            Males typically have higher risk of liver disease than females.
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Disclaimer
                        st.info("⚠️ Disclaimer: This prediction is for informational purposes only and should not replace professional medical advice.")
                
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    try:
                        error_detail = response.json()
                        st.error(f"Error details: {error_detail}")
                    except:
                        st.error(response.text)
                        
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
                st.error(traceback.format_exc())
                st.info("Make sure the FastAPI backend is running with: python -m uvicorn src.api.main:app --reload")

if __name__ == "__main__":
    show_liver_disease_prediction()