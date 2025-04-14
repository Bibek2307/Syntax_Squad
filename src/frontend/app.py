import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px
import traceback
# Fix imports to use relative paths or sys.path modification
import sys
import os
import numpy as np
import pickle
import plotly.graph_objects as go
from datetime import datetime
import joblib

st.set_page_config(
    page_title="AI Prediction Dashboard",
    page_icon="🎯",
    layout="wide"
)
# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the prediction interfaces
from loan_prediction import show_loan_prediction
from src.frontend.attrition_prediction import show_attrition_prediction

# Now import the modules
try:
    from src.utils.chatbot import ExplainableChatbot
    from src.explainers.bias_detector import BiasDetector
    from src.utils.report_generator import ReportGenerator
except ImportError as e:
    # Create dummy classes for development/testing
    class ExplainableChatbot:
        def __init__(self):
            pass
        
        def get_response(self, question, context=None):
            return f"This is a placeholder response for: {question}"
    
    class BiasDetector:
        def __init__(self, model=None, sensitive_features=None):
            self.sensitive_features = sensitive_features or []
    
    class ReportGenerator:
        def generate_loan_report(self, output_path, data, **kwargs):
            # Create a simple text file as placeholder
            with open(output_path, 'w') as f:
                f.write("Sample Report\n\n")
                f.write(str(data))
            return True
import tempfile

# Custom CSS to fix text alignment in status boxes and add progress bars
st.markdown("""
<style>
    .project-title {
        text-align: center;
        background: linear-gradient(120deg, #4c78a8, #2c3153);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 4rem;
        font-weight: 800;
        margin: -1rem auto 1rem auto;
        padding: 1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        font-family: 'Helvetica Neue', sans-serif;
        display: block;
    }
    .status-box {
        display: inline-block;
        padding: 6px 0px;
        border-radius: 4px;
        text-align: center;
        font-weight: bold;
        width: 100%;
        white-space: nowrap;
        font-size: 14px;
    }
    .status-good {
        background-color: #0e5814;
        color: white;
    }
    .status-ok {
        background-color: #9c6a00;
        color: white;
    }
    .status-bad {
        background-color: #8b0000;
        color: white;
    }
    .metric-label {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
    }
    .metric-desc {
        font-size: 14px;
        color: #ccc;
        margin-top: 5px;
    }
    .progress-container {
        width: 100%;
        background-color: #333;
        border-radius: 5px;
        margin-top: 10px;
    }
    .progress-bar {
        height: 10px;
        border-radius: 5px;
    }
    .progress-good {
        background-color: #0e5814;
    }
    .progress-ok {
        background-color: #9c6a00;
    }
    .progress-bad {
        background-color: #8b0000;
    }
    .model-card {
        padding: 20px;
        border-radius: 10px;
        background-color: #1e2130;
        border: 1px solid #2c3153;
        margin: 10px 0;
        transition: transform 0.3s;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .model-card:hover {
        transform: translateY(-5px);
        border-color: #4c78a8;
        cursor: pointer;
    }
    .model-title {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
        color: #ffffff;
    }
    .model-description {
        font-size: 16px;
        color: #ccc;
        margin: 15px 0;
        text-align: center;
    }
    .centered-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        margin-top: 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .low-risk {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# Function to create status box
def status_box(status, text):
    if status == "good":
        return f'<div class="status-box status-good">{text}</div>'
    elif status == "ok":
        return f'<div class="status-box status-ok">{text}</div>'
    else:
        return f'<div class="status-box status-bad">{text}</div>'

# Function to create a metric with progress bar
def metric_with_progress(label, value, description, status, progress_percent):
    progress_class = "progress-good" if status == "good" else "progress-ok" if status == "ok" else "progress-bad"
    return f"""
    <div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div>
    <div class="metric-desc">{description}</div>
    <div class="progress-container">
        <div class="progress-bar {progress_class}" style="width: {progress_percent}%"></div>
    </div>
    """

# API endpoint
API_URL = "http://127.0.0.1:8000"

def show_dashboard():
    st.markdown('<div class="project-title">Intuitron</div>', unsafe_allow_html=True)
    st.markdown("## AI Prediction Dashboard")
    st.markdown("### Select a Prediction Model")
    
    # Create three columns for the model cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="model-card">
            <div class="model-title">🏦 Loan Approval</div>
            <div class="model-description">
                Predict loan approval probability based on financial and personal information.
                Features include credit score, income, assets, and more.
            </div>
            <div class="centered-content">
                <img src="https://img.icons8.com/fluency/96/000000/bank-building.png" style="margin: 20px 0;">
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Loan Predictor"):
            st.session_state.selected_model = "loan"
            st.experimental_rerun()
    
    with col2:
        st.markdown("""
        <div class="model-card">
            <div class="model-title">👥 Employee Attrition</div>
            <div class="model-description">
                Predict employee attrition risk using HR analytics data.
                Analyze factors like job satisfaction, salary, and work-life balance.
            </div>
            <div class="centered-content">
                <img src="https://img.icons8.com/fluency/96/000000/employee-card.png" style="margin: 20px 0;">
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Attrition Predictor"):
            st.session_state.selected_model = "attrition"
            st.experimental_rerun()
    
    with col3:
        st.markdown("""
        <div class="model-card">
            <div class="model-title">🏥 Healthcare</div>
            <div class="model-description">
                Predict healthcare outcomes and risk factors. Analyze patient data for better healthcare decisions.
                Includes diabetes and liver disease prediction models.
            </div>
            <div class="centered-content">
                <img src="https://img.icons8.com/fluency/96/000000/hospital.png" style="margin: 20px 0;">
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Healthcare Predictor"):
            st.session_state.selected_model = "healthcare"
            st.experimental_rerun()

def get_loan_explanation(data, result):
    """Generate personalized explanation for loan prediction"""
    explanation = []
    
    # Credit Score Analysis
    if data['cibil_score'] >= 750:
        explanation.append("• **Credit Score**: Excellent credit score (750+) significantly improves your loan approval chances.")
    elif data['cibil_score'] >= 650:
        explanation.append("• **Credit Score**: Good credit score (650-749) supports your application.")
    else:
        explanation.append("• **Credit Score**: Your credit score is below 650, which may affect loan approval.")
    
    # Income and Loan Amount Analysis
    monthly_income = data['income_annum'] / 12
    loan_term_months = data['loan_term'] * 12
    # Assuming 8% annual interest rate for EMI calculation
    r = 0.08 / 12
    emi = (data['loan_amount'] * r * (1 + r)**loan_term_months) / ((1 + r)**loan_term_months - 1)
    dti_ratio = (emi / monthly_income) * 100
    
    if dti_ratio <= 40:
        explanation.append(f"• **Affordability**: Your EMI would be ₹{emi:,.2f}, which is {dti_ratio:.1f}% of monthly income - this is within acceptable limits.")
    else:
        explanation.append(f"• **Affordability**: Your EMI would be ₹{emi:,.2f}, which is {dti_ratio:.1f}% of monthly income - this may be too high.")
    
    # Assets Analysis
    total_assets = (data['residential_assets_value'] + data['commercial_assets_value'] + 
                   data['luxury_assets_value'] + data['bank_asset_value'])
    asset_ratio = total_assets / data['loan_amount']
    
    if asset_ratio >= 2:
        explanation.append(f"• **Asset Coverage**: Your total assets (₹{total_assets:,.2f}) provide excellent security, covering {asset_ratio:.1f}x the loan amount.")
    elif asset_ratio >= 1:
        explanation.append(f"• **Asset Coverage**: Your assets provide adequate security, covering {asset_ratio:.1f}x the loan amount.")
    else:
        explanation.append(f"• **Asset Coverage**: Your assets cover only {asset_ratio:.1f}x the loan amount, which may be insufficient.")
    
    # Employment and Education
    if data['education'] == "Graduate":
        explanation.append("• **Education**: Being a graduate strengthens your application.")
    else:
        explanation.append("• **Education**: Higher education could improve future loan eligibility.")
        
    if data['self_employed'] == "Yes":
        explanation.append("• **Employment**: Being self-employed may require additional income documentation.")
    else:
        explanation.append("• **Employment**: Salaried employment provides stable income assessment.")
    
    # Overall Assessment
    if result["prediction"]:
        explanation.append("\n### Key Approval Factors:")
        strengths = []
        if data['cibil_score'] >= 650:
            strengths.append("Strong credit history")
        if dti_ratio <= 40:
            strengths.append("Good affordability ratio")
        if asset_ratio >= 1.5:
            strengths.append("Strong asset coverage")
        if data['education'] == "Graduate":
            strengths.append("Higher education")
        explanation.append("• " + ", ".join(strengths))
    else:
        explanation.append("\n### Areas for Improvement:")
        if data['cibil_score'] < 650:
            explanation.append("1. Work on improving credit score")
        if dti_ratio > 40:
            explanation.append("2. Consider a lower loan amount or longer tenure")
        if asset_ratio < 1.5:
            explanation.append("3. Strengthen asset position")
        if data['education'] != "Graduate":
            explanation.append("4. Consider additional qualification")
    
    return "\n".join(explanation)

# Load the trained model and components
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'loan_model.joblib')
        if not os.path.exists(model_path):
            st.error("Model file not found. Please train the model first.")
            return None
            
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def show_loan_prediction():
    st.title("Loan Approval Prediction System")
    
    # Create sidebar for inputs
    with st.sidebar:
        st.header("Loan Application Details")
        
        # Personal information
        st.subheader("Personal Information")
        no_of_dependents = st.slider("Number of Dependents", 0, 10, 2)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        
        # Financial information
        st.subheader("Financial Information")
        income_annum = st.number_input("Annual Income (₹)", min_value=100000, max_value=20000000, value=5000000, step=100000)
        loan_amount = st.number_input("Loan Amount (₹)", min_value=100000, max_value=50000000, value=10000000, step=100000)
        loan_term = st.slider("Loan Term (years)", 2, 30, 15)
        cibil_score = st.slider("CIBIL Score", 300, 900, 650)
        
        # Assets information
        st.subheader("Assets Information")
        residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, max_value=50000000, value=5000000, step=100000)
        commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, max_value=50000000, value=2000000, step=100000)
        luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, max_value=50000000, value=1000000, step=100000)
        bank_asset_value = st.number_input("Bank Assets Value (₹)", min_value=0, max_value=50000000, value=3000000, step=100000)
        
        # Submit button
        predict_button = st.button("Predict Loan Approval")
    
    # Main content area
    if predict_button:
        # Prepare input data
        input_data = {
            "no_of_dependents": no_of_dependents,
            "education": education,
            "self_employed": self_employed,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value
        }
        
        try:
            # Make API request
            response = requests.post(
                "http://localhost:8000/predict/loan_approval",
                json=input_data
            )
            response.raise_for_status()
            result = response.json()
            
            # Display results
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Prediction result
                if result["prediction"] == "Approved":
                    st.success("### Loan Approved! ✅")
                else:
                    st.error("### Loan Rejected ❌")
                
                # Explanation section
                st.subheader("Explanation")
                for explanation in result["explanation"]:
                    st.write(explanation)
                
                # Factors Affecting Application
                st.subheader("Factors Affecting Your Application")
                
                # Create DataFrame for visualization
                feature_importance = result["feature_importance"]
                importance_df = pd.DataFrame({
                    'Feature': list(feature_importance.keys()),
                    'Importance': list(feature_importance.values())
                })
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # Create horizontal bar chart
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale=px.colors.sequential.Blues
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Approval Probability
                st.subheader("Approval Probability")
                st.markdown(f"### {result['probability']:.2%}")
                
                # Financial Metrics
                st.subheader("Financial Metrics")
                metrics = result["financial_metrics"]
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    st.markdown("**CIBIL Score**")
                    st.markdown(f"### {cibil_score}")
                    if cibil_score >= 750:
                        st.markdown("Excellent - Very favorable")
                    elif cibil_score >= 650:
                        st.markdown("Fair - Average credit history")
                    else:
                        st.markdown("Poor - Unfavorable credit history")
                
                with metrics_col2:
                    st.markdown("**Debt-to-Income**")
                    st.markdown(f"### {metrics['debt_to_income']:.2f}")
                    if metrics['debt_to_income'] <= 0.3:
                        st.markdown("Low - Very manageable")
                    elif metrics['debt_to_income'] <= 0.5:
                        st.markdown("Medium - Manageable")
                    else:
                        st.markdown("High - Significant burden")
                
                with metrics_col3:
                    st.markdown("**Assets-to-Loan**")
                    st.markdown(f"### {metrics['asset_to_loan']:.2f}")
                    if metrics['asset_to_loan'] >= 2:
                        st.markdown("Strong - Excellent coverage")
                    elif metrics['asset_to_loan'] >= 1:
                        st.markdown("Fair - Adequate coverage")
                    else:
                        st.markdown("Weak - Limited coverage")
                
                # Monthly Payment Analysis
                st.subheader("Monthly Payment Analysis")
                st.markdown(f"**Monthly Payment:** ₹{metrics['monthly_payment']:,.2f}")
                st.markdown(f"This represents {(metrics['monthly_payment'] / metrics['monthly_income'] * 100):.2f}% of your monthly income (₹{metrics['monthly_income']:,.2f})")
                
                if metrics['debt_to_income'] <= 0.3:
                    st.markdown("Excellent - Very affordable payment")
                elif metrics['debt_to_income'] <= 0.5:
                    st.markdown("Good - Manageable payment")
                else:
                    st.markdown("Caution - High payment burden")
                
                # Assets Information
                st.subheader("Assets Information")
                st.markdown(f"Total Assets: ₹{metrics['total_assets']:,.2f}")
                
                # Create pie chart for asset breakdown
                asset_labels = ['Residential', 'Commercial', 'Luxury', 'Bank']
                asset_values = [residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value]
                
                fig = px.pie(
                    values=asset_values,
                    names=asset_labels,
                    title='Assets Breakdown'
                )
                fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error(traceback.format_exc())
            st.info("Make sure the FastAPI backend is running with: python -m uvicorn src.api.main:app --reload")

def get_attrition_explanation(data, result):
    """Generate personalized explanation for attrition prediction"""
    explanation = []
    
    # Job Level Analysis
    if data['job_level'] <= 2:
        explanation.append("• **Career Growth**: Being in a junior/mid-level position may present opportunities for advancement. Consider discussing career development paths with your manager.")
    else:
        explanation.append("• **Senior Position**: Your senior position indicates strong career progression, which typically correlates with lower attrition.")
    
    # Satisfaction Analysis
    if data['job_satisfaction'] < 3:
        explanation.append("• **Job Satisfaction**: Your job satisfaction score indicates room for improvement. Consider discussing any concerns with your supervisor.")
    else:
        explanation.append("• **Job Satisfaction**: Your high job satisfaction is a positive indicator for retention.")
        
    if data['work_life_balance'] < 3:
        explanation.append("• **Work-Life Balance**: Your work-life balance score suggests potential stress. Consider discussing flexible work arrangements.")
    else:
        explanation.append("• **Work-Life Balance**: You maintain a healthy work-life balance, which is crucial for long-term retention.")
    
    # Compensation Analysis
    expected_income = 3000 * (1 + data['total_working_years'] * 0.1) * data['job_level']
    if data['monthly_income'] < expected_income:
        explanation.append("• **Compensation**: Your current compensation might be below market rate for your experience level. Consider discussing this during your next review.")
    else:
        explanation.append("• **Compensation**: Your compensation is competitive for your experience level.")
    
    # Experience and Tenure
    if data['years_at_company'] < 2:
        explanation.append("• **Tenure**: Being relatively new to the company, it's important to focus on integration and building strong relationships.")
    elif data['years_at_company'] > 5:
        explanation.append("• **Tenure**: Your long tenure indicates strong company loyalty and established relationships.")
    
    # Overall Risk Assessment
    if result["prediction"]:
        explanation.append("\n### Risk Mitigation Strategies:")
        if data['job_satisfaction'] < 3:
            explanation.append("1. Schedule a career development discussion")
        if data['work_life_balance'] < 3:
            explanation.append("2. Review workload and scheduling")
        if data['monthly_income'] < expected_income:
            explanation.append("3. Prepare for compensation discussion")
    else:
        explanation.append("\n### Retention Strengths:")
        strengths = []
        if data['job_satisfaction'] >= 3:
            strengths.append("High job satisfaction")
        if data['work_life_balance'] >= 3:
            strengths.append("Good work-life balance")
        if data['monthly_income'] >= expected_income:
            strengths.append("Competitive compensation")
        explanation.append("• " + ", ".join(strengths))
    
    return "\n".join(explanation)

def show_attrition_prediction():
    st.title("Employee Attrition Prediction")
    
    # Create sidebar for inputs
    with st.sidebar:
        st.header("Employee Details")
        
        # Personal Information
        st.subheader("Personal Information")
        age = st.slider("Age", 18, 65, 30)
        
        # Job Information
        st.subheader("Job Information")
        distance_from_home = st.slider("Distance From Home (km)", 1, 30, 5)
        job_level = st.slider("Job Level", 1, 5, 2)
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000, step=500)
        overtime = st.selectbox("Overtime", ["Yes", "No"])
        
        # Work Experience
        st.subheader("Work Experience")
        total_working_years = st.slider("Total Working Years", 0, 40, 5)
        years_at_company = st.slider("Years at Company", 0, 40, 3)
        
        # Satisfaction Metrics
        st.subheader("Satisfaction Metrics")
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3, 
                                          help="1=Low, 2=Medium, 3=High, 4=Very High")
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3,
                                   help="1=Low, 2=Medium, 3=High, 4=Very High")
        work_life_balance = st.slider("Work Life Balance", 1, 4, 3,
                                    help="1=Low, 2=Medium, 3=High, 4=Very High")
        
        # Submit button
        predict_button = st.button("Predict Attrition Risk")
    
    # Main content area
    if predict_button:
        # Prepare data for API
        data = {
            "model_type": "attrition",
            "features": {
                "Age": age,
                "DistanceFromHome": distance_from_home,
                "EnvironmentSatisfaction": environment_satisfaction,
                "JobLevel": job_level,
                "JobSatisfaction": job_satisfaction,
                "MonthlyIncome": monthly_income,
                "OverTime": overtime,
                "TotalWorkingYears": total_working_years,
                "WorkLifeBalance": work_life_balance,
                "YearsAtCompany": years_at_company
            }
        }
        
        # Call API
        with st.spinner("Predicting..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Prediction result
                        if result["prediction"]:
                            st.error("### High Attrition Risk ⚠️")
                            st.markdown("This employee may be at risk of leaving.")
                        else:
                            st.success("### Low Attrition Risk ✅")
                            st.markdown("This employee is likely to stay with the company.")
                        
                        # Add personalized explanation
                        st.info("### Personalized Analysis")
                        explanation_data = {
                            'job_level': job_level,
                            'monthly_income': monthly_income,
                            'job_satisfaction': job_satisfaction,
                            'work_life_balance': work_life_balance,
                            'years_at_company': years_at_company,
                            'total_working_years': total_working_years
                        }
                        st.markdown(get_attrition_explanation(explanation_data, result))
                        
                        # Feature importance visualization
                        st.subheader("Factors Affecting Attrition Risk")
                        feature_importance = {
                            "Monthly Income": 0.20,
                            "Years at Company": 0.15,
                            "Job Satisfaction": 0.15,
                            "Work Life Balance": 0.12,
                            "Distance From Home": 0.10,
                            "Environment Satisfaction": 0.10,
                            "Job Level": 0.08,
                            "Age": 0.05,
                            "Total Working Years": 0.03,
                            "Overtime": 0.02
                        }
                        
                        feature_df = pd.DataFrame({
                            "feature": list(feature_importance.keys()),
                            "importance": list(feature_importance.values())
                        })
                        
                        fig = px.bar(
                            feature_df,
                            x="importance",
                            y="feature",
                            orientation='h',
                            title="Impact of Different Factors on Attrition Risk",
                            labels={"importance": "Impact", "feature": "Factor"},
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Add container for better styling
                        with st.container():
                            # Probability gauge
                            st.subheader("Attrition Probability")
                            attrition_prob = result["probability"]
                            st.metric(
                                label="Risk Level",
                                value=f"{attrition_prob:.1%}"
                            )
                            
                            st.markdown("---")
                            
                            # Employee Profile Summary
                            st.subheader("Employee Profile")
                            
                            # Job Metrics
                            metrics_col1, metrics_col2 = st.columns(2)
                            
                            with metrics_col1:
                                # Job Level
                                level_status = "good" if job_level >= 3 else "ok"
                                level_desc = "Senior Position" if job_level >= 3 else "Junior/Mid-level Position"
                                level_percent = (job_level / 5) * 100
                                
                                st.markdown(
                                    metric_with_progress("Job Level", str(job_level), level_desc, level_status, level_percent),
                                    unsafe_allow_html=True
                                )
                            
                            with metrics_col2:
                                # Years at Company
                                tenure_status = "good" if years_at_company >= 5 else "ok"
                                tenure_desc = "Experienced Employee" if years_at_company >= 5 else "Growing Experience"
                                tenure_percent = min((years_at_company / 10) * 100, 100)
                                
                                st.markdown(
                                    metric_with_progress("Tenure", f"{years_at_company} years", tenure_desc, tenure_status, tenure_percent),
                                    unsafe_allow_html=True
                                )
                            
                            # Satisfaction Metrics
                            st.markdown("### Satisfaction Levels")
                            
                            sat_col1, sat_col2 = st.columns(2)
                            
                            with sat_col1:
                                # Job Satisfaction
                                job_sat_status = "good" if job_satisfaction >= 3 else "ok" if job_satisfaction >= 2 else "bad"
                                job_sat_desc = "High Satisfaction" if job_satisfaction >= 3 else "Moderate Satisfaction" if job_satisfaction >= 2 else "Low Satisfaction"
                                job_sat_percent = (job_satisfaction / 4) * 100
                                
                                st.markdown(
                                    metric_with_progress("Job Satisfaction", f"{job_satisfaction}/4", job_sat_desc, job_sat_status, job_sat_percent),
                                    unsafe_allow_html=True
                                )
                            
                            with sat_col2:
                                # Work Life Balance
                                wlb_status = "good" if work_life_balance >= 3 else "ok" if work_life_balance >= 2 else "bad"
                                wlb_desc = "Good Balance" if work_life_balance >= 3 else "Moderate Balance" if work_life_balance >= 2 else "Poor Balance"
                                wlb_percent = (work_life_balance / 4) * 100
                                
                                st.markdown(
                                    metric_with_progress("Work-Life Balance", f"{work_life_balance}/4", wlb_desc, wlb_status, wlb_percent),
                                    unsafe_allow_html=True
                                )
                            
                            # Compensation Analysis
                            st.markdown("### Compensation Analysis")
                            
                            # Monthly Income vs Experience
                            income_per_year = monthly_income * 12
                            expected_income = 3000 * (1 + total_working_years * 0.1) * job_level  # Simple model for expected income
                            income_ratio = income_per_year / expected_income
                            
                            income_status = "good" if income_ratio >= 1 else "ok" if income_ratio >= 0.8 else "bad"
                            income_desc = f"{'Above' if income_ratio >= 1 else 'Near' if income_ratio >= 0.8 else 'Below'} expected for experience level"
                            income_percent = min(income_ratio * 100, 100)
                            
                            st.markdown(
                                metric_with_progress(
                                    "Annual Income",
                                    f"₹{income_per_year:,.0f}",
                                    income_desc,
                                    income_status,
                                    income_percent
                                ),
                                unsafe_allow_html=True
                            )
                            
                            # If high attrition risk, show recommendations
                            if result["prediction"]:
                                st.markdown("### 🚨 Retention Recommendations")
                                
                                recommendations = []
                                
                                if monthly_income < 5000:
                                    recommendations.append("Consider a compensation review")
                                if job_satisfaction < 3:
                                    recommendations.append("Schedule a job satisfaction discussion")
                                if work_life_balance < 3:
                                    recommendations.append("Review workload and scheduling")
                                if environment_satisfaction < 3:
                                    recommendations.append("Assess work environment concerns")
                                if overtime == "Yes":
                                    recommendations.append("Review workload distribution and overtime requirements")
                                
                                for rec in recommendations:
                                    st.markdown(f"• {rec}")
                
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    try:
                        error_detail = response.json()
                        st.error(f"Error details: {error_detail}")
                    except:
                        st.code(response.text)
                        
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
                st.error(traceback.format_exc())
                st.info("Make sure the FastAPI backend is running with: python -m uvicorn src.api.main:app --reload")

def show_healthcare_prediction():
    st.title("Healthcare Prediction Models")
    st.markdown("### Select a healthcare prediction model to get started")

    # Create two columns for the prediction options
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="model-card">
            <div class="centered-content">
                <img src="https://img.icons8.com/color/96/000000/dna-helix.png" style="margin: 20px 0;">
            </div>
            <div class="model-title">Diabetes Prediction</div>
            <div class="model-description">
                Assess the risk of diabetes based on health metrics and patient history.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Diabetes Prediction", key="select_diabetes"):
            st.session_state.healthcare_model = "diabetes"
            st.experimental_rerun()

    with col2:
        st.markdown("""
        <div class="model-card">
            <div class="centered-content">
                <img src="https://img.icons8.com/color/96/000000/microscope.png" style="margin: 20px 0;">
            </div>
            <div class="model-title">Liver Disease Prediction</div>
            <div class="model-description">
                Evaluate liver health and disease risk based on patient data and lab results.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Select Liver Disease Prediction", key="select_liver"):
            st.session_state.healthcare_model = "liver"
            st.experimental_rerun()

    # Add a back button
    if st.button("← Back to Dashboard", key="healthcare_back"):
        st.session_state.selected_model = None
        st.session_state.healthcare_model = None
        st.experimental_rerun()

def get_diabetes_explanation(data, result):
    """Generate personalized explanation for diabetes prediction"""
    explanation = []
    
    # Glucose Analysis
    if data['glucose'] < 100:
        explanation.append("• **Glucose Level**: Your glucose level is within normal range (<100 mg/dL).")
    elif data['glucose'] < 126:
        explanation.append("• **Glucose Level**: Your glucose level indicates pre-diabetes (100-125 mg/dL). Consider lifestyle modifications.")
    else:
        explanation.append("• **Glucose Level**: Your glucose level is in the diabetic range (>126 mg/dL). Medical consultation recommended.")
    
    # BMI Analysis
    if data['bmi'] < 18.5:
        explanation.append("• **BMI**: Your BMI indicates underweight status. Consider nutritional consultation.")
    elif data['bmi'] < 25:
        explanation.append("• **BMI**: Your BMI is in the healthy range.")
    elif data['bmi'] < 30:
        explanation.append("• **BMI**: Your BMI indicates overweight status. Consider lifestyle modifications.")
    else:
        explanation.append("• **BMI**: Your BMI indicates obesity. This increases diabetes risk.")
    
    # Blood Pressure Analysis
    if data['blood_pressure'] < 80:
        explanation.append("• **Blood Pressure**: Your blood pressure is normal/low.")
    elif data['blood_pressure'] < 90:
        explanation.append("• **Blood Pressure**: Your blood pressure is elevated.")
    else:
        explanation.append("• **Blood Pressure**: Your blood pressure is high. This can increase diabetes risk.")
    
    # Family History Impact
    if data['diabetes_pedigree'] > 0.5:
        explanation.append("• **Family History**: Your diabetes pedigree function indicates increased hereditary risk.")
    else:
        explanation.append("• **Family History**: Your hereditary risk appears to be lower than average.")
    
    # Risk Level and Recommendations
    if result["prediction"]:
        explanation.append("\n### Key Risk Factors:")
        if data['glucose'] >= 126:
            explanation.append("1. High glucose levels")
        if data['bmi'] >= 30:
            explanation.append("2. Elevated BMI")
        if data['blood_pressure'] >= 90:
            explanation.append("3. High blood pressure")
        
        explanation.append("\n### Recommended Actions:")
        explanation.append("1. Schedule a consultation with a healthcare provider")
        if data['bmi'] >= 25:
            explanation.append("2. Consider a structured weight management program")
        if data['glucose'] >= 100:
            explanation.append("3. Monitor blood glucose regularly")
        explanation.append("4. Maintain regular physical activity")
        explanation.append("5. Follow a balanced, low-sugar diet")
    else:
        explanation.append("\n### Preventive Measures:")
        explanation.append("1. Maintain regular health check-ups")
        explanation.append("2. Continue balanced diet and exercise")
        explanation.append("3. Monitor glucose levels annually")
        if data['bmi'] >= 25:
            explanation.append("4. Consider weight management strategies")
        if data['diabetes_pedigree'] > 0.5:
            explanation.append("5. Regular screening due to family history")
    
    return "\n".join(explanation)

def show_diabetes_prediction():
    st.title("Diabetes Risk Prediction")
    
    # Create sidebar for inputs
    with st.sidebar:
        st.header("Patient Information")
        
        # Personal Information
        st.subheader("Basic Information")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
        
        # Health Metrics
        st.subheader("Health Metrics")
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5,
                                          help="A function that scores likelihood of diabetes based on family history")
        
        # Submit button
        predict_button = st.button("Predict Diabetes Risk", key="predict_diabetes")
    
    # Main content area
    if predict_button:
        # Prepare data for API
        data = {
            "model_type": "diabetes",
            "features": {
                "Age": age,
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": diabetes_pedigree
            }
        }
        
        # Call API
        with st.spinner("Predicting..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Prediction result
                        if result["prediction"]:
                            st.error("### High Risk of Diabetes ⚠️")
                            st.markdown("Based on your health metrics, our model predicts an elevated risk of diabetes.")
                        else:
                            st.success("### Low Risk of Diabetes ✅")
                            st.markdown("Based on your health metrics, our model predicts a lower risk of diabetes.")
                        
                        # Add personalized explanation
                        st.info("### Personalized Analysis")
                        explanation_data = {
                            'glucose': glucose,
                            'bmi': bmi,
                            'blood_pressure': blood_pressure,
                            'diabetes_pedigree': diabetes_pedigree
                        }
                        st.markdown(get_diabetes_explanation(explanation_data, result))
                        
                        # Risk Factor Analysis
                        st.subheader("Risk Factor Analysis")
                        
                        # Create radar chart for risk factors
                        categories = ['Glucose', 'BMI', 'Age', 'Blood Pressure', 'Insulin', 'Family History']
                        values = [glucose/300, bmi/50, age/100, blood_pressure/200, insulin/1000, diabetes_pedigree/3]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name='Patient Values'
                        ))
                        
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig)
                    
                    with col2:
                        # Risk Level
                        st.subheader("Risk Assessment")
                        risk_probability = result["probability"]
                        st.metric(
                            label="Risk Level",
                            value=f"{risk_probability:.1%}"
                        )
                        
                        # Health Metrics Analysis
                        st.markdown("### Health Metrics Analysis")
                        
                        # Glucose Analysis
                        glucose_status = ("Normal" if glucose < 100 else 
                                        "Pre-diabetic" if glucose < 126 else 
                                        "Diabetic Range")
                        glucose_color = ("good" if glucose < 100 else 
                                       "ok" if glucose < 126 else 
                                       "bad")
                        st.markdown(
                            metric_with_progress(
                                "Glucose Level",
                                f"{glucose} mg/dL",
                                glucose_status,
                                glucose_color,
                                min(100, (glucose/126)*100)
                            ),
                            unsafe_allow_html=True
                        )
                        
                        # BMI Analysis
                        bmi_status = ("Normal" if 18.5 <= bmi <= 24.9 else 
                                    "Underweight" if bmi < 18.5 else 
                                    "Overweight")
                        bmi_color = ("good" if 18.5 <= bmi <= 24.9 else "ok")
                        st.markdown(
                            metric_with_progress(
                                "BMI",
                                f"{bmi:.1f}",
                                bmi_status,
                                bmi_color,
                                min(100, (bmi/30)*100)
                            ),
                            unsafe_allow_html=True
                        )
                        
                        # Blood Pressure Analysis
                        bp_status = ("Normal" if blood_pressure < 120 else 
                                   "Elevated" if blood_pressure < 130 else 
                                   "High")
                        bp_color = ("good" if blood_pressure < 120 else 
                                  "ok" if blood_pressure < 130 else 
                                  "bad")
                        st.markdown(
                            metric_with_progress(
                                "Blood Pressure",
                                f"{blood_pressure} mm Hg",
                                bp_status,
                                bp_color,
                                min(100, (blood_pressure/140)*100)
                            ),
                            unsafe_allow_html=True
                        )
                        
                        # Recommendations
                        if result["prediction"]:
                            st.markdown("### 🚨 Recommendations")
                            recommendations = [
                                "Schedule a consultation with a healthcare provider",
                                "Monitor blood glucose regularly",
                                "Maintain a balanced, low-sugar diet",
                                "Exercise regularly (at least 150 minutes per week)",
                                "Consider weight management if BMI is high"
                            ]
                            for rec in recommendations:
                                st.markdown(f"• {rec}")
                
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    try:
                        error_detail = response.json()
                        st.error(f"Error details: {error_detail}")
                    except:
                        st.code(response.text)
                        
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
                st.error(traceback.format_exc())
                st.info("Make sure the FastAPI backend is running with: python -m uvicorn src.api.main:app --reload")

def show_liver_prediction():
    st.title("Liver Disease Risk Prediction")
    
    # Create sidebar for inputs
    with st.sidebar:
        st.header("Patient Information")
        
        # Personal Information
        st.subheader("Basic Information")
        age = st.number_input("Age", min_value=0, max_value=120, value=45, key="liver_age")
        gender = st.selectbox("Gender", ["Male", "Female"], key="liver_gender")
        
        # Lab Results Part 1
        st.subheader("Lab Results (Part 1)")
        total_bilirubin = st.number_input("Total Bilirubin", min_value=0.0, max_value=100.0, value=1.0, step=0.1, key="liver_bilirubin",
                                        help="Normal range: 0.3-1.2 mg/dL")
        direct_bilirubin = st.number_input("Direct Bilirubin", min_value=0.0, max_value=50.0, value=0.3, step=0.1, key="liver_direct_bilirubin",
                                         help="Normal range: 0.0-0.3 mg/dL")
        alkaline_phosphotase = st.number_input("Alkaline Phosphotase", min_value=0, max_value=2000, value=290, key="liver_alk_phos",
                                             help="Normal range: 45-115 U/L")
        
        # Lab Results Part 2
        st.subheader("Lab Results (Part 2)")
        alamine_aminotransferase = st.number_input("Alamine Aminotransferase (ALT)", min_value=0, max_value=2000, value=80, key="liver_alamine",
                                                  help="Normal range: 7-56 U/L")
        aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (AST)", min_value=0, max_value=2000, value=70, key="liver_aspartate",
                                                    help="Normal range: 10-40 U/L")
        
        # Protein Analysis
        st.subheader("Protein Analysis")
        total_proteins = st.number_input("Total Proteins", min_value=0.0, max_value=20.0, value=6.8, step=0.1, key="liver_proteins",
                                       help="Normal range: 6.0-8.3 g/dL")
        albumin = st.number_input("Albumin", min_value=0.0, max_value=10.0, value=3.3, step=0.1, key="liver_albumin",
                                help="Normal range: 3.5-5.5 g/dL")
        albumin_globulin_ratio = st.number_input("Albumin/Globulin Ratio", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="liver_albumin_ratio",
                                                help="Normal range: 1.0-2.5")
        
        # Submit button
        predict_button = st.button("Predict Liver Disease Risk", key="predict_liver")
    
    # Main content area
    if predict_button:
        # Prepare data for API
        data = {
            "model_type": "liver",
            "features": {
                "Age": age,
                "Gender": gender,
                "Total_Bilirubin": total_bilirubin,
                "Direct_Bilirubin": direct_bilirubin,
                "Alkaline_Phosphotase": alkaline_phosphotase,
                "Alamine_Aminotransferase": alamine_aminotransferase,
                "Aspartate_Aminotransferase": aspartate_aminotransferase,
                "Total_Protiens": total_proteins,
                "Albumin": albumin,
                "Albumin_and_Globulin_Ratio": albumin_globulin_ratio
            }
        }
        
        # Call API
        with st.spinner("Analyzing lab results..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Prediction result
                        if result["prediction"]:
                            st.error("### High Risk of Liver Disease ⚠️")
                            st.markdown("Based on the lab results and analysis, there are indicators suggesting potential liver issues that require medical attention.")
                        else:
                            st.success("### Low Risk of Liver Disease ✅")
                            st.markdown("Based on the lab results and analysis, the indicators are within normal ranges.")
                        
                        # Lab Results Analysis
                        st.subheader("Lab Results Analysis")
                        
                        # Create metrics for each test with color-coded status
                        metrics_data = [
                            {
                                "name": "Total Bilirubin",
                                "value": total_bilirubin,
                                "unit": "mg/dL",
                                "normal_range": (0.3, 1.2),
                                "description": "Measures breakdown product of red blood cells"
                            },
                            {
                                "name": "Direct Bilirubin",
                                "value": direct_bilirubin,
                                "unit": "mg/dL",
                                "normal_range": (0.0, 0.3),
                                "description": "Measures conjugated bilirubin"
                            },
                            {
                                "name": "Alkaline Phosphotase",
                                "value": alkaline_phosphotase,
                                "unit": "U/L",
                                "normal_range": (45, 115),
                                "description": "Enzyme found in liver and bone"
                            },
                            {
                                "name": "ALT",
                                "value": alamine_aminotransferase,
                                "unit": "U/L",
                                "normal_range": (7, 56),
                                "description": "Liver-specific enzyme"
                            },
                            {
                                "name": "AST",
                                "value": aspartate_aminotransferase,
                                "unit": "U/L",
                                "normal_range": (10, 40),
                                "description": "Enzyme found in liver and other tissues"
                            }
                        ]
                        
                        # Display metrics in a grid
                        for i in range(0, len(metrics_data), 2):
                            col1_metric, col2_metric = st.columns(2)
                            
                            with col1_metric:
                                metric = metrics_data[i]
                                value = metric["value"]
                                min_val, max_val = metric["normal_range"]
                                status = "good" if min_val <= value <= max_val else "bad"
                                progress = min(100, (value / max_val) * 100)
                                
                                st.markdown(
                                    metric_with_progress(
                                        metric["name"],
                                        f"{value} {metric['unit']}",
                                        f"Normal range: {min_val}-{max_val} {metric['unit']}",
                                        status,
                                        progress
                                    ),
                                    unsafe_allow_html=True
                                )
                            
                            if i + 1 < len(metrics_data):
                                with col2_metric:
                                    metric = metrics_data[i + 1]
                                    value = metric["value"]
                                    min_val, max_val = metric["normal_range"]
                                    status = "good" if min_val <= value <= max_val else "bad"
                                    progress = min(100, (value / max_val) * 100)
                                    
                                    st.markdown(
                                        metric_with_progress(
                                            metric["name"],
                                            f"{value} {metric['unit']}",
                                            f"Normal range: {min_val}-{max_val} {metric['unit']}",
                                            status,
                                            progress
                                        ),
                                        unsafe_allow_html=True
                                    )
                        
                        # Protein Analysis
                        st.subheader("Protein Analysis")
                        protein_metrics = [
                            {
                                "name": "Total Proteins",
                                "value": total_proteins,
                                "unit": "g/dL",
                                "normal_range": (6.0, 8.3),
                                "description": "Total protein in blood"
                            },
                            {
                                "name": "Albumin",
                                "value": albumin,
                                "unit": "g/dL",
                                "normal_range": (3.5, 5.5),
                                "description": "Main protein in blood"
                            },
                            {
                                "name": "A/G Ratio",
                                "value": albumin_globulin_ratio,
                                "unit": "",
                                "normal_range": (1.0, 2.5),
                                "description": "Ratio of albumin to globulin"
                            }
                        ]
                        
                        # Display protein metrics
                        for metric in protein_metrics:
                            value = metric["value"]
                            min_val, max_val = metric["normal_range"]
                            status = "good" if min_val <= value <= max_val else "bad"
                            progress = min(100, (value / max_val) * 100)
                            
                            st.markdown(
                                metric_with_progress(
                                    metric["name"],
                                    f"{value} {metric['unit']}",
                                    f"Normal range: {min_val}-{max_val} {metric['unit']}",
                                    status,
                                    progress
                                ),
                                unsafe_allow_html=True
                            )
                    
                    with col2:
                        # Risk Assessment
                        st.subheader("Risk Assessment")
                        risk_probability = result["probability"]
                        st.metric(
                            label="Risk Level",
                            value=f"{risk_probability:.1%}"
                        )
                        
                        # Create radar chart for key indicators
                        st.subheader("Key Indicators Analysis")
                        
                        # Normalize values for radar chart
                        radar_data = {
                            'Liver Function': min(1, (alamine_aminotransferase / 56 + aspartate_aminotransferase / 40) / 2),
                            'Bilirubin': min(1, total_bilirubin / 1.2),
                            'Protein Status': min(1, (total_proteins / 8.3 + albumin / 5.5) / 2),
                            'Enzyme Levels': min(1, alkaline_phosphotase / 115),
                            'A/G Balance': min(1, albumin_globulin_ratio / 2.5)
                        }
                        
                        fig = go.Figure(data=go.Scatterpolar(
                            fill='toself'
                            ))
                        fig.add_trace(go.Scatterpolar(
                            r=list(radar_data.values()),
                            theta=list(radar_data.keys()),
                            fill='toself',
                            name='Patient Values'
                        ))
                        
                        fig.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            showlegend=False,
                            margin=dict(l=0, r=40, t=20, b=20),  # Adjust margins to shift graph left
                            width=500,
                            height=400
                        )
                        
                        st.plotly_chart(fig)
                        
                        # Recommendations
                        if result["prediction"]:
                            st.markdown("### 🚨 Recommendations")
                            recommendations = []
                            
                            # Add recommendations based on test results
                            if total_bilirubin > 1.2:
                                recommendations.append("• Further liver enzyme tests recommended")
                                if direct_bilirubin > 0.3:
                                    recommendations.append("• Evaluate for liver/bone conditions")
                            
                            if total_proteins < 6.0:
                                recommendations.append("• Improve protein nutrition")
                                recommendations.append("• Consult a nutritionist")
                            
                            if alkaline_phosphotase > 115:
                                recommendations.append("• Consult a hepatologist for detailed evaluation")
                                recommendations.append("• Consider ultrasound or imaging studies")
                            
                            if alamine_aminotransferase > 56 or aspartate_aminotransferase > 40:
                                recommendations.append("• Maintain a liver-healthy diet")
                                recommendations.append("• Avoid alcohol and hepatotoxic substances")
                            
                            # Add follow-up recommendation based on severity
                            abnormal_count = sum([
                                total_bilirubin > 1.2,
                                direct_bilirubin > 0.3,
                                alkaline_phosphotase > 115,
                                alamine_aminotransferase > 56,
                                aspartate_aminotransferase > 40,
                                total_proteins < 6.0,
                                albumin < 3.5
                            ])
                            
                            if abnormal_count >= 3:
                                recommendations.append("• Regular follow-up monitoring every 3 months")
                            elif abnormal_count >= 1:
                                recommendations.append("• Follow-up monitoring every 6 months")
                            else:
                                recommendations.append("• Annual health check-up recommended")
                            
                            for rec in recommendations:
                                st.markdown(rec)
                        else:
                            st.markdown("### ✅ Preventive Measures")
                            preventive_measures = []
                            
                            # Add personalized preventive measures
                            if total_bilirubin > 0.8:
                                preventive_measures.append("• Monitor bilirubin levels during routine check-ups")
                            
                            if total_proteins < 7.0:
                                preventive_measures.append("• Maintain adequate protein intake through balanced diet")
                            
                            if albumin < 4.0:
                                preventive_measures.append("• Focus on protein-rich foods to maintain albumin levels")
                            
                            # Add general preventive measures based on age and gender
                            if age > 40:
                                preventive_measures.append("• Regular liver function screening every 6-12 months")
                            else:
                                preventive_measures.append("• Annual liver function screening")
                            
                            preventive_measures.extend([
                                "• Maintain a balanced, liver-friendly diet",
                                "• Regular exercise to support liver health",
                                "• Limit alcohol consumption",
                                "• Stay well-hydrated"
                            ])
                            
                            for measure in preventive_measures:
                                st.markdown(measure)
                
                else:
                    st.error(f"Error: API returned status code {response.status_code}")
                    try:
                        error_detail = response.json()
                        st.error(f"Error details: {error_detail}")
                    except:
                        st.code(response.text)
                        
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
                st.error(traceback.format_exc())
                st.info("Make sure the FastAPI backend is running with: python -m uvicorn src.api.main:app --reload")

def main():
    # Initialize session state for model selection if not exists
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None
    if 'healthcare_model' not in st.session_state:
        st.session_state.healthcare_model = None
    
    # Show dashboard if no model is selected
    if st.session_state.selected_model is None:
        show_dashboard()
        return
    
    # Add a "Back to Dashboard" button
    if st.sidebar.button("← Back to Dashboard"):
        st.session_state.selected_model = None
        st.session_state.healthcare_model = None
        st.experimental_rerun()
    
    # Show the selected model's interface
    if st.session_state.selected_model == "loan":
        show_loan_prediction()
    elif st.session_state.selected_model == "attrition":
        show_attrition_prediction()
    elif st.session_state.selected_model == "healthcare":
        if st.session_state.healthcare_model == "diabetes":
            show_diabetes_prediction()
        elif st.session_state.healthcare_model == "liver":
            show_liver_prediction()
        else:
            show_healthcare_prediction()

if __name__ == "__main__":
    main()
