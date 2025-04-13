import streamlit as st
import requests
import pandas as pd
import plotly.express as px

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

# Custom CSS for styling
st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)

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
        
        # Work Experience
        st.subheader("Work Experience")
        total_working_years = st.slider("Total Working Years", 0, 40, 5)
        years_at_company = st.slider("Years at Company", 0, 40, 3)
        
        # Compensation
        st.subheader("Compensation")
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000, step=500)
        
        # Work Environment & Satisfaction
        st.subheader("Work Environment & Satisfaction")
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3, 
                                          help="1=Low, 2=Medium, 3=High, 4=Very High")
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3,
                                   help="1=Low, 2=Medium, 3=High, 4=Very High")
        work_life_balance = st.slider("Work Life Balance", 1, 4, 3,
                                    help="1=Low, 2=Medium, 3=High, 4=Very High")
        overtime = st.selectbox("Overtime", ["Yes", "No"])
        
        # Submit button
        predict_button = st.button("Predict Attrition Risk")
    
    # Main content area
    if predict_button:
        # Prepare data for API
        data = {
            "model_type": "attrition",
            "features": {
                # Top 10 most important features
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
                response = requests.post("http://127.0.0.1:8000/predict", json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    col1, col2 = st.columns([3, 2])
                    
                    with col1:
                        # Prediction result
                        if result["prediction"] == 0:
                            st.success("### Low Attrition Risk ✅")
                            st.markdown("This employee is likely to stay with the company.")
                        else:
                            st.error("### High Attrition Risk ⚠️")
                            st.markdown("This employee may be at risk of leaving.")
                        
                        # Enhanced explanation
                        st.info("### Key Factors")
                        
                        # Feature importance visualization - Updated with actual top 10 features
                        st.subheader("Factors Affecting Attrition Risk")
                        feature_importance = {
                            "Monthly Income": 0.20,
                            "Years at Company": 0.15,
                            "Overtime": 0.12,
                            "Job Satisfaction": 0.11,
                            "Distance From Home": 0.10,
                            "Work Life Balance": 0.09,
                            "Age": 0.08,
                            "Job Level": 0.06,
                            "Total Working Years": 0.05,
                            "Environment Satisfaction": 0.04
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
                            if result["prediction"] == 1:
                                st.markdown("### 🚨 Retention Recommendations")
                                
                                recommendations = []
                                
                                if monthly_income < 5000:
                                    recommendations.append("Consider a compensation review")
                                if job_satisfaction < 3:
                                    recommendations.append("Schedule a job satisfaction discussion")
                                if work_life_balance < 3:
                                    recommendations.append("Review workload and scheduling")
                                if overtime == "Yes":
                                    recommendations.append("Evaluate workload distribution and overtime requirements")
                                if environment_satisfaction < 3:
                                    recommendations.append("Assess work environment concerns")
                                
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
                st.error("Make sure the FastAPI backend is running with: python -m uvicorn src.api.main:app --reload") 