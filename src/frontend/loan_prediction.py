import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import traceback

# API endpoint
API_URL = "http://127.0.0.1:8000/predict"

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
    if predict_button or ('prediction_result' in st.session_state and st.session_state.prediction_result):
        # If button was just pressed, API call happens
        if predict_button:
            # Prepare data for API
            data = {
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
            
            # Call API
            with st.spinner("Predicting..."):
                try:
                    # Update the API endpoint to use the specific loan endpoint
                    response = requests.post(
                        "http://127.0.0.1:8000/predict/loan",
                        json={"features": data}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Store result in session state
                        st.session_state.prediction_result = result
                        st.session_state.loan_data = data
                except Exception as e:
                    st.error(f"Error connecting to API: {str(e)}")
                    st.error(traceback.format_exc())
                    st.info("Make sure the FastAPI backend is running with: python -m uvicorn src.api.main:app --reload")
                    return
        
        # Use stored result if available
        if 'prediction_result' in st.session_state:
            result = st.session_state.prediction_result
            data = st.session_state.loan_data
            
            # Display prediction result
            if result["prediction"]:
                st.success("### Loan Approved! ✅")
            else:
                st.error("### Loan Rejected ❌")
            
            # Display approval probability
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Explanation")
                st.write("Based on your financial profile and application details, here's why your loan was approved/rejected.")
                
                # Display factors affecting application
                st.subheader("Factors Affecting Your Application")
                
                # Create feature importance dataframe
                feature_names = [
                    "CIBIL Score", 
                    "Annual Income", 
                    "Loan Amount", 
                    "Loan Term", 
                    "Residential Assets",
                    "Commercial Assets",
                    "Luxury Assets",
                    "Bank Assets",
                    "Dependents",
                    "Education",
                    "Self Employed"
                ]
                
                # Make sure we have the right number of feature names
                importance_values = result.get("feature_importance", [])
                
                # If no feature importance is returned, create dummy values
                if not importance_values:
                    importance_values = [0.2, 0.15, 0.15, 0.1, 0.1, 0.08, 0.07, 0.05, 0.05, 0.03, 0.02]
                
                # Adjust lengths if needed
                if len(importance_values) != len(feature_names):
                    if len(importance_values) > len(feature_names):
                        feature_names = feature_names + [f"Feature {i+1}" for i in range(len(feature_names), len(importance_values))]
                    else:
                        importance_values = importance_values + [0] * (len(feature_names) - len(importance_values))
                
                # Create and sort the dataframe
                importance_df = pd.DataFrame({
                    "Feature": feature_names[:len(importance_values)],
                    "Importance": importance_values
                })
                importance_df = importance_df.sort_values("Importance", ascending=False)
                
                # Create the bar chart
                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation='h',
                    color="Importance",
                    color_continuous_scale=px.colors.sequential.Blues
                )
                fig.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display approval probability
                st.subheader("Approval Probability")
                st.markdown("### Confidence")
                probability = result["probability"] * 100
                st.markdown(f"### {probability:.2f}%")
                
                # Display financial metrics
                st.subheader("Application Summary")
                st.subheader("Detailed Financial Metrics")
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                
                with metrics_col1:
                    st.markdown("**CIBIL Score**")
                    st.markdown(f"### {data['cibil_score']}")
                    if data['cibil_score'] >= 700:
                        st.markdown("Good - Favorable credit history")
                    elif data['cibil_score'] >= 600:
                        st.markdown("Fair - Average credit history")
                    else:
                        st.markdown("Poor - Unfavorable credit history")
                
                with metrics_col2:
                    # Calculate debt-to-income ratio
                    monthly_income = data['income_annum'] / 12
                    loan_amount = data['loan_amount']
                    interest_rate = 0.08  # Assuming 8% interest rate
                    loan_term_months = data['loan_term'] * 12
                    
                    # Calculate monthly payment using the formula: P = L[i(1+i)^n]/[(1+i)^n-1]
                    monthly_interest = interest_rate / 12
                    monthly_payment = loan_amount * (monthly_interest * (1 + monthly_interest) ** loan_term_months) / ((1 + monthly_interest) ** loan_term_months - 1)
                    
                    debt_to_income = monthly_payment / monthly_income
                    
                    st.markdown("**Debt-to-Income**")
                    st.markdown(f"### {debt_to_income:.2f}")
                    
                    if debt_to_income <= 0.36:
                        st.markdown("Low - Manageable debt burden")
                    elif debt_to_income <= 0.43:
                        st.markdown("Medium - Moderate debt burden")
                    else:
                        st.markdown("High - Significant debt burden")
                
                with metrics_col3:
                    # Calculate assets-to-loan ratio
                    total_assets = data['residential_assets_value'] + data['commercial_assets_value'] + data['luxury_assets_value'] + data['bank_asset_value']
                    assets_to_loan = total_assets / loan_amount
                    
                    st.markdown("**Assets-to-Loan**")
                    st.markdown(f"### {assets_to_loan:.2f}")
                    
                    if assets_to_loan >= 2:
                        st.markdown("Strong - Excellent asset coverage")
                    elif assets_to_loan >= 1:
                        st.markdown("Fair - Minimal asset coverage")
                    else:
                        st.markdown("Weak - Insufficient asset coverage")
                
                # Monthly payment analysis
                st.subheader("Monthly Payment Analysis")
                
                payment_col1, payment_col2 = st.columns(2)
                
                with payment_col1:
                    st.markdown("**Monthly Payment**")
                    st.markdown(f"### ₹{monthly_payment:,.2f}")
                    
                    payment_ratio = monthly_payment / monthly_income
                    payment_percentage = payment_ratio * 100
                    
                    st.markdown(f"This represents {payment_percentage:.2f}% of your monthly income (₹{monthly_income:,.2f}).")
                    
                    if payment_ratio <= 0.28:
                        st.markdown("Excellent - Very affordable payment")
                    elif payment_ratio <= 0.36:
                        st.markdown("Good - Affordable payment")
                    elif payment_ratio <= 0.43:
                        st.markdown("Fair - Manageable payment")
                    else:
                        st.markdown("Poor - High payment burden")
                
                with payment_col2:
                    # Create pie chart for income allocation
                    remaining_income = monthly_income - monthly_payment
                    
                    income_allocation = pd.DataFrame({
                        'Category': ['Loan Payment', 'Remaining Income'],
                        'Amount': [monthly_payment, remaining_income]
                    })
                    
                    fig = px.pie(
                        income_allocation, 
                        values='Amount', 
                        names='Category',
                        color_discrete_sequence=['#FF9900', '#0066CC'],
                        hole=0.4
                    )
                    fig.update_layout(
                        margin=dict(l=20, r=20, t=20, b=20),
                        height=200
                    )
                    st.plotly_chart(fig, use_container_width=True)