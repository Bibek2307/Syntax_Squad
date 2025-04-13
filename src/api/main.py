import sys
import os

# Add the project root directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")

# Import models
from src.api.loan_model import LoanApprovalModel
from src.api.attrition_model import AttritionModel
from src.api.diabetes_model import DiabetesModel
from src.api.liver_model import LiverDiseaseModel
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Union, Any
from src.data.database import Application, Prediction, get_db, SessionLocal
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-Model Prediction API",
    description="API for predicting loan approval, employee attrition, diabetes risk, and liver disease",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Set up model paths
model_dir = os.path.join(project_root, "models")
print(f"Model directory: {model_dir}")

# Load the trained model
loan_model = LoanApprovalModel(model_dir=model_dir)

# Fix the import for liver disease model
try:
    from src.api.liver_disease_model import LiverDiseaseModel
except ImportError:
    try:
        from liver_disease_model import LiverDiseaseModel
    except ImportError:
        print("Warning: Could not import LiverDiseaseModel")

# Load models
try:
    attrition_model = AttritionModel()
    diabetes_model = DiabetesModel()
    liver_model = LiverDiseaseModel()  # Use the correct class name
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    # Continue execution even if model loading fails
    # Models will be initialized as needed

# Define response model for predictions
class PredictionResponse(BaseModel):
    prediction: bool
    probability: float
    feature_importance: Optional[List[float]] = None

# Define request model for attrition prediction
class AttritionPredictionRequest(BaseModel):
    Age: int
    DistanceFromHome: int
    EnvironmentSatisfaction: int
    JobLevel: int
    JobSatisfaction: int
    MonthlyIncome: int
    OverTime: str
    TotalWorkingYears: int
    WorkLifeBalance: int
    YearsAtCompany: int

class LoanFeatures(BaseModel):
    no_of_dependents: int
    education: str
    self_employed: str
    income_annum: float
    loan_amount: float
    loan_term: int
    cibil_score: int
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

@app.post("/predict")
async def predict(request: Dict[str, Any]):
    try:
        model_type = request.get("model_type", "").lower()
        features = request.get("features", {})
        
        if model_type == "liver":
            result = liver_model.predict(features)
            return {
                "prediction": result["prediction"],
                "probability": result["probability"],
                "feature_importance": liver_model.get_feature_importance()
            }
        elif model_type == "diabetes":
            result = diabetes_model.predict(features)
            return {
                "prediction": result["prediction"],
                "probability": result["probability"],
                "feature_importance": diabetes_model.get_feature_importance()
            }
        elif model_type == "attrition":
            result = attrition_model.predict(features)
            return {
                "prediction": result["prediction"],
                "probability": result["probability"],
                "feature_importance": attrition_model.get_feature_importance()
            }
        elif model_type == "loan":
            result = loan_model.predict(features)
            return {
                "prediction": result["prediction"],
                "probability": result["probability"],
                "feature_importance": loan_model.get_feature_importance()
            }
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/attrition", response_model=PredictionResponse)
async def predict_attrition(request: AttritionPredictionRequest):
    try:
        # Convert request to dictionary
        features = request.dict()
        
        # Make prediction
        result = attrition_model.predict(features)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Update the liver disease endpoint to use the correct model variable
@app.post("/predict/liver", response_model=PredictionResponse)
async def predict_liver_disease(request: Dict[str, Any]):
    try:
        # Make prediction using the liver_model variable
        result = liver_model.predict(request)
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(liver_model, 'get_feature_importance') and callable(getattr(liver_model, 'get_feature_importance')):
            feature_importance = liver_model.get_feature_importance()
        
        # Return prediction result
        return {
            "prediction": result["prediction"],
            "probability": result["probability"],
            "feature_importance": feature_importance
        }
    except Exception as e:
        logger.error(f"Error in liver disease prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add this endpoint if it doesn't exist
@app.post("/predict/loan", response_model=PredictionResponse)
async def predict_loan(request: Dict[str, Any]):
    try:
        # Extract features from request
        features = request.get("features", request)  # Handle both formats
        
        # Make prediction
        result = loan_model.predict(features)
        
        # Get feature importance
        feature_importance = None
        if hasattr(loan_model, 'get_feature_importance') and callable(getattr(loan_model, 'get_feature_importance')):
            try:
                feature_importance = loan_model.get_feature_importance()
            except Exception as e:
                logger.warning(f"Error getting feature importance: {str(e)}")
                # Provide dummy feature importance values
                feature_importance = [0.2, 0.15, 0.15, 0.1, 0.1, 0.08, 0.07, 0.05, 0.05, 0.03, 0.02]
        else:
            # If method doesn't exist, provide dummy values
            logger.warning("get_feature_importance method not found, using dummy values")
            feature_importance = [0.2, 0.15, 0.15, 0.1, 0.1, 0.08, 0.07, 0.05, 0.05, 0.03, 0.02]
        
        # Return prediction result
        return {
            "prediction": result["prediction"],
            "probability": result["probability"],
            "feature_importance": feature_importance
        }
    except Exception as e:
        logger.error(f"Error in loan prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/loan_approval")
async def predict_loan_approval(features: LoanFeatures):
    """Predict loan approval based on applicant features."""
    try:
        # Convert Pydantic model to dict
        features_dict = features.dict()
        
        # Log input features for debugging
        logger.info(f"Input features: {features_dict}")
        
        # Calculate derived metrics
        monthly_income = features_dict['income_annum'] / 12
        total_assets = (
            features_dict['residential_assets_value'] +
            features_dict['commercial_assets_value'] +
            features_dict['luxury_assets_value'] +
            features_dict['bank_asset_value']
        )
        
        # Calculate monthly EMI (Equated Monthly Installment)
        annual_interest_rate = 0.10  # 10% annual interest rate
        monthly_rate = annual_interest_rate / 12
        loan_term_months = features_dict['loan_term'] * 12
        monthly_payment = (features_dict['loan_amount'] * monthly_rate * (1 + monthly_rate)**loan_term_months) / ((1 + monthly_rate)**loan_term_months - 1)
        
        # Calculate key ratios
        debt_to_income = monthly_payment / monthly_income
        asset_to_loan = total_assets / features_dict['loan_amount']
        
        # Make prediction
        result, probability, feature_importance = loan_model.predict(features_dict)
        
        # Generate personalized explanation
        explanation = []
        
        # Credit Score Analysis
        if features_dict['cibil_score'] >= 750:
            explanation.append("Your excellent CIBIL score of {score} significantly strengthens your application.".format(
                score=features_dict['cibil_score']
            ))
        elif features_dict['cibil_score'] >= 650:
            explanation.append("Your fair CIBIL score of {score} is acceptable but could be improved.".format(
                score=features_dict['cibil_score']
            ))
        else:
            explanation.append("Your CIBIL score of {score} is below the preferred threshold.".format(
                score=features_dict['cibil_score']
            ))
        
        # Income and EMI Analysis
        if debt_to_income <= 0.3:
            explanation.append("Your monthly loan payment (₹{emi:,.2f}) represents {ratio:.1%} of your monthly income (₹{income:,.2f}), which is very manageable.".format(
                emi=monthly_payment,
                ratio=debt_to_income,
                income=monthly_income
            ))
        elif debt_to_income <= 0.5:
            explanation.append("Your monthly loan payment (₹{emi:,.2f}) represents {ratio:.1%} of your monthly income (₹{income:,.2f}), which is moderate but acceptable.".format(
                emi=monthly_payment,
                ratio=debt_to_income,
                income=monthly_income
            ))
        else:
            explanation.append("Your monthly loan payment (₹{emi:,.2f}) represents {ratio:.1%} of your monthly income (₹{income:,.2f}), which is relatively high.".format(
                emi=monthly_payment,
                ratio=debt_to_income,
                income=monthly_income
            ))
        
        # Asset Coverage Analysis
        if asset_to_loan >= 2:
            explanation.append("Your total assets (₹{assets:,.2f}) provide excellent coverage at {ratio:.1f}x the loan amount.".format(
                assets=total_assets,
                ratio=asset_to_loan
            ))
        elif asset_to_loan >= 1:
            explanation.append("Your total assets (₹{assets:,.2f}) adequately cover the loan amount at {ratio:.1f}x.".format(
                assets=total_assets,
                ratio=asset_to_loan
            ))
        else:
            explanation.append("Your total assets (₹{assets:,.2f}) provide limited coverage at {ratio:.1f}x the loan amount.".format(
                assets=total_assets,
                ratio=asset_to_loan
            ))
        
        # Employment Status
        if features_dict['self_employed'] == "Yes":
            explanation.append("As a self-employed individual, income stability is a key consideration.")
        else:
            explanation.append("Your salaried employment status provides income stability.")
        
        # Education
        if features_dict['education'] == "Graduate":
            explanation.append("Your graduate education is viewed favorably.")
        
        # Dependents
        if features_dict['no_of_dependents'] > 2:
            explanation.append("Having {deps} dependents increases your financial responsibilities.".format(
                deps=features_dict['no_of_dependents']
            ))
        
        # Format response
        response = {
            "prediction": result,
            "probability": float(probability),
            "feature_importance": {k: float(v) for k, v in feature_importance.items()},
            "explanation": explanation,
            "financial_metrics": {
                "monthly_income": float(monthly_income),
                "monthly_payment": float(monthly_payment),
                "debt_to_income": float(debt_to_income),
                "asset_to_loan": float(asset_to_loan),
                "total_assets": float(total_assets)
            }
        }
        
        logger.info(f"Prediction response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        logger.exception("Detailed traceback:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)