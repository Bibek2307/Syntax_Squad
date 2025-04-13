import pandas as pd
import os
from models.loan_model import LoanApprovalModel

def test_trained_model():
    # Check if model exists
    model_path = 'models/loan_model.joblib'
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        print("Please run 'python src/train_model.py' first to train the model")
        return
    
    # Load the trained model
    print(f"Loading trained model from {model_path}...")
    model = LoanApprovalModel.load(model_path)
    
    # Print available categories for categorical features
    print("\nAvailable categories in the trained model:")
    for col, encoder in model.label_encoders.items():
        print(f"{col}: {list(encoder.classes_)}")
    
    # Test with a good application (likely to be approved)
    good_application = {
        'no_of_dependents': 2,
        'education': ' Graduate',  # Note the space before Graduate to match dataset format
        'self_employed': ' No',    # Note the space before No to match dataset format
        'income_annum': 8000000,
        'loan_amount': 15000000,
        'loan_term': 10,
        'cibil_score': 750,
        'residential_assets_value': 10000000,
        'commercial_assets_value': 5000000,
        'luxury_assets_value': 20000000,
        'bank_asset_value': 7000000
    }
    
    # Test with a risky application (likely to be rejected)
    risky_application = {
        'no_of_dependents': 5,
        'education': ' Not Graduate',  # Note the space before Not Graduate
        'self_employed': ' Yes',       # Note the space before Yes
        'income_annum': 2000000,
        'loan_amount': 15000000,
        'loan_term': 20,
        'cibil_score': 350,
        'residential_assets_value': 2000000,
        'commercial_assets_value': 1000000,
        'luxury_assets_value': 5000000,
        'bank_asset_value': 1000000
    }
    
    # Make predictions
    print("\nTesting with a good application:")
    good_result = model.predict(good_application)
    print(f"Prediction: {'Approved' if good_result['prediction'] == 1 else 'Rejected'}")
    print(f"Confidence: {good_result['probability']:.2%}")
    print(f"Explanation: {good_result['explanation']['text']}")
    
    print("\nTop 3 factors:")
    for i, factor in enumerate(good_result['feature_importance'][:3]):
        print(f"{i+1}. {factor['feature']}: {factor['importance']:.4f}")
    
    print("\nTesting with a risky application:")
    risky_result = model.predict(risky_application)
    print(f"Prediction: {'Approved' if risky_result['prediction'] == 1 else 'Rejected'}")
    print(f"Confidence: {risky_result['probability']:.2%}")
    print(f"Explanation: {risky_result['explanation']['text']}")
    
    print("\nTop 3 factors:")
    for i, factor in enumerate(risky_result['feature_importance'][:3]):
        print(f"{i+1}. {factor['feature']}: {factor['importance']:.4f}")

if __name__ == "__main__":
    test_trained_model()