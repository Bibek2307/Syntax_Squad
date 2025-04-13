import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import logging
import shap
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import from src
from src.api.loan_model import LoanApprovalModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoanModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """Load and preprocess the loan approval dataset."""
        logger.info("Loading dataset...")
        df = pd.read_csv(file_path)
        
        # Convert loan status to binary
        df['loan_status'] = df['loan_status'].map({'Approved': 1, 'Rejected': 0})
        
        # Calculate derived features
        df['debt_to_income'] = df['loan_amount'] / df['income_annum']
        df['total_assets'] = df['residential_assets_value'] + df['commercial_assets_value'] + df['luxury_assets_value'] + df['bank_asset_value']
        df['asset_to_loan'] = df['total_assets'] / df['loan_amount']
        
        # Define features
        numerical_features = [
            'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term',
            'cibil_score', 'residential_assets_value', 'commercial_assets_value',
            'luxury_assets_value', 'bank_asset_value', 'debt_to_income',
            'total_assets', 'asset_to_loan'
        ]
        
        categorical_features = ['education', 'self_employed']
        
        # Encode categorical features
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder()
            df[feature] = self.label_encoders[feature].fit_transform(df[feature])
        
        # Prepare X and y
        X = df[numerical_features + categorical_features]
        y = df['loan_status']
        
        return X, y, numerical_features, categorical_features
        
    def train(self, X, y, numerical_features, categorical_features):
        """Train the model and evaluate its performance."""
        logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale numerical features
        logger.info("Scaling numerical features...")
        X_train[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        X_test[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        # Train the model
        logger.info("Training the model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        logger.info("Evaluating the model...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info("Classification Report:")
        logger.info(report)
        
        return accuracy, report
        
    def save_model(self, save_dir='models'):
        """Save the trained model and preprocessing objects."""
        logger.info("Saving model components...")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model components
        joblib.dump(self.model, os.path.join(save_dir, 'loan_model.joblib'))
        joblib.dump(self.scaler, os.path.join(save_dir, 'loan_scaler.joblib'))
        joblib.dump(self.label_encoders, os.path.join(save_dir, 'loan_label_encoders.joblib'))
        
        logger.info("Model components saved successfully.")

def train_loan_model():
    # Create models directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Load the dataset
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "loan_approval_dataset.csv")
    data = pd.read_csv(data_path)
    
    # Clean column names and string values by removing leading/trailing spaces
    data.columns = data.columns.str.strip()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = data[col].str.strip()
    
    # Remove rows with NaN values
    data = data.dropna()
    
    # Convert loan status to binary
    data['loan_status'] = data['loan_status'].map({'Approved': 1, 'Rejected': 0})
    
    # Separate features and target
    X = data.drop(['loan_status', 'loan_id'], axis=1)  # Also drop loan_id as it's not a feature
    y = data['loan_status']
    
    print("Dataset shape:", X.shape)
    print("Number of approved loans:", sum(y == 1))
    print("Number of rejected loans:", sum(y == 0))
    
    # Initialize model without loading existing components
    model = LoanApprovalModel(model_dir=model_dir, load_model=False)
    
    # Train the model
    model.train(X, y)
    
    # Save the model
    model.save(model_dir)
    
    print(f"Model trained and saved successfully in {model_dir}!")

if __name__ == "__main__":
    train_loan_model()