import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import shap
import logging
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoanApprovalModel:
    """Loan approval model for predicting loan application outcomes."""
    
    def __init__(self, model_dir: str = "models", load_model: bool = True):
        """Initialize the loan approval model.
        
        Args:
            model_dir (str): Directory containing the trained model components
            load_model (bool): Whether to load existing model components
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.explainer = None
        
        # Initialize label encoders for categorical columns
        self.categorical_columns = ['education', 'self_employed']
        self.label_encoders = {}
        for col in self.categorical_columns:
            self.label_encoders[col] = LabelEncoder()
        
        # Load model components if requested
        if load_model:
            self.load_components()
        
    def load_components(self):
        """Load the trained model and preprocessing components."""
        try:
            logger.info("Loading model components...")
            
            # Load model
            model_path = os.path.join(self.model_dir, 'loan_model.joblib')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            self.model = joblib.load(model_path)
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'loan_scaler.joblib')
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler file not found at {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            
            # Load label encoders
            encoders_path = os.path.join(self.model_dir, 'loan_label_encoders.joblib')
            if not os.path.exists(encoders_path):
                raise FileNotFoundError(f"Label encoders file not found at {encoders_path}")
            self.label_encoders = joblib.load(encoders_path)
            
            # Load feature names
            features_path = os.path.join(self.model_dir, 'loan_feature_names.joblib')
            if not os.path.exists(features_path):
                raise FileNotFoundError(f"Feature names file not found at {features_path}")
            self.feature_names = joblib.load(features_path)
            
            # Try to load explainer if available
            explainer_path = os.path.join(self.model_dir, 'loan_explainer.joblib')
            if os.path.exists(explainer_path):
                self.explainer = joblib.load(explainer_path)
            
            logger.info("Model components loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model components: {str(e)}")
            raise
            
    def save(self, output_dir: str = "models") -> None:
        """Save model components to disk.
        
        Args:
            output_dir (str): Directory to save model components
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(output_dir, "loan_model.joblib")
            joblib.dump(self.model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(output_dir, "loan_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            
            # Save label encoders
            encoders_path = os.path.join(output_dir, "loan_label_encoders.joblib")
            joblib.dump(self.label_encoders, encoders_path)
            
            # Save feature names
            features_path = os.path.join(output_dir, "loan_feature_names.joblib")
            joblib.dump(self.feature_names, features_path)
            
            # Save explainer if available
            if self.explainer is not None:
                explainer_path = os.path.join(output_dir, "loan_explainer.joblib")
                joblib.dump(self.explainer, explainer_path)
            
            logger.info(f"Model components saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model components: {str(e)}")
            raise
            
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train the loan approval model.
        
        Args:
            X (pd.DataFrame): Training features
            y (pd.Series): Target values
        """
        try:
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Preprocess features
            X_processed = self._preprocess_features(X, is_training=True)
            
            # Initialize and train model
            logger.info("Training RandomForestClassifier...")
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            # Fit the model
            self.model.fit(X_processed, y)
            
            # Initialize SHAP explainer
            logger.info("Initializing SHAP explainer...")
            self.explainer = shap.TreeExplainer(self.model)
            
            logger.info("Model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def predict(self, features: Dict[str, Any]) -> Tuple[str, float, Dict[str, float]]:
        """Make a prediction for loan approval.
        
        Args:
            features (Dict[str, Any]): Input features for prediction
            
        Returns:
            Tuple[str, float, Dict[str, float]]: Prediction result, probability, and feature importance
        """
        try:
            # Validate required features
            required_features = [
                'no_of_dependents', 'education', 'self_employed', 'income_annum',
                'loan_amount', 'loan_term', 'cibil_score', 'residential_assets_value',
                'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'
            ]
            
            missing_features = [f for f in required_features if f not in features]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Calculate derived features
            features = features.copy()  # Create a copy to avoid modifying the input
            features['debt_to_income'] = features['loan_amount'] / features['income_annum']
            features['total_assets'] = (
                features['residential_assets_value'] +
                features['commercial_assets_value'] +
                features['luxury_assets_value'] +
                features['bank_asset_value']
            )
            features['asset_to_loan'] = features['total_assets'] / features['loan_amount']
            
            # Create DataFrame with all required features
            X = pd.DataFrame([features])
            
            # Ensure all required features are present
            required_features = self.feature_names
            missing_features = set(required_features) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing required features after preprocessing: {missing_features}")
            
            # Reorder columns to match training data
            X = X[required_features]
            
            # Encode categorical features first
            for feature in ['education', 'self_employed']:
                try:
                    X[feature] = self.label_encoders[feature].transform(X[feature].astype(str))
                except Exception as e:
                    raise ValueError(f"Error encoding {feature}: {str(e)}. Valid values are: {self.label_encoders[feature].classes_}")
            
            # Scale numerical features
            numerical_features = [f for f in X.columns if f not in ['education', 'self_employed']]
            X[numerical_features] = self.scaler.transform(X[numerical_features])
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probability = self.model.predict_proba(X)[0][1]  # Probability of approval
            
            # Calculate feature importance
            feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
            
            # Map prediction to string
            result = "Approved" if prediction == 1 else "Rejected"
            
            return result, probability, feature_importance
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            logger.exception("Detailed traceback:")
            raise
            
    def _preprocess_features(self, X: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Preprocess features for model training or prediction.
        
        Args:
            X (pd.DataFrame): Input features
            is_training (bool): Whether preprocessing is for training
            
        Returns:
            pd.DataFrame: Preprocessed features
        """
        try:
            # Create copy to avoid modifying original data
            df = X.copy()
            
            # Encode categorical variables
            for col in self.categorical_columns:
                if col in df.columns:
                    if is_training:
                        df[col] = self.label_encoders[col].fit_transform(df[col])
                    else:
                        df[col] = self.label_encoders[col].transform(df[col])
            
            # Scale numerical features
            numerical_features = [f for f in df.columns if f not in self.categorical_columns]
            if is_training:
                df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
            else:
                df[numerical_features] = self.scaler.transform(df[numerical_features])
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {str(e)}")
            raise

    def get_feature_importance(self):
        """Return feature importance values from the model."""
        try:
            if self.model is None:
                print("Model not loaded, cannot get feature importance")
                return None
                
            # For tree-based models like RandomForest, we can get feature importance directly
            if hasattr(self.model, 'feature_importances_'):
                # Return the feature importances as a list
                return self.model.feature_importances_.tolist()
            elif hasattr(self.model, 'coef_'):
                # For linear models, use coefficients as importance
                return np.abs(self.model.coef_[0]).tolist()
            else:
                # Create dummy feature importance if not available
                print("Feature importance not available in model, returning dummy values")
                # Create dummy values for each feature
                feature_count = len(self.feature_names) if hasattr(self, 'feature_names') else 10
                return [0.1] * feature_count
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            # Return dummy values as fallback
            feature_count = len(self.feature_names) if hasattr(self, 'feature_names') else 10
            return [0.1] * feature_count