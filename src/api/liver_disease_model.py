import pandas as pd
import numpy as np
import pickle
import os
import sys
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class LiverDiseaseModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Get the project root directory
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Set paths for model files
        self.model_path = os.path.join(self.project_root, 'models', 'liver_disease_model.pkl')
        
        # Default feature names
        self.default_feature_names = [
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
            'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
            'Aspartate_Aminotransferase', 'Total_Protiens', 
            'Albumin', 'Albumin_and_Globulin_Ratio'
        ]
        
        # Initialize feature names
        self.feature_names = self.default_feature_names
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Load the model or create a dummy one if not found
        self.load_model()

    def load_model(self):
        """Load the trained model from disk."""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    try:
                        model_data = pickle.load(f, encoding='latin1')
                        if isinstance(model_data, dict):
                            self.model = model_data.get('model')
                            self.scaler = model_data.get('scaler')
                            if self.model is None or self.scaler is None:
                                raise ValueError("Model or scaler missing from loaded data")
                        else:
                            self.model = model_data
                            # Create a new scaler if not found in model data
                            self.scaler = StandardScaler()
                        logger.info("Liver disease model loaded successfully")
                    except Exception as inner_e:
                        logger.error(f"Error during pickle load: {str(inner_e)}")
                        raise ValueError(f"Failed to load liver disease model: {str(inner_e)}")
            else:
                raise FileNotFoundError(f"Liver disease model file not found at {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading liver disease model: {str(e)}")
            raise ValueError(f"Failed to load liver disease model: {str(e)}")

    # Remove the _create_dummy_model method entirely

    def _create_dummy_model(self):
        """Create a dummy model for testing purposes."""
        try:
            logger.warning("Creating dummy liver disease model")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            
            # Create dummy data to fit the scaler and model
            dummy_data = pd.DataFrame(np.random.randn(100, len(self.feature_names)), 
                                    columns=self.feature_names)
            self.scaler.fit(dummy_data)
            
            # Fit the model with dummy data
            dummy_target = np.random.randint(0, 2, 100)
            self.model.fit(dummy_data, dummy_target)
            
            # Save the dummy model
            self.save_model()
            
            logger.info("Dummy liver disease model created and saved successfully")
        except Exception as e:
            logger.error(f"Error creating dummy liver disease model: {str(e)}")
            raise

    def save_model(self):
        """Save the model and scaler together in one file."""
        try:
            # Create a dictionary containing both model and scaler
            model_data = {
                'model': self.model,
                'scaler': self.scaler
            }
            
            # Save to file
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info("Liver disease model and scaler saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving liver disease model: {str(e)}")
            raise

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction using the trained model."""
        try:
            if self.model is None:
                raise ValueError(f"Model not loaded. Please ensure model file exists at {self.model_path} and is valid.")
            
            print(f"Input features for liver disease prediction: {features}")
            
            # Convert string inputs to appropriate numeric types
            processed_features = {}
            for key, value in features.items():
                if key == 'Gender':
                    # Convert 'Male'/'Female' to 1/0
                    if isinstance(value, str):
                        processed_features[key] = 1 if value.lower() in ['male', 'm', '1'] else 0
                    else:
                        processed_features[key] = 1 if value else 0
                else:
                    # Convert other values to appropriate numeric types
                    try:
                        processed_features[key] = float(value)
                    except (ValueError, TypeError):
                        # Handle conversion errors
                        raise ValueError(f"Invalid value for feature {key}: {value}. Expected numeric value.")
            
            # Create DataFrame with processed values
            X = pd.DataFrame([processed_features])
            
            # Ensure all required columns are present
            for col in self.feature_names:
                if col not in X.columns:
                    raise ValueError(f"Missing required feature: {col}")
            
            # Ensure columns are in the correct order
            X = X[self.feature_names]
            
            # Convert all data to float64 to ensure compatibility
            X = X.astype(float)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = bool(self.model.predict(X_scaled)[0])
            
            # Get probability
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_scaled)[0]
                probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
            else:
                probability = 0.5 + (float(self.model.decision_function(X_scaled)[0]) / 10)
                probability = max(0, min(1, probability))  # Clamp between 0 and 1
            
            return {
                "prediction": prediction,
                "probability": probability
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Error during prediction: {str(e)}")

    def train_model(self, X, y):
        """Train the model with the given data."""
        try:
            logger.info("Starting liver disease model training...")
            
            # Initialize the scaler and scale the features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Initialize and train the model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(X_scaled, y)
            
            # Save the model and scaler
            self.save_model()
            
            logger.info("Liver disease model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in train_model: {str(e)}")
            raise

    def get_feature_importance(self):
        """Return feature importance values from the model."""
        try:
            if self.model is None:
                logger.warning("Model not loaded, cannot get feature importance")
                return None
                
            # For RandomForestClassifier, we can get feature importance directly
            if hasattr(self.model, 'feature_importances_'):
                # Return the feature importances as a list
                return self.model.feature_importances_.tolist()
            else:
                # Create dummy feature importance if not available
                logger.warning("Feature importance not available in model, returning dummy values")
                return [0.15, 0.05, 0.12, 0.08, 0.18, 0.14, 0.10, 0.08, 0.06, 0.04]
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            # Return dummy values as fallback
            return [0.15, 0.05, 0.12, 0.08, 0.18, 0.14, 0.10, 0.08, 0.06, 0.04]

def train_model():
    """Train and save the liver disease prediction model"""
    try:
        model = LiverDiseaseModel()
        
        # Get absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_file = os.path.join(project_root, "data", "indian_liver_patient.csv")
        
        print(f"Loading data from: {data_file}")
        print(f"Model will be saved to: {model.model_path}")
        
        # Ensure data file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found at {data_file}")
        
        # Load data
        print("Loading and preparing data...")
        data = pd.read_csv(data_file)
        
        # Preprocess data
        data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
        
        # Handle missing values
        data = data.fillna(data.median())
        
        # Select features and target
        X = data[model.feature_names]
        y = data['Dataset']  # Assuming 'Dataset' is the target column
        
        # Train the model
        print("Training model...")
        model.train_model(X, y)
        print("Model trained and saved successfully")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    train_model()