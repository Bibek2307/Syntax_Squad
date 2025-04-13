import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import sys
from typing import List

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

class AttritionModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.model_path = os.path.join(project_root, "models", "attrition_model.pkl")
        self.preprocessor_path = os.path.join(project_root, "models", "attrition_preprocessor.pkl")
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Define the features we'll use
        self.numeric_features = [
            'Age', 'DistanceFromHome', 'EnvironmentSatisfaction',
            'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
            'TotalWorkingYears', 'WorkLifeBalance', 'YearsAtCompany'
        ]
        self.categorical_features = ['OverTime']
        
        # Try to load existing model and preprocessor
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
        except:
            print("No existing model found. Please train the model first.")
    
    def preprocess_data(self, X):
        """Preprocess the input data"""
        # Create preprocessing steps for numeric and categorical data
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse=False)
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # Drop any columns not specified in features
        )
        
        return self.preprocessor.fit_transform(X)
    
    def train(self, X, y):
        """Train the model with the given data"""
        # Preprocess the data
        X_processed = self.preprocess_data(X)
        
        # Create and train the model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_processed, y)
        
        # Save the model and preprocessor
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
    
    def predict(self, features):
        """Make a prediction using the trained model."""
        try:
            if self.model is None:
                raise ValueError("Model not loaded. Please ensure model file exists and is valid.")
            
            print(f"Input features: {features}")
            
            # Convert string inputs to appropriate types
            processed_features = {}
            for key, value in features.items():
                if key == 'OverTime':
                    # Convert 'Yes'/'No' to 1/0
                    if isinstance(value, str):
                        processed_features[key] = 1 if value.lower() in ['yes', 'true', '1'] else 0
                    else:
                        processed_features[key] = 1 if value else 0
                else:
                    # Convert other values to appropriate numeric types
                    try:
                        processed_features[key] = float(value)
                    except (ValueError, TypeError):
                        # Handle conversion errors
                        raise ValueError(f"Invalid value for feature {key}: {value}. Expected numeric value.")
            
            print(f"Processed features: {processed_features}")
            
            # Create DataFrame with processed values
            X = pd.DataFrame([processed_features])
            
            # Ensure all required columns are present
            required_columns = self.numeric_features + self.categorical_features
            
            for col in required_columns:
                if col not in X.columns:
                    raise ValueError(f"Missing required feature: {col}")
            
            # Ensure columns are in the correct order for the preprocessor
            X = X[required_columns]
            
            # Debug information
            print(f"Input data types before conversion: {X.dtypes}")
            
            # Convert all numeric columns to float64
            for col in self.numeric_features:
                X[col] = pd.to_numeric(X[col], errors='coerce').astype(np.float64)
            
            # Convert categorical columns to appropriate types
            for col in self.categorical_features:
                X[col] = X[col].astype(np.int64)
            
            print(f"Input data types after conversion: {X.dtypes}")
            print(f"Input data: {X.to_dict('records')}")
            
            # Check for NaN values
            if X.isnull().any().any():
                print(f"Warning: NaN values detected in input: {X.isnull().sum()}")
                # Fill NaN values with appropriate defaults
                X = X.fillna(X.mean())
            
            # Use preprocessor
            if self.preprocessor is not None:
                try:
                    X_processed = self.preprocessor.transform(X)
                    print("Preprocessing successful")
                except Exception as e:
                    print(f"Error during preprocessing: {str(e)}")
                    # Try direct prediction without preprocessing as fallback
                    try:
                        # For direct prediction, we need to handle categorical features manually
                        # Convert 'OverTime' to one-hot encoding manually
                        X_direct = X.copy()
                        X_direct['OverTime_Yes'] = X_direct['OverTime']
                        X_direct = X_direct.drop('OverTime', axis=1)
                        
                        # Make prediction with direct features
                        prediction = bool(self.model.predict(X_direct.values)[0])
                        probability = float(self.model.predict_proba(X_direct.values)[0][1])
                        
                        print("Used direct prediction as fallback")
                        return {
                            "prediction": prediction,
                            "probability": probability
                        }
                    except Exception as direct_error:
                        print(f"Direct prediction also failed: {str(direct_error)}")
                        raise ValueError(f"Failed to process input data: {str(e)}")
            else:
                # If no preprocessor, just use the raw values
                X_processed = X.values
                print("No preprocessor available, using raw values")
            
            # Make prediction
            prediction = bool(self.model.predict(X_processed)[0])
            probability = float(self.model.predict_proba(X_processed)[0][1])
            
            print(f"Prediction result: {prediction}, probability: {probability}")
            
            return {
                "prediction": prediction,
                "probability": probability
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Error during prediction: {str(e)}")

    def get_feature_importance(self) -> List[float]:
        """Get the feature importance scores as a list of floats."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                # Convert feature importances to a list of floats
                return [float(x) for x in self.model.feature_importances_]
            return None
        except Exception as e:
            print(f"Error getting feature importance: {str(e)}")
            return None

def train_model():
    """Train and save the attrition prediction model"""
    try:
        model = AttritionModel()
        
        # Get absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_file = os.path.join(project_root, "data", "HR-Employee-Attrition.csv")
        model_dir = os.path.join(project_root, 'models')
        
        print(f"Loading data from: {data_file}")
        print(f"Model will be saved to: {model_dir}")
        
        # Ensure data file exists
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found at {data_file}")
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load data
        print("Loading and preparing data...")
        data = pd.read_csv(data_file)
        
        # Select only the features we want to use
        features = model.numeric_features + model.categorical_features
        print(f"Using features: {features}")
        
        X = data[features]
        y = data['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Train the model
        print("Training model...")
        model.train(X, y)
        print("Model trained and saved successfully")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    train_model()