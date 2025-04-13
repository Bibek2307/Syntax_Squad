import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

class BiasDetector:
    def __init__(self, model, sensitive_features=None):
        self.model = model
        self.sensitive_features = sensitive_features or []
        self.metrics = {}
        
    def analyze_bias(self, X, y, sensitive_feature):
        """Analyze bias for a sensitive feature"""
        if sensitive_feature not in X.columns:
            raise ValueError(f"Sensitive feature {sensitive_feature} not found in data")
            
        # Get unique values for the sensitive feature
        unique_values = X[sensitive_feature].unique()
        
        results = {}
        for value in unique_values:
            # Filter data for this value
            mask = X[sensitive_feature] == value
            X_group = X[mask]
            y_group = y[mask]
            
            # Skip if no data
            if len(X_group) == 0:
                continue
                
            # Make predictions
            if hasattr(self.model, 'predict'):
                y_pred = self.model.predict(X_group)
            else:
                y_pred = self.model.model.predict(X_group)
                
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_group, y_pred).ravel()
            
            # Calculate rates
            acceptance_rate = (tp + fp) / len(y_group)
            true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            results[value] = {
                "count": len(X_group),
                "acceptance_rate": acceptance_rate,
                "true_positive_rate": true_positive_rate,
                "false_positive_rate": false_positive_rate
            }
            
        self.metrics[sensitive_feature] = results
        return results
        
    def calculate_disparate_impact(self, sensitive_feature, privileged_value):
        """Calculate disparate impact"""
        if sensitive_feature not in self.metrics:
            raise ValueError(f"No metrics available for {sensitive_feature}. Run analyze_bias first.")
            
        metrics = self.metrics[sensitive_feature]
        if privileged_value not in metrics:
            raise ValueError(f"Privileged value {privileged_value} not found in metrics")
            
        privileged_rate = metrics[privileged_value]["acceptance_rate"]
        
        disparate_impact = {}
        for value, data in metrics.items():
            if value != privileged_value:
                unprivileged_rate = data["acceptance_rate"]
                # Disparate impact is the ratio of unprivileged to privileged acceptance rates
                # A value < 0.8 is often considered problematic
                di = unprivileged_rate / privileged_rate if privileged_rate > 0 else 0
                disparate_impact[value] = di
                
        return disparate_impact
        
    def generate_bias_report(self):
        """Generate a comprehensive bias report"""
        report = {}
        
        for feature in self.metrics:
            feature_report = {
                "metrics": self.metrics[feature],
                "disparate_impact": {}
            }
            
            # Find privileged group (highest acceptance rate)
            acceptance_rates = {value: data["acceptance_rate"] 
                               for value, data in self.metrics[feature].items()}
            privileged_value = max(acceptance_rates, key=acceptance_rates.get)
            
            # Calculate disparate impact
            feature_report["disparate_impact"] = self.calculate_disparate_impact(
                feature, privileged_value
            )
            
            # Add to report
            report[feature] = feature_report
            
        return report