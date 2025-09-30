import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import re
from datetime import datetime
import json

# ONNX dependencies
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

class LogAnomalyDetector:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def parse_log_line(self, line):
        """Parse a single log line into structured features"""
        pattern = r'\[(.*?)\] \[(.*?)\] \[(.*?)\]\s*(.*)'
        match = re.match(pattern, line)
        
        if match:
            timestamp, level, component, message = match.groups()
            return {
                'timestamp': timestamp,
                'level': level,
                'component': component,
                'message': message,
                'message_length': len(message),
                'has_error_keywords': int(any(keyword in message.lower() for keyword in 
                                            ['error', 'failed', 'exception', 'timeout', 'critical']))
            }
        return None
    
    def generate_training_data(self, num_samples=10000):
        """Generate synthetic log data for training with all required features"""
        levels = ['INFO', 'WARN', 'ERROR', 'DEBUG']
        components = ['auth', 'database', 'api', 'network', 'storage']
        messages = {
            'normal': [
                'User login successful',
                'Database query executed',
                'API response sent',
                'Cache updated',
                'Request processed',
                'Connection established'
            ],
            'suspicious': [
                'Failed login attempt',
                'Database connection timeout',
                'API rate limit exceeded',
                'Memory allocation failed',
                'Unauthorized access attempt'
            ]
        }
        
        logs = []
        for i in range(num_samples):
            if np.random.random() < 0.95:
                msg_type = 'normal'
            else:
                msg_type = 'suspicious'
                
            message = np.random.choice(messages[msg_type])
                
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'level': np.random.choice(levels, p=[0.7, 0.2, 0.05, 0.05]),
                'component': np.random.choice(components),
                'message': message,
                'message_length': len(message),
                'has_error_keywords': int(any(keyword in message.lower() for keyword in 
                                            ['error', 'failed', 'exception', 'timeout', 'critical']))
            }
            logs.append(log_entry)
            
        return pd.DataFrame(logs)
    
    def extract_features(self, df):
        """Extract features from parsed log data"""
        # Convert timestamp to numerical features
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Encode categorical variables
        categorical_columns = ['level', 'component']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])
        
        # Select feature columns
        feature_columns = ['level', 'component', 'message_length', 
                          'has_error_keywords', 'hour', 'day_of_week', 'is_weekend']
        
        return df[feature_columns], feature_columns
    
    def train(self, df=None):
        """Train the isolation forest model"""
        if df is None:
            df = self.generate_training_data()
        
        # Extract features
        features, self.feature_columns = self.extract_features(df)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Train model
        self.model.fit(scaled_features)
        
        print(f"Model trained on {len(features)} samples")
        print(f"Features used: {self.feature_columns}")
        
        return self.model
    
    def save_model_onnx(self, filepath='log_anomaly_model.onnx'):
        """Save the trained model in ONNX format"""
        # Create initial type for ONNX (batch_size, 7 features)
        initial_type = [('float_input', FloatTensorType([None, 7]))]
        
        # Convert to ONNX with specific target opset
        try:
            # Try with opset 12 first (most compatible)
            onnx_model = convert_sklearn(
                self.model, 
                initial_types=initial_type,
                target_opset=12
            )
        except Exception as e:
            print(f"Opset 12 failed: {e}")
            # Fall back to opset 11
            try:
                onnx_model = convert_sklearn(
                    self.model, 
                    initial_types=initial_type,
                    target_opset=11
                )
            except Exception as e2:
                print(f"Opset 11 failed: {e2}")
                # Final fallback with specific domain version
                onnx_model = convert_sklearn(
                    self.model, 
                    initial_types=initial_type,
                    target_opset={'ai.onnx.ml': 3, '': 10}
                )
        
        # Save ONNX model
        onnx.save(onnx_model, filepath)
        print(f"ONNX model saved to {filepath}")
        
        # Save preprocessors separately
        preprocessor_data = {
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(preprocessor_data, 'preprocessors.joblib')
        print("Preprocessors saved to preprocessors.joblib")
    
    def save_model(self, filepath='log_anomaly_model.joblib'):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

if __name__ == "__main__":
    # Train and save models
    detector = LogAnomalyDetector()
    detector.train()
    
    # Save both formats
    detector.save_model()
    detector.save_model_onnx()
    
    print("✅ ML model training completed successfully!")