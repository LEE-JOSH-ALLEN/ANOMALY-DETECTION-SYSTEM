import grpc
from concurrent import futures
import logging
import joblib
import numpy as np
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_model import LogAnomalyDetector
from python_grpc import anomaly_detection_pb2, anomaly_detection_pb2_grpc

class AnomalyDetectionServicer(anomaly_detection_pb2_grpc.AnomalyDetectionServicer):
    def __init__(self):
        self.detector = LogAnomalyDetector()
        # Load pre-trained model
        try:
            model_data = joblib.load('log_anomaly_model.joblib')
            self.detector.model = model_data['model']
            self.detector.label_encoders = model_data['label_encoders']
            self.detector.scaler = model_data['scaler']
            self.detector.feature_columns = model_data['feature_columns']
            print("Model loaded successfully")
        except:
            print("No pre-trained model found, using rule-based fallback")
            self.detector = None

    def DetectAnomaly(self, request, context):
        try:
            # Convert gRPC request to features
            features = {
                'timestamp': request.timestamp,
                'level': request.level,
                'component': request.component,
                'message': request.message
            }
            
            # Create feature array for model
            feature_array = np.array([[
                request.level,  # Should be encoded
                request.component,  # Should be encoded
                request.message_length,
                request.has_error_keywords,
                request.hour,
                request.day_of_week,
                request.is_weekend
            ]])
            
            if self.detector and self.detector.model:
                # Use ML model
                scaled_features = self.detector.scaler.transform(feature_array)
                anomaly_score = -self.detector.model.decision_function(scaled_features)[0]
                is_anomaly = anomaly_score > 0.6
            else:
                # Rule-based fallback
                anomaly_score = 0.0
                if request.level == "ERROR":
                    anomaly_score += 0.4
                if request.has_error_keywords:
                    anomaly_score += 0.3
                if request.message_length > 100:
                    anomaly_score += 0.3
                is_anomaly = anomaly_score > 0.6
            
            # Generate recommendation
            recommendation = "No action needed"
            if is_anomaly:
                if "database" in request.component.lower():
                    recommendation = "Check database connectivity and performance"
                elif "auth" in request.component.lower():
                    recommendation = "Investigate authentication issues"
                else:
                    recommendation = "Review system logs for patterns"
            
            return anomaly_detection_pb2.AnomalyResponse(
                anomaly_score=anomaly_score,
                is_anomaly=is_anomaly,
                original_log=f"[{request.timestamp}] [{request.level}] [{request.component}] {request.message}",
                recommendation=recommendation
            )
            
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Error processing request: {str(e)}")
            return anomaly_detection_pb2.AnomalyResponse()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    anomaly_detection_pb2_grpc.add_AnomalyDetectionServicer_to_server(
        AnomalyDetectionServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()