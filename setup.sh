#!/bin/sh
# 🚀 Setting up Enhanced Log Anomaly Detection System...

echo "🚀 Setting up Enhanced Log Anomaly Detection System..."

# Create directories
[ ! -d "logs" ] && mkdir logs
[ ! -d "backups" ] && mkdir backups

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install scikit-learn pandas numpy joblib onnx onnxruntime skl2onnx

# Install Go dependencies
echo "📦 Installing Go dependencies..."
go mod tidy

# Train ML model and export ONNX
echo "🤖 Training ML model..."
python train_model.py

# Create manual test script (adds a log entry)
# Use current date/time
timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$timestamp] [ERROR] [database] Manual test error - this should trigger an anomaly" > test_logs.log

echo
echo "✅ Enhanced System Setup Completed!"
echo
echo "🎯 System Features:"
echo "   - Real-time log monitoring"
echo "   - Production-ready features"
echo "   - Performance benchmarking"
echo "   - Comparative analysis"
echo "   - Web dashboard with metrics"
echo "   - Automatic log generation (500ms intervals)"
echo
echo "🚀 To run the system:"
echo "   go run main.go"
echo
echo "🌐 Web dashboard: http://localhost:8080"
echo
echo "📊 Testing Options:"
echo "  1. Automatic: System generates logs automatically"
echo "  2. Manual: Run ./add_test_log.sh to add specific test logs"
echo "  3. Real-time: Watch anomalies appear in dashboard"
echo "  4. Analysis: Press Ctrl+C to generate final reports"
echo

# Pause equivalent
read -r _  # Wait for user to press Enter
