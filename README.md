# 🚨 Log Anomaly Detection System

A high-performance, real-time log anomaly detection system built with **Go** and **Python**.  
This system monitors log files, detects anomalies using machine learning, and provides a live web dashboard for visualization.

---

## 📋 Project Overview

This project implements a comprehensive log analysis tool that:

- Processes logs in real-time using Go's concurrency model  
- Detects anomalies using ML models trained in Python  
- Provides live monitoring through a web dashboard  
- Generates performance benchmarks for comparative analysis  

---

## 🏗️ System Architecture

Log Sources<br>
↓<br>
Go Processing Engine (Concurrent Parsing & Feature Extraction)<br>
↓<br>
Anomaly Detection (Rule-based + ML Model)<br>
↓<br>
Web Dashboard (Real-time Visualization)<br>
↓<br>
Benchmarking & Reporting

---

## ✨ Features

### 🔍 Core Capabilities
- Real-time log monitoring with automatic file watching  
- Dual detection modes: Rule-based and ML-based anomaly detection  
- ONNX model integration for cross-platform ML inference  
- Concurrent processing with configurable worker pools  
- Automatic log generation for testing and demonstration  

### 📊 Monitoring & Analytics
- Live web dashboard with real-time metrics  
- Performance benchmarking with detailed reports  
- Anomaly history tracking with data persistence  
- Memory and goroutine monitoring  

### 🛠️ Production Features
- Log rotation and automatic backup  
- Configurable alert thresholds  
- Graceful shutdown with report generation  
- Error recovery and resilience  

---

## 🚀 Quick Start

### Prerequisites
- Go 1.21+  
- Python 3.8+  
- Git  

### Installation

Clone the repository:
```bash
git clone https://github.com/LEE-JOSH-ALLEN/ANOMALY-DETECTION-SYSTEM.git
cd log-anomaly-detector

```
Run the setup script
```bash
# Windows
setup.bat

# Linux/Mac
chmod +x setup.sh
./setup.sh
```
Start the system:
```bash
go run main.go
```
Access the dashboard: <br>
👉 Open http://localhost:8080 in your browser.

---

## 🎯 Usage Examples

### Basic Monitoring
```bash
# Start with default settings
go run main.go

# Monitor a specific log file
go run main.go -logfile /var/log/myapp.log
```
### Test Mode (Higher Anomaly Rate)
```bash
# Enable test mode for demonstration
go run main.go -testmode true
```
### Performance Benchmarking
```bash
# Run with detailed benchmarking
go run main.go -benchmark true
```

---

## ⚙️ Configuration

The system can be configured through the Config struct in main.go:
```go
config := &Config{
    Workers:             10,           // Number of concurrent workers
    Port:               "8080",        // Web dashboard port
    LogFile:            "test_logs.log", // Log file to monitor
    EnableRealTime:     true,          // Enable real-time file watching
    EnableProduction:   true,          // Enable production features
    EnableBenchmark:    true,          // Enable performance benchmarking
    EnableLogGenerator: true,          // Enable automatic log generation
    TestMode:           false,         // Higher anomaly rate for demos
    LogGeneratorInterval: 500 * time.Millisecond, // Log generation frequency
}
```

---

## 📁 Project Structure
```perl
log-anomaly-detector/
├── main.go                   # Main Go application (1000+ lines)
├── train_model.py            # Python ML model training
├── go.mod                    # Go dependencies
├── setup.bat                 # Windows setup script
├── setup.sh                  # Linux/Mac setup script
├── add_test_log.bat          # Manual log testing (Windows)
├── add_test_log.sh           # Manual log testing (Linux/Mac)
├── logs/                     # Log directory (auto-created)
├── backups/                  # Backup directory (auto-created)
├── anomaly_history.json      # Anomaly records (auto-generated)
└── performance_metrics.json  # Performance data (auto-generated)
```

---

## 🔧 Key Components

### Go Modules

- **Log Processor:** Concurrent log parsing and feature extraction
- **Real-time Monitor:** File system watching with fsnotify
- **Web Dashboard:** HTTP server with live metrics
- **Benchmark System:** Performance tracking and reporting
- **Production System:** Log rotation and data persistence

### Python Components

- **ML Model Training:** Isolation Forest algorithm
- **Feature Engineering:** Log parsing and normalization
- **ONNX Export:** Cross-platform model serialization

---

## 📊 Performance Metrics

Typical performance on standard hardware:

- **Processing Rate:** 1,000–2,000 logs/second
- **Memory Usage:** 50–100 MB (Go component)
- **Detection Accuracy:** 90–95% (ML vs rule-based)
- **Concurrent Workers:** Configurable (default: 10)

---

## 🧪 Testing

### Automatic Testing

The system includes an automatic log generator that creates realistic log entries:

- Normal logs: 85% (INFO/DEBUG level, successful operations)
- Warning logs: 10% (WARN level, performance issues)
- Error logs: 5% (ERROR level, system failures)

### Manual Testing
```bash
# Add a test log entry manually
./add_test_log.sh

# Or on Windows
add_test_log.bat
```

---

## 📈 Benchmark Results

The system generates comprehensive reports on shutdown (Ctrl+C):
```yaml
📊 ========== BENCHMARK RESULTS ==========
Processing Rate:      1,250 logs/second
Memory Usage:         48.2 MB
Total Logs Processed: 45,230
Anomalies Detected:   2,261 (5.0% rate)
Detection Accuracy:   92.3% (ML) vs 88.1% (Rule-based)
System Uptime:        36 hours
False Positive Rate:  4.2%
==========================================
```

---

## 🎓 Academic Context

This project was developed as part of a programming languages comparative analysis, demonstrating:

### Language Strengths

- **Go:** Concurrent processing, performance, production readiness
- **Python:** Machine learning, rapid prototyping, data science ecosystem

### Comparative Findings

- **Development Speed:** Python faster for ML, Go more robust for systems
- **Performance:** Go superior for concurrent data processing
- **Ecosystem:** Python better for ML, Go better for systems programming

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

---

## 🚨 Emergency Stop

To immediately stop the system:
```bash
# Press Ctrl+C for graceful shutdown
# Or use task manager to kill the process
```
**⚠️ Note:** This system is designed for educational and demonstration purposes.
For production use, additional security, authentication, and monitoring features should be implemented.

---

Built with ❤️ using Go and Python
