#!/bin/bash
echo "Adding test log entry to test_logs.log..."

TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
LEVEL="ERROR"
COMPONENT="database"
MESSAGE="Manual test error - this should trigger an anomaly"

echo "[$TIMESTAMP] [$LEVEL] [$COMPONENT] $MESSAGE" >> test_logs.log

echo "✅ Test log entry added!"
echo ""
echo "Current directory: $(pwd)"
echo "Log file: test_logs.log"