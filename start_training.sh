#!/bin/bash
# Start training in background with nohup

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="training_${TIMESTAMP}.log"

echo "Starting glaucoma detection pipeline in background..."
echo "Log file: $LOG_FILE"
echo ""

# Run pipeline with nohup
nohup ./run_pipeline.sh > "$LOG_FILE" 2>&1 &

# Save the process ID
PID=$!
echo $PID > training.pid

echo "✓ Training started successfully!"
echo "  Process ID: $PID"
echo "  Log file: $LOG_FILE"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check if still running:"
echo "  ./check_training.sh"
echo ""
echo "To stop training:"
echo "  kill $PID"
echo "  or: kill \$(cat training.pid)"
