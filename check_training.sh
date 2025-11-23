#!/bin/bash
# Check training status

echo "========================================"
echo "Training Status Check"
echo "========================================"
echo ""

# Check if PID file exists
if [ -f training.pid ]; then
    PID=$(cat training.pid)
    
    # Check if process is running
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Training is RUNNING"
        echo "  Process ID: $PID"
        
        # Show CPU and memory usage
        echo ""
        echo "Resource usage:"
        ps -p $PID -o pid,pcpu,pmem,etime,cmd
        
    else
        echo "✗ Training process has STOPPED"
        echo "  Last PID: $PID"
    fi
else
    echo "✗ No training.pid file found"
    echo "  Training may not have been started with start_training.sh"
fi

echo ""
echo "----------------------------------------"
echo "Recent log files:"
ls -lht training_*.log 2>/dev/null | head -5

echo ""
echo "To view the latest log:"
LATEST_LOG=$(ls -t training_*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "  tail -f $LATEST_LOG"
    echo ""
    echo "Last 10 lines of $LATEST_LOG:"
    echo "----------------------------------------"
    tail -10 "$LATEST_LOG"
fi
