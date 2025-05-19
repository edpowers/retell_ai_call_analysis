#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project directory
cd "$SCRIPT_DIR"

# Create log directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/logs"
# Create Hive-like partitioned directory structure based on current date
YEAR=$(date +%Y)
MONTH=$(date +%m)
DAY=$(date +%d)
LOG_DIR="$SCRIPT_DIR/logs/year=$YEAR/month=$MONTH/day=$DAY"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%H%M%S).log"

# Check if we're within the 8 AM to 8 PM EST time window
current_hour=$(TZ="America/New_York" date +%H)
if [ "$current_hour" -lt 8 ] || [ "$current_hour" -ge 20 ]; then
    echo "Outside of operating hours (8 AM - 8 PM EST). Exiting."
    exit 0
fi

# Activate the virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Virtual environment not found. Pleaxse create one first."
    exit 1
fi

# Set database path
export DB_PATH="$SCRIPT_DIR/data/call_analysis.db"

# Run with logging
echo "Running script at $(date)" | tee -a "$LOG_FILE"
PYTHONPATH="$SCRIPT_DIR" python retell_ai_call_analysis/run.py 2>&1 | tee -a "$LOG_FILE"

# Check exit status
if [ $? -ne 0 ]; then
    echo "Script failed with error code $?" | tee -a "$LOG_FILE"
    # Optionally send notification about failure
fi

# Deactivate the virtual environment
deactivate

echo "Completed at $(date)"
