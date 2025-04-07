#!/bin/bash

# Define directories
FLUCTUATING_DIR="/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/fluctuating_env"
ENV_SPECIFIC_DIR="/Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/outputs/env_specific"

# What is the correct path to pyfitseq.py?
echo "Enter the correct path to pyfitseq.py:"
read -p "> " CORRECT_PATH

# Verify that the path exists
if [ ! -f "$CORRECT_PATH" ]; then
    echo "Error: File not found at $CORRECT_PATH"
    echo "Please check the path and try again."
    exit 1
fi

echo "Using PyFitSeq script at: $CORRECT_PATH"

# Fix scripts in fluctuating_env directory
echo "Fixing scripts in fluctuating_env directory..."
cd "$FLUCTUATING_DIR"
for script in run_pyfitseq_*.sh; do
    if [ -f "$script" ]; then
        echo "Updating $script..."
        # Replace the incorrect path with the correct one
        sed -i.bak "s|/Users/dawsontrotman/Documents/GitHub/PyFitSeq/pyfitseq.py|${CORRECT_PATH}|g" "$script"
        # Make sure it's executable
        chmod +x "$script"
    fi
done

# Fix scripts in env_specific directory
echo "Fixing scripts in env_specific directory..."
cd "$ENV_SPECIFIC_DIR"
for script in run_pyfitseq_*.sh; do
    if [ -f "$script" ]; then
        echo "Updating $script..."
        # Replace the incorrect path with the correct one
        sed -i.bak "s|/Users/dawsontrotman/Documents/GitHub/PyFitSeq/PyFitSeq/pyfitseq.py|${CORRECT_PATH}|g" "$script"
        # Make sure it's executable
        chmod +x "$script"
    fi
done

echo "All scripts have been updated to use: $CORRECT_PATH"
echo ""
echo "Next steps:"
echo "1. Run the PyFitSeq analysis for fluctuating environments:"
echo "   cd $FLUCTUATING_DIR"
echo "   ./run_all_pyfitseq.sh"
echo ""
echo "2. After that completes, run the visualization script:"
echo "   python3 /Users/dawsontrotman/Documents/GitHub/PyFitSeq/farah_data/code/fluctuating-env-fitness-plot.py"
