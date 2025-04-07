#!/bin/bash

# Define directories
BASE_DIR="/Users/dawsontrotman/Documents/GitHub/PyFitSeq"
ENV_SPECIFIC_DIR="${BASE_DIR}/farah_data/outputs/env_specific"
PYFITSEQ_PATH="${BASE_DIR}/pyfitseq.py"

# Check if pyfitseq.py exists
if [ ! -f "$PYFITSEQ_PATH" ]; then
    echo "Error: PyFitSeq script not found at $PYFITSEQ_PATH"
    echo "Please locate pyfitseq.py and provide the correct path:"
    read -p "Path to pyfitseq.py: " PYFITSEQ_PATH
    
    if [ ! -f "$PYFITSEQ_PATH" ]; then
        echo "Error: File not found at $PYFITSEQ_PATH"
        echo "Cannot continue."
        exit 1
    fi
fi

echo "Using PyFitSeq script at: $PYFITSEQ_PATH"

# Fix all individual run scripts
cd "$ENV_SPECIFIC_DIR"
for script in run_pyfitseq_*.sh; do
    if [ -f "$script" ]; then
        echo "Updating $script..."
        # Replace the old PyFitSeq path with the correct one
        sed -i.bak "s|${BASE_DIR}/pyfitseq.py|${PYFITSEQ_PATH}|g" "$script"
        # Make sure it's executable
        chmod +x "$script"
    fi
done

# Update the master script too
if [ -f "run_all_pyfitseq.sh" ]; then
    echo "Updating run_all_pyfitseq.sh..."
    chmod +x "run_all_pyfitseq.sh"
fi

echo "All scripts have been updated."
echo "You can now run the analyses with:"
echo "cd $ENV_SPECIFIC_DIR"
echo "./run_all_pyfitseq.sh"
