#!/bin/bash

# Run the onboard_data script
echo "Starting data onboarding..."
python3 onboard_data.py

# Check if the onboarding was successful
if [ $? -eq 0 ]; then
    echo "Data onboarding completed successfully."
    echo "Starting computation..."
    
    # Run the start_computation script
    python3 start_computation.py

    # Check if the computation started successfully
    if [ $? -eq 0 ]; then
        echo "Computation started successfully."
    else
        echo "Error: Failed to start computation."
        exit 1
    fi
else
    echo "Error: Data onboarding failed."
    exit 1
fi
