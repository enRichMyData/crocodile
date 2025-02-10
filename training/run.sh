#!/bin/bash

# Run the onboard_data script
echo "Starting data onboarding..."
python3 onboard_data.py

echo "Data onboarding completed successfully."
echo "Starting computation..."

# Run the start_computation script
python3 start_computation.py

echo "Computation ended successfully."
