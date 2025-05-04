#!/bin/bash


# Navigate to the Backend directory
cd Backend

# Activate the virtual environment
source venv/bin/activate

# Start the backend using uvicorn
# uvicorn main:app --reload
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8001