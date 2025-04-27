#!/bin/bash

# Check if venv exists, if not create it
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip setuptools wheel

# Install dependencies with progress bar
echo "Installing dependencies..."
pip install --progress-bar=on -r ../requirements.txt

# Register the virtual environment as a Jupyter kernel
echo "Registering Jupyter kernel..."
python3 -m ipykernel install --user --name=venv --display-name "Python (venv)"

echo "Setup complete! Virtual environment is ready."
