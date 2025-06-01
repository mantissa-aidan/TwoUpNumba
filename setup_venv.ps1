# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Ensure pip is installed and working
python -m ensurepip --upgrade
python -m pip install --upgrade pip setuptools wheel

# Install requirements
python -m pip install -r requirements.txt

Write-Host "Virtual environment setup complete! You can now run 'python train_agent.py'" 