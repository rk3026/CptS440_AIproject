@echo off

:: Check if venv exists, if not create it
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment
call venv\Scripts\activate

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

:: Install dependencies in pip with progress bar
echo Installing dependencies...
pip install --progress-bar=on -r requirements.txt

:: Register the virtual environment as a Jupyter kernel
echo Registering Jupyter kernel...
python -m ipykernel install --user --name=venv --display-name "Python (venv)"

echo Setup complete! Virtual environment is ready.