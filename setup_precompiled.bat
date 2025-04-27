@echo off

:: Check if venv exists, if not create it
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment
call venv\Scripts\activate

:: Upgrade pip to ensure it's the latest version
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

:: Install dependencies in pip with progress bar, prioritize precompiled wheels
echo Installing dependencies...
pip install --prefer-binary --progress-bar=on -r requirements.txt

:: Register the virtual environment as a Jupyter kernel
echo Registering Jupyter kernel...
python -m ipykernel install --user --name=venv --display-name "Python (venv)"

echo Setup complete! Virtual environment is ready.
