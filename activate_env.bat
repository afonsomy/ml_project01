@echo off
echo Activating ML/AI virtual environment...
call venv\Scripts\activate.bat
echo.
echo Virtual environment activated!
echo Python version:
python --version
echo.
echo To start Jupyter Lab, run: jupyter lab
echo To deactivate, run: deactivate
echo.
