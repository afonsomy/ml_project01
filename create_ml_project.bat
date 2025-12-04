@echo off
REM ============================================================================
REM  CREATE NEW ML/AI PROJECT
REM  Portable script - copy to any location and run!
REM  Usage: create_ml_project.bat [project_name]
REM ============================================================================
setlocal enabledelayedexpansion

echo.
echo  ╔═══════════════════════════════════════════════════════════════════════╗
echo  ║         CREATE NEW ML/AI PROJECT                                      ║
echo  ╚═══════════════════════════════════════════════════════════════════════╝
echo.

REM Get project name from argument or prompt
set "PROJECT_NAME=%~1"
if "%PROJECT_NAME%"=="" (
    set /p PROJECT_NAME="Enter project name: "
)

if "%PROJECT_NAME%"=="" (
    echo [ERROR] Project name is required!
    pause
    exit /b 1
)

REM Check if directory already exists
if exist "%PROJECT_NAME%" (
    echo [ERROR] Directory '%PROJECT_NAME%' already exists!
    pause
    exit /b 1
)

echo [INFO] Creating project: %PROJECT_NAME%
echo.

REM Create project directory and enter it
mkdir "%PROJECT_NAME%"
cd "%PROJECT_NAME%"

REM ============================================================================
REM  Create Folder Structure
REM ============================================================================
echo [1/6] Creating folder structure...

REM Data folders
mkdir "data\raw"
mkdir "data\processed"
mkdir "data\external"
mkdir "data\interim"

REM Notebooks
mkdir "notebooks\exploratory"
mkdir "notebooks\learning"
mkdir "notebooks\experiments"

REM Source code
mkdir "src\data"
mkdir "src\features"
mkdir "src\models"
mkdir "src\visualization"
mkdir "src\utils"
mkdir "src\pipelines"

REM Models
mkdir "models\saved"
mkdir "models\checkpoints"
mkdir "models\pretrained"

REM Reports
mkdir "reports\figures"
mkdir "reports\metrics"

REM Others
mkdir "config"
mkdir "tests\unit"
mkdir "tests\integration"
mkdir "docs"
mkdir "logs"
mkdir "scripts"

echo    [OK] Folders created!

REM ============================================================================
REM  Create Python package files
REM ============================================================================
echo [2/6] Creating Python packages...

echo. > "src\__init__.py"
echo. > "src\data\__init__.py"
echo. > "src\features\__init__.py"
echo. > "src\models\__init__.py"
echo. > "src\visualization\__init__.py"
echo. > "src\utils\__init__.py"
echo. > "src\pipelines\__init__.py"
echo. > "tests\__init__.py"
echo. > "tests\unit\__init__.py"
echo. > "tests\integration\__init__.py"

REM Create .gitkeep files
echo. > "data\raw\.gitkeep"
echo. > "data\processed\.gitkeep"
echo. > "data\external\.gitkeep"
echo. > "data\interim\.gitkeep"
echo. > "models\saved\.gitkeep"
echo. > "models\checkpoints\.gitkeep"
echo. > "models\pretrained\.gitkeep"
echo. > "logs\.gitkeep"

echo    [OK] Package files created!

REM ============================================================================
REM  Create requirements.txt
REM ============================================================================
echo [3/6] Creating requirements.txt...

(
echo # ==============================================
echo # Python ML/AI Requirements
echo # ==============================================
echo.
echo # --- Core Scientific Computing ---
echo numpy
echo pandas
echo scipy
echo.
echo # --- Data Visualization ---
echo matplotlib
echo seaborn
echo plotly
echo.
echo # --- Machine Learning ---
echo scikit-learn
echo xgboost
echo lightgbm
echo.
echo # --- Deep Learning ---
echo tensorflow
echo # torch  # Uncomment for PyTorch
echo # torchvision
echo.
echo # --- Natural Language Processing ---
echo nltk
echo # transformers  # Hugging Face
echo # spacy
echo.
echo # --- Computer Vision ---
echo opencv-python
echo pillow
echo.
echo # --- Jupyter Notebooks ---
echo jupyter
echo jupyterlab
echo ipykernel
echo ipywidgets
echo.
echo # --- Data Loading ---
echo openpyxl
echo xlrd
echo python-dotenv
echo.
echo # --- Experiment Tracking ---
echo mlflow
echo # wandb
echo.
echo # --- Hyperparameter Tuning ---
echo optuna
echo.
echo # --- Feature Engineering ---
echo category_encoders
echo imbalanced-learn
echo.
echo # --- Utilities ---
echo tqdm
echo requests
echo pyyaml
echo click
echo.
echo # --- Code Quality ---
echo pytest
echo black
echo flake8
echo isort
) > "requirements.txt"

echo    [OK] requirements.txt created!

REM ============================================================================
REM  Create configuration files
REM ============================================================================
echo [4/6] Creating configuration files...

REM .gitignore
(
echo # Python
echo __pycache__/
echo *.py[cod]
echo *$py.class
echo *.so
echo .Python
echo build/
echo dist/
echo *.egg-info/
echo *.egg
echo.
echo # Virtual Environment
echo venv/
echo ENV/
echo env/
echo.
echo # Jupyter
echo .ipynb_checkpoints/
echo.
echo # IDE
echo .idea/
echo .vscode/
echo *.swp
echo.
echo # Data ^(large files^)
echo data/raw/*
echo data/external/*
echo !data/*/.gitkeep
echo.
echo # Models ^(large files^)
echo models/saved/*
echo models/checkpoints/*
echo models/pretrained/*
echo !models/*/.gitkeep
echo.
echo # Logs
echo logs/*
echo !logs/.gitkeep
echo *.log
echo.
echo # Environment
echo .env
echo .env.local
echo.
echo # MLflow
echo mlruns/
echo.
echo # OS
echo .DS_Store
echo Thumbs.db
) > ".gitignore"

REM config.yaml
(
echo # Project Configuration
echo project:
echo   name: "%PROJECT_NAME%"
echo   version: "1.0.0"
echo.
echo data:
echo   raw: "data/raw"
echo   processed: "data/processed"
echo   external: "data/external"
echo   interim: "data/interim"
echo.
echo models:
echo   saved: "models/saved"
echo   checkpoints: "models/checkpoints"
echo   pretrained: "models/pretrained"
echo.
echo training:
echo   batch_size: 32
echo   epochs: 100
echo   learning_rate: 0.001
echo   seed: 42
echo.
echo logging:
echo   level: "INFO"
echo   path: "logs"
) > "config\config.yaml"

REM .env.example
(
echo # Environment Variables
echo # Copy to .env and fill in your values
echo.
echo # OPENAI_API_KEY=your_key
echo # HUGGINGFACE_TOKEN=your_token
echo # MLFLOW_TRACKING_URI=your_uri
) > ".env.example"

echo    [OK] Configuration files created!

REM ============================================================================
REM  Create utility batch files
REM ============================================================================
echo [5/6] Creating utility scripts...

REM activate_env.bat
(
echo @echo off
echo echo Activating virtual environment...
echo call venv\Scripts\activate.bat
echo echo.
echo echo Virtual environment activated!
echo python --version
echo echo.
) > "activate_env.bat"

REM start_jupyter.bat
(
echo @echo off
echo call venv\Scripts\activate.bat
echo echo Starting Jupyter Lab...
echo jupyter lab
) > "start_jupyter.bat"

REM README.md
(
echo # %PROJECT_NAME%
echo.
echo ## Setup
echo.
echo 1. Run `setup_project.bat` to create virtual environment and install dependencies
echo 2. Activate environment: `call venv\Scripts\activate.bat`
echo 3. Start Jupyter: `jupyter lab`
echo.
echo ## Project Structure
echo.
echo ```
echo %PROJECT_NAME%/
echo ├── config/              # Configuration files
echo ├── data/
echo │   ├── raw/             # Original data
echo │   ├── processed/       # Cleaned data
echo │   ├── external/        # External data
echo │   └── interim/         # Intermediate data
echo ├── docs/                # Documentation
echo ├── logs/                # Log files
echo ├── models/
echo │   ├── saved/           # Trained models
echo │   ├── checkpoints/     # Training checkpoints
echo │   └── pretrained/      # Pre-trained models
echo ├── notebooks/
echo │   ├── exploratory/     # EDA notebooks
echo │   ├── learning/        # Learning notebooks
echo │   └── experiments/     # Experiment notebooks
echo ├── reports/
echo │   ├── figures/         # Graphics
echo │   └── metrics/         # Metrics
echo ├── scripts/             # Utility scripts
echo ├── src/
echo │   ├── data/            # Data processing
echo │   ├── features/        # Feature engineering
echo │   ├── models/          # Model definitions
echo │   ├── pipelines/       # ML pipelines
echo │   ├── utils/           # Utilities
echo │   └── visualization/   # Visualization
echo └── tests/               # Tests
echo ```
) > "README.md"

echo    [OK] Utility scripts created!

REM ============================================================================
REM  Create virtual environment and install dependencies
REM ============================================================================
echo [6/6] Creating virtual environment and installing dependencies...
echo    This may take several minutes...
echo.

python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment!
    echo Make sure Python is installed and in your PATH.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [WARNING] Some packages may have failed. Check output above.
) else (
    echo.
    echo [OK] All dependencies installed!
)

echo.
echo  ╔═══════════════════════════════════════════════════════════════════════╗
echo  ║                PROJECT CREATED SUCCESSFULLY!                          ║
echo  ╚═══════════════════════════════════════════════════════════════════════╝
echo.
echo  Project Location: %cd%
echo.
echo  Quick Start:
echo    cd %PROJECT_NAME%
echo    call venv\Scripts\activate.bat
echo    jupyter lab
echo.
echo  Happy coding!
echo.

pause

