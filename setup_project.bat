@echo off
REM ============================================================================
REM  ML/AI Project Setup Script
REM  Creates folder structure and installs all Python dependencies
REM ============================================================================
setlocal enabledelayedexpansion

echo.
echo  ╔═══════════════════════════════════════════════════════════════════════╗
echo  ║            ML/AI PROJECT SETUP SCRIPT                                 ║
echo  ║            Creating project structure and installing dependencies     ║
echo  ╚═══════════════════════════════════════════════════════════════════════╝
echo.

REM Set project name (can be customized)
set "PROJECT_NAME=ml_project"
if not "%~1"=="" set "PROJECT_NAME=%~1"

echo [INFO] Project Name: %PROJECT_NAME%
echo.

REM ============================================================================
REM  STEP 1: Create Folder Structure
REM ============================================================================
echo [STEP 1/5] Creating folder structure...
echo.

REM --- Data folders ---
echo    Creating data folders...
if not exist "data" mkdir "data"
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "data\external" mkdir "data\external"
if not exist "data\interim" mkdir "data\interim"

REM --- Notebooks folders ---
echo    Creating notebooks folders...
if not exist "notebooks" mkdir "notebooks"
if not exist "notebooks\exploratory" mkdir "notebooks\exploratory"
if not exist "notebooks\learning" mkdir "notebooks\learning"
if not exist "notebooks\experiments" mkdir "notebooks\experiments"

REM --- Source code folders ---
echo    Creating source code folders...
if not exist "src" mkdir "src"
if not exist "src\data" mkdir "src\data"
if not exist "src\features" mkdir "src\features"
if not exist "src\models" mkdir "src\models"
if not exist "src\visualization" mkdir "src\visualization"
if not exist "src\utils" mkdir "src\utils"
if not exist "src\pipelines" mkdir "src\pipelines"

REM --- Models folders ---
echo    Creating models folders...
if not exist "models" mkdir "models"
if not exist "models\saved" mkdir "models\saved"
if not exist "models\checkpoints" mkdir "models\checkpoints"
if not exist "models\pretrained" mkdir "models\pretrained"

REM --- Reports and outputs ---
echo    Creating reports folders...
if not exist "reports" mkdir "reports"
if not exist "reports\figures" mkdir "reports\figures"
if not exist "reports\metrics" mkdir "reports\metrics"

REM --- Configuration ---
echo    Creating config folders...
if not exist "config" mkdir "config"

REM --- Tests ---
echo    Creating tests folders...
if not exist "tests" mkdir "tests"
if not exist "tests\unit" mkdir "tests\unit"
if not exist "tests\integration" mkdir "tests\integration"

REM --- Documentation ---
echo    Creating docs folders...
if not exist "docs" mkdir "docs"

REM --- Logs ---
echo    Creating logs folder...
if not exist "logs" mkdir "logs"

REM --- Scripts ---
echo    Creating scripts folder...
if not exist "scripts" mkdir "scripts"

echo    [OK] Folder structure created!
echo.

REM ============================================================================
REM  STEP 2: Create __init__.py files for Python packages
REM ============================================================================
echo [STEP 2/5] Creating Python package files...

if not exist "src\__init__.py" echo. > "src\__init__.py"
if not exist "src\data\__init__.py" echo. > "src\data\__init__.py"
if not exist "src\features\__init__.py" echo. > "src\features\__init__.py"
if not exist "src\models\__init__.py" echo. > "src\models\__init__.py"
if not exist "src\visualization\__init__.py" echo. > "src\visualization\__init__.py"
if not exist "src\utils\__init__.py" echo. > "src\utils\__init__.py"
if not exist "src\pipelines\__init__.py" echo. > "src\pipelines\__init__.py"
if not exist "tests\__init__.py" echo. > "tests\__init__.py"
if not exist "tests\unit\__init__.py" echo. > "tests\unit\__init__.py"
if not exist "tests\integration\__init__.py" echo. > "tests\integration\__init__.py"

echo    [OK] Python package files created!
echo.

REM ============================================================================
REM  STEP 3: Create requirements.txt if it doesn't exist
REM ============================================================================
echo [STEP 3/5] Checking requirements.txt...

if not exist "requirements.txt" (
    echo    Creating requirements.txt...
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
        echo # torch  # Uncomment for PyTorch ^(install separately: pip install torch^)
        echo # torchvision  # Uncomment for PyTorch vision
        echo.
        echo # --- Natural Language Processing ---
        echo nltk
        echo # transformers  # Uncomment for Hugging Face Transformers
        echo # spacy  # Uncomment for spaCy
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
        echo # --- Data Loading and Processing ---
        echo openpyxl
        echo xlrd
        echo python-dotenv
        echo.
        echo # --- Model Tracking and Experiments ---
        echo mlflow
        echo # wandb  # Uncomment for Weights ^& Biases
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
) else (
    echo    [OK] requirements.txt already exists!
)
echo.

REM ============================================================================
REM  STEP 4: Create Virtual Environment
REM ============================================================================
echo [STEP 4/5] Setting up Python virtual environment...

if not exist "venv" (
    echo    Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo    [ERROR] Failed to create virtual environment!
        echo    Make sure Python is installed and in your PATH.
        pause
        exit /b 1
    )
    echo    [OK] Virtual environment created!
) else (
    echo    [OK] Virtual environment already exists!
)
echo.

REM ============================================================================
REM  STEP 5: Install Dependencies
REM ============================================================================
echo [STEP 5/5] Installing Python dependencies...
echo    This may take several minutes...
echo.

call venv\Scripts\activate.bat

REM Upgrade pip first
echo    Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install dependencies
echo    Installing packages from requirements.txt...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo    [WARNING] Some packages may have failed to install.
    echo    Check the output above for errors.
) else (
    echo.
    echo    [OK] All dependencies installed successfully!
)

REM ============================================================================
REM  STEP 6: Create additional configuration files
REM ============================================================================
echo.
echo [BONUS] Creating additional configuration files...

REM Create .gitignore if it doesn't exist
if not exist ".gitignore" (
    (
        echo # Python
        echo __pycache__/
        echo *.py[cod]
        echo *$py.class
        echo *.so
        echo .Python
        echo build/
        echo develop-eggs/
        echo dist/
        echo downloads/
        echo eggs/
        echo .eggs/
        echo lib/
        echo lib64/
        echo parts/
        echo sdist/
        echo var/
        echo wheels/
        echo *.egg-info/
        echo .installed.cfg
        echo *.egg
        echo.
        echo # Virtual Environment
        echo venv/
        echo ENV/
        echo env/
        echo.
        echo # Jupyter Notebook
        echo .ipynb_checkpoints/
        echo.
        echo # IDE
        echo .idea/
        echo .vscode/
        echo *.swp
        echo *.swo
        echo.
        echo # Data files ^(large files^)
        echo data/raw/*
        echo data/external/*
        echo !data/raw/.gitkeep
        echo !data/external/.gitkeep
        echo.
        echo # Model files ^(large files^)
        echo models/saved/*
        echo models/checkpoints/*
        echo models/pretrained/*
        echo !models/saved/.gitkeep
        echo !models/checkpoints/.gitkeep
        echo !models/pretrained/.gitkeep
        echo.
        echo # Logs
        echo logs/
        echo *.log
        echo.
        echo # Environment variables
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
    echo    [OK] .gitignore created!
)

REM Create .gitkeep files to preserve empty folders in git
echo. > "data\raw\.gitkeep"
echo. > "data\external\.gitkeep"
echo. > "data\processed\.gitkeep"
echo. > "data\interim\.gitkeep"
echo. > "models\saved\.gitkeep"
echo. > "models\checkpoints\.gitkeep"
echo. > "models\pretrained\.gitkeep"
echo. > "logs\.gitkeep"
echo    [OK] .gitkeep files created!

REM Create config.yaml if it doesn't exist
if not exist "config\config.yaml" (
    (
        echo # Project Configuration
        echo project:
        echo   name: "%PROJECT_NAME%"
        echo   version: "1.0.0"
        echo.
        echo # Data paths
        echo data:
        echo   raw: "data/raw"
        echo   processed: "data/processed"
        echo   external: "data/external"
        echo   interim: "data/interim"
        echo.
        echo # Model paths
        echo models:
        echo   saved: "models/saved"
        echo   checkpoints: "models/checkpoints"
        echo   pretrained: "models/pretrained"
        echo.
        echo # Training parameters
        echo training:
        echo   batch_size: 32
        echo   epochs: 100
        echo   learning_rate: 0.001
        echo   seed: 42
        echo.
        echo # Logging
        echo logging:
        echo   level: "INFO"
        echo   path: "logs"
    ) > "config\config.yaml"
    echo    [OK] config.yaml created!
)

REM Create .env.example
if not exist ".env.example" (
    (
        echo # Environment Variables Example
        echo # Copy this file to .env and fill in your values
        echo.
        echo # API Keys
        echo # OPENAI_API_KEY=your_api_key_here
        echo # HUGGINGFACE_TOKEN=your_token_here
        echo.
        echo # Database
        echo # DATABASE_URL=your_database_url
        echo.
        echo # MLflow
        echo # MLFLOW_TRACKING_URI=your_mlflow_uri
    ) > ".env.example"
    echo    [OK] .env.example created!
)

echo.
echo  ╔═══════════════════════════════════════════════════════════════════════╗
echo  ║                     SETUP COMPLETE!                                   ║
echo  ╚═══════════════════════════════════════════════════════════════════════╝
echo.
echo  Project Structure Created:
echo.
echo    %PROJECT_NAME%/
echo    ├── config/              # Configuration files
echo    ├── data/
echo    │   ├── raw/             # Original, immutable data
echo    │   ├── processed/       # Cleaned, transformed data
echo    │   ├── external/        # Data from external sources
echo    │   └── interim/         # Intermediate data
echo    ├── docs/                # Documentation
echo    ├── logs/                # Log files
echo    ├── models/
echo    │   ├── saved/           # Trained models
echo    │   ├── checkpoints/     # Training checkpoints
echo    │   └── pretrained/      # Pre-trained models
echo    ├── notebooks/
echo    │   ├── exploratory/     # EDA notebooks
echo    │   ├── learning/        # Learning notebooks
echo    │   └── experiments/     # Experiment notebooks
echo    ├── reports/
echo    │   ├── figures/         # Generated graphics
echo    │   └── metrics/         # Model metrics
echo    ├── scripts/             # Utility scripts
echo    ├── src/
echo    │   ├── data/            # Data processing scripts
echo    │   ├── features/        # Feature engineering
echo    │   ├── models/          # Model definitions
echo    │   ├── pipelines/       # ML pipelines
echo    │   ├── utils/           # Utility functions
echo    │   └── visualization/   # Visualization scripts
echo    ├── tests/
echo    │   ├── unit/            # Unit tests
echo    │   └── integration/     # Integration tests
echo    └── venv/                # Virtual environment
echo.
echo  Next Steps:
echo    1. Activate the environment: call venv\Scripts\activate.bat
echo    2. Start Jupyter Lab: jupyter lab
echo    3. Copy .env.example to .env and add your API keys
echo.
echo  Happy coding!
echo.

pause

