# ML/AI Workspace

## Folder Structure

```
ml_workspace/
├── data/
│   ├── raw/           # Original, immutable data
│   ├── processed/     # Cleaned, transformed data
│   └── external/      # Data from external sources
├── src/
│   ├── data/          # Data loading and processing scripts
│   ├── features/      # Feature engineering scripts
│   ├── models/        # Model training and prediction
│   ├── visualization/ # Plotting and visualization
│   └── utils/         # Utility functions
├── notebooks/
│   ├── exploratory/   # EDA notebooks
│   └── experiments/   # Experiment notebooks
├── models/
│   ├── saved/         # Final trained models
│   └── checkpoints/   # Training checkpoints
├── reports/
│   └── figures/       # Generated graphics and figures
├── config/            # Configuration files
├── scripts/           # Automation scripts
├── tests/             # Unit tests
└── venv/              # Virtual environment
```

## Getting Started

1. Activate virtual environment:
   ```
   venv\Scripts\activate
   ```

2. Start Jupyter Lab:
   ```
   jupyter lab
   ```

3. Run a Python script:
   ```
   python src/your_script.py
   ```

## Installed Libraries

- **numpy, pandas, scipy** - Scientific computing
- **matplotlib, seaborn, plotly** - Visualization
- **scikit-learn, xgboost, lightgbm** - Machine Learning
- **tensorflow** - Deep Learning
- **opencv-python, pillow** - Computer Vision
- **nltk** - Natural Language Processing
- **jupyter, jupyterlab** - Interactive notebooks
- **mlflow** - Experiment tracking

