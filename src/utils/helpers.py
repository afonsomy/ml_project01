"""
Utility functions for the ML project.
"""

import os
import yaml
import pickle
import numpy as np
import pandas as pd
from datetime import datetime


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_model(model, filename: str, path: str = "models/saved"):
    """Save a model to disk."""
    filepath = os.path.join(path, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filename: str, path: str = "models/saved"):
    """Load a model from disk."""
    filepath = os.path.join(path, filename)
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_timestamp() -> str:
    """Get current timestamp as string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
