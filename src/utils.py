"""
For Common functions and utilities
"""

import os
import sys
import json
import logging
import math
import joblib
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.exception import CustomException


def save_object(file_path, obj):

    try:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, file_path)
        logger.info(f"Saved object to {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):

    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact not found: {file_path}")
        
        return joblib.load(file_path)
        
    except Exception as e:
        raise CustomException(e, sys)


def save_json(file_path, data):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {file_path}")


def rul_score(y_true, y_pred):
    d = y_pred - y_true
    scores = np.where(d < 0, np.expm1(-d / 13.0), np.expm1(d / 10.0))
    
    return float(np.sum(scores))


def compute_all_metrics(y_true, y_pred, prefix):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    phm = rul_score(y_true, y_pred)

    tag = f"{prefix}_" if prefix else ""
    
    return {
        f"{tag}rmse": round(rmse, 4),
        f"{tag}mae": round(mae, 4),
        f"{tag}r2": round(r2, 4),
        f"{tag}phm_score": round(phm, 4),
        f"{tag}n_samples": int(len(y_true)),
    }

