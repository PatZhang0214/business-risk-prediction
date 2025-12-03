import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def scale_numeric_features(df, numeric_cols):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_cols].to_numpy())

    # Create artifacts directory if it doesn't exist
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    
    # Save scaler for later use
    joblib.dump(scaler, "artifacts/scaler.pkl")

    return scaled
