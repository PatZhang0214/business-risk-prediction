import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def scale_numeric_features(df, numeric_cols, scaler_path="artifacts/scaler.pkl"):
    """
    Scale numeric features using StandardScaler.
    
    Args:
        df: Polars DataFrame
        numeric_cols: List of column names to scale
        scaler_path: Path to save the fitted scaler
    
    Returns:
        Scaled numpy array
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_cols].to_numpy())

    # Create parent directory if it doesn't exist
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save scaler for later use
    joblib.dump(scaler, scaler_path)
    print(f"💾 Scaler saved to {scaler_path}")

    return scaled