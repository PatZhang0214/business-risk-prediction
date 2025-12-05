import polars as pl
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from scale_features import scale_numeric_features, transform_features

# Paths
DATA_DIR = Path(__file__).parent.parent / "dataset"
RAW_DATA_PATH = DATA_DIR / "yelp_fred_merged.parquet"
TRAIN_DATA_PATH = DATA_DIR / "train.parquet"
TEST_DATA_PATH = DATA_DIR / "test.parquet"
SCALER_PATH = Path(__file__).parent.parent / "artifacts" / "scaler.pkl"

# Feature columns used for training
FEATURE_COLS = [
    "rating",
    "pcpi",
    "poverty_rate",
    "median_household_income",
    "unemployment_rate",
    "avg_weekly_wages",
]

TARGET_COL = "is_open"


def split_and_save(
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> None:
    """
    Split the raw dataset into train/test sets and save as parquet files.
    
    Args:
        test_size: Fraction of data for testing (default 0.2)
        random_state: Random seed for reproducibility
        stratify: Whether to stratify split by target
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}")
    
    df = pl.read_parquet(RAW_DATA_PATH)
    
    # Get indices for splitting
    indices = np.arange(len(df))
    y = df.select(TARGET_COL).to_numpy().ravel()
    
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )
    
    # Split and save
    train_df = df[train_idx.tolist()]
    test_df = df[test_idx.tolist()]
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.write_parquet(TRAIN_DATA_PATH)
    test_df.write_parquet(TEST_DATA_PATH)


def load_train_data(
    feature_cols: list[str] = None,
    target_col: str = TARGET_COL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load training features and target, scaling features using StandardScaler.
    
    Returns:
        Tuple of (X_train, y_train) numpy arrays with scaled features
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
        
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Train data not found at {TRAIN_DATA_PATH}. "
        )
    
    df = pl.read_parquet(TRAIN_DATA_PATH)
    # Fit scaler on training data and scale features
    X = scale_numeric_features(df, feature_cols, scaler_path=str(SCALER_PATH))
    y = df.select(target_col).to_numpy().ravel()
    return X, y


def load_test_data(
    feature_cols: list[str] = None,
    target_col: str = TARGET_COL,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load test features and target, scaling features using the scaler fitted on training data.
    
    Returns:
        Tuple of (X_test, y_test) numpy arrays with scaled features
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
        
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test data not found at {TEST_DATA_PATH}. "
        )
    
    df = pl.read_parquet(TEST_DATA_PATH)
    # Transform test data using the scaler fitted on training data
    X = transform_features(df, feature_cols, scaler_path=str(SCALER_PATH))
    y = df.select(target_col).to_numpy().ravel()
    return X, y


def get_dataset_info() -> dict:
    """Get summary information about the train/test datasets."""
    X_train, y_train = load_train_data()
    X_test, y_test = load_test_data()
    
    return {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": len(FEATURE_COLS),
        "feature_names": FEATURE_COLS,
        "target_name": TARGET_COL,
        "train_class_balance": float((y_train == 1).mean()),
        "test_class_balance": float((y_test == 1).mean()),
    }


if __name__ == "__main__":
    print("Splitting dataset into train/test files...")
    split_and_save(test_size=0.2, random_state=42, stratify=True)
    
    print("\n Dataset Info:")
    info = get_dataset_info()
    print(f"   Train samples: {info['n_train']}")
    print(f"   Test samples: {info['n_test']}")
    print(f"   Features: {info['feature_names']}")
    print(f"   Train class balance (% open): {info['train_class_balance']:.1%}")
    print(f"   Test class balance (% open): {info['test_class_balance']:.1%}")
