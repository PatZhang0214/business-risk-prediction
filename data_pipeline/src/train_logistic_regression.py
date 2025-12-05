import argparse
import joblib
import shutil
from pathlib import Path
from datetime import datetime
from scipy.stats import loguniform

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score
)

from split_data import load_train_data, load_test_data, FEATURE_COLS

MODEL_DIR = Path(__file__).parent.parent / "models"

# --- Training ---

def get_param_distributions() -> list:
    """Parameter distributions for RandomizedSearchCV."""
    return [
        {
            "solver": ["lbfgs", "newton-cg", "sag"],
            "penalty": ["l2"],
            "C": loguniform(1e-4, 1e3),
            "class_weight": [None, "balanced"],
            "max_iter": [10000],
        },
        {
            "solver": ["saga"],
            "penalty": ["l1", "l2"],
            "C": loguniform(1e-4, 1e3),
            "class_weight": [None, "balanced"],
            "max_iter": [10000],
        },
        {
            "solver": ["liblinear"],
            "penalty": ["l1", "l2"],
            "C": loguniform(1e-4, 1e3),
            "class_weight": [None, "balanced"],
            "max_iter": [10000],
        },
    ]


def train(n_iter: int = 100, cv_folds: int = 5, random_state: int = 42):
    """
    Train logistic regression with hyperparameter tuning.
    
    Args:
        n_iter: Number of random search iterations
        cv_folds: Number of cross-validation folds
        random_state: Random seed
        
    Returns:
        Best trained pipeline
    """
    X_train, y_train = load_train_data()
    
    # Features are already scaled in load_train_data()
    clf = LogisticRegression(random_state=random_state)
    
    search = RandomizedSearchCV(
        clf,
        param_distributions=get_param_distributions(),
        n_iter=n_iter,
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
        n_jobs=-1,
        random_state=random_state,
        refit=True,
    )
    
    print(f"Training with {n_iter} iterations, {cv_folds}-fold CV...")
    search.fit(X_train, y_train)
    
    print(f"\nBest CV Score (Avg Precision): {search.best_score_:.4f}")
    print("Best Parameters:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    
    return search.best_estimator_, search.best_params_, search.best_score_


def save_model(model: LogisticRegression, best_params: dict, cv_score: float) -> Path:
    """Save trained model, removing old versions."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Remove old models
    for old_dir in MODEL_DIR.glob("logistic_regression_*"):
        if old_dir.is_dir():
            shutil.rmtree(old_dir)
    
    # Save new model
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_dir = MODEL_DIR / f"logistic_regression_{timestamp}"
    model_dir.mkdir()
    
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)
    
    # Save summary
    with open(model_dir / "summary.txt", "w") as f:
        f.write(f"Logistic Regression Model\n{'=' * 30}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"CV Score (Avg Precision): {cv_score:.4f}\n\n")
        f.write("Best Parameters:\n")
        for param, value in best_params.items():
            f.write(f"  {param}: {value}\n")
    
    print(f"\n💾 Model saved to: {model_dir}")
    return model_path


# --- Evaluation ---

def load_model() -> LogisticRegression:
    """Load the latest saved model."""
    model_dirs = list(MODEL_DIR.glob("logistic_regression_*"))
    if not model_dirs:
        raise FileNotFoundError(f"No model found in {MODEL_DIR}")
    
    latest = max(model_dirs, key=lambda p: p.name)
    return joblib.load(latest / "model.pkl")


def evaluate(model: LogisticRegression = None) -> dict:
    """
    Evaluate model on test set.
    
    Args:
        model: Pipeline to evaluate (loads saved model if None)
        
    Returns:
        Dictionary of metrics
    """
    if model is None:
        model = load_model()
    
    X_test, y_test = load_test_data()
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred)
    }
    
    print("\n📈 Test Set Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")

    return metrics


# --- Main ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/evaluate logistic regression")
    parser.add_argument("--eval", action="store_true", help="Only evaluate saved model")
    parser.add_argument("--n-iter", type=int, default=100, help="Random search iterations")
    args = parser.parse_args()
    
    if args.eval:
        evaluate()
    else:
        model, best_params, cv_score = train(n_iter=args.n_iter)
        save_model(model, best_params, cv_score)
        evaluate(model)
