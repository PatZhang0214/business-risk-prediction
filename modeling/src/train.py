import pandas as pd
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    average_precision_score
)
import json

# ===============================
# Load v3 dataset from HF
# ===============================
local_path = hf_hub_download(
    repo_id="PatZhang0214/business-risk-prediction-dataset",
    filename="v3/yelp_fred_merged.parquet",
    repo_type="dataset"
)

df = pl.read_parquet(local_path)

# ===============================
# Train / Val / Test split (70/15/15)
# ===============================
df_train, df_test_val = train_test_split(
    df, test_size=0.3, random_state=67
)
df_test, df_validation = train_test_split(
    df_test_val, test_size=0.5, random_state=67
)

# ===============================
# Feature selection (v3)
# ===============================

# Core numeric + economic features
base_features = [
    "rating_x_reviews",
    "review_count",
    "num_categories",
    "years_in_business",
    "num_checkins",
    "has_checkin",
    "pcpi",
    "poverty_rate",
    "median_household_income",
    "unemployment_rate",
    "avg_weekly_wages",
    "latitude",
    "longitude",
]

# All one-hot encoded category features
category_features = [
    col for col in df.columns if col.startswith("cat_")
]

features = base_features + category_features
target = "is_open"

# ===============================
# Extract numpy arrays
# ===============================
X_train = df_train.select(features).to_numpy()
y_train = df_train.select(target).to_numpy().ravel()

X_val = df_validation.select(features).to_numpy()
y_val = df_validation.select(target).to_numpy().ravel()

X_test = df_test.select(features).to_numpy()
y_test = df_test.select(target).to_numpy().ravel()

# ===============================
# Handle class imbalance
# ===============================
# recompute based on training split
num_open = (y_train == 1).sum()
num_closed = (y_train == 0).sum()
scale_pos_weight = num_closed / num_open

# ===============================
# XGBoost model
# ===============================
model = XGBClassifier(
    n_estimators=1000,
    max_depth=7,
    learning_rate=0.05,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=30,   # ✅ moved here
    random_state=67
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)


print("\n🚀 Training XGBoost model...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)

print("✅ Training complete!")

# ===============================
# Evaluation
# ===============================
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n{'='*50}")
print("MODEL PERFORMANCE")
print(f"{'='*50}")
print(f"ROC-AUC Score:  {roc_auc:.4f}")
print(f"Accuracy:       {accuracy:.4f}")
print(f"Precision:      {precision:.4f}")
print(f"Recall:         {recall:.4f}")
print(f"F1-Score:       {f1:.4f}")
print(f"{'='*50}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print("                Predicted")
print("              Closed  Open")
print(f"Actual Closed   {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"       Open     {cm[1,0]:4d}  {cm[1,1]:4d}")

print("\n" + classification_report(y_test, y_pred, target_names=["Closed", "Open"]))

# ===============================
# Feature importance
# ===============================
importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n📈 Feature Importance:")
print(importance_df.head(25).to_string(index=False))

plt.figure(figsize=(10, 8))
plt.barh(
    importance_df.head(20)["feature"],
    importance_df.head(20)["importance"]
)
plt.xlabel("Importance")
plt.title("Top 20 XGBoost Feature Importances")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
print("\n💾 Feature importance plot saved to: feature_importance.png")

# ===============================
# Save artifacts
# ===============================
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/xgboost_model.pkl")

metrics = {
    "roc_auc": roc_auc,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "scale_pos_weight": scale_pos_weight,
    "n_features": len(features),
    "test_samples": int(len(y_test))
}

Path("reports").mkdir(exist_ok=True)
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n💾 Model saved to: models/xgboost_model.pkl")
print("📊 Metrics saved to: reports/metrics.json")

# ===============================
# Feature stats (numeric only)
# ===============================
numeric_features = [
    f for f in base_features if f in df.columns
]
print("\n📊 Feature Statistics:")
print(df.select(numeric_features).describe())

y_val_pred_proba = model.predict_proba(X_val)[:, 1]
y_test_pred_proba = model.predict_proba(X_test)[:, 1]

val_auc = roc_auc_score(y_val, y_val_pred_proba)
test_auc = roc_auc_score(y_test, y_test_pred_proba)

pr_auc = average_precision_score(y_test, y_test_pred_proba)

print(f"PR-AUC Score: {pr_auc:.4f}")
print(f"Validation ROC-AUC: {val_auc:.4f}")
print(f"Test ROC-AUC:       {test_auc:.4f}")
print(f"Difference:         {abs(val_auc - test_auc):.4f}")

# At the end of train.py, after saving the model

print("\n" + "="*60)
print("Running evaluation on test set...")
print("="*60)

# Save feature names for evaluate.py
with open("models/features.txt", "w") as f:
    f.write("\n".join(features))

print("\nTo re-run evaluation later:")
print(f"python evaluate.py --model models/xgboost_model.pkl --features '{' '.join(features)}'")
