"""
Evaluation script for Business Risk Prediction Model
Loads a trained model and generates comprehensive evaluation metrics and visualizations.

Usage:
    python evaluate.py --model models/xgboost_model.pkl --repo-id PatZhang0214/business-risk-prediction-dataset --filename v3/yelp_fred_merged.parquet
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
import joblib
from huggingface_hub import hf_hub_download
import polars as pl
from sklearn.model_selection import train_test_split

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def load_model(model_path):
    """Load a trained model from disk"""
    print(f"📦 Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully")
    return model


def load_test_data(repo_id, filename, features, target='is_open', test_size=0.3, val_size=0.5, random_state=67):
    """Load and prepare test data"""
    from huggingface_hub import hf_hub_download
    import polars as pl
    
    print(f"📥 Loading dataset from {repo_id}/{filename}...")
    
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset"
    )
    
    df = pl.read_parquet(local_path).to_pandas()
    
    print(f"Dataset shape: {df.shape}")
    
    # Split data (same way as training)
    df_train, df_test_val = train_test_split(df, test_size=test_size, random_state=random_state)
    df_test, df_validation = train_test_split(df_test_val, test_size=val_size, random_state=random_state)
    
    X_test = df_test[features].values
    y_test = df_test[target].values
    
    print(f"Test set shape: {X_test.shape}")
    print(f"Test set class distribution:")
    unique, counts = np.unique(y_test, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"  Class {val}: {count} ({count/len(y_test)*100:.1f}%)")
    
    return X_test, y_test, df_test


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all evaluation metrics"""
    metrics = {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'average_precision': average_precision_score(y_true, y_pred_proba)
    }
    
    # Per-class metrics
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # Specificity (True Negative Rate)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def print_metrics(metrics):
    """Pretty print metrics"""
    print(f"\n{'='*60}")
    print(f"{'MODEL PERFORMANCE METRICS':^60}")
    print(f"{'='*60}")
    print(f"ROC-AUC Score:        {metrics['roc_auc']:.4f}")
    print(f"Average Precision:    {metrics['average_precision']:.4f}")
    print(f"Accuracy:             {metrics['accuracy']:.4f}")
    print(f"Precision:            {metrics['precision']:.4f}")
    print(f"Recall (Sensitivity): {metrics['recall']:.4f}")
    print(f"Specificity:          {metrics['specificity']:.4f}")
    print(f"F1-Score:             {metrics['f1_score']:.4f}")
    print(f"{'='*60}")
    
    print(f"\n{'CONFUSION MATRIX':^60}")
    print(f"{'='*60}")
    print(f"                    Predicted")
    print(f"                 Closed    Open")
    print(f"Actual  Closed    {metrics['true_negatives']:5d}   {metrics['false_positives']:5d}")
    print(f"        Open      {metrics['false_negatives']:5d}   {metrics['true_positives']:5d}")
    print(f"{'='*60}\n")


def plot_confusion_matrix(y_true, y_pred, output_path='reports/confusion_matrix.png'):
    """Generate confusion matrix heatmap"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Closed', 'Open'],
                yticklabels=['Closed', 'Open'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Confusion matrix saved to: {output_path}")


def plot_roc_curve(y_true, y_pred_proba, output_path='reports/roc_curve.png'):
    """Generate ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 ROC curve saved to: {output_path}")


def plot_precision_recall_curve(y_true, y_pred_proba, output_path='reports/precision_recall_curve.png'):
    """Generate Precision-Recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Precision-Recall curve saved to: {output_path}")


def plot_feature_importance(model, features, output_path='reports/feature_importance.png', top_n=15):
    """Plot feature importance"""
    if not hasattr(model, 'feature_importances_'):
        print("⚠️  Model does not have feature_importances_ attribute, skipping...")
        return
    
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Take top N features
    plot_df = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(plot_df['feature'], plot_df['importance'], color='steelblue')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Feature importance plot saved to: {output_path}")
    
    # Print top features
    print(f"\n{'TOP FEATURES':^60}")
    print(f"{'='*60}")
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:30s} {row['importance']:.4f}")
    print(f"{'='*60}\n")


def plot_prediction_distribution(y_pred_proba, output_path='reports/prediction_distribution.png'):
    """Plot distribution of predicted probabilities"""
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    plt.xlabel('Predicted Probability (Business Stays Open)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Prediction distribution saved to: {output_path}")


def generate_classification_report(y_true, y_pred, output_path='reports/classification_report.txt'):
    """Generate and save detailed classification report"""
    report = classification_report(y_true, y_pred, 
                                   target_names=['Closed', 'Open'],
                                   digits=4)
    
    print(f"\n{'CLASSIFICATION REPORT':^60}")
    print(f"{'='*60}")
    print(report)
    
    with open(output_path, 'w') as f:
        f.write("BUSINESS RISK PREDICTION - CLASSIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(report)
    
    print(f"💾 Classification report saved to: {output_path}")


def save_metrics_json(metrics, features, output_path='reports/evaluation_metrics.json'):
    """Save metrics to JSON file"""
    output = {
        'metrics': metrics,
        'features_used': features,
        'num_features': len(features)
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 Metrics JSON saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained XGBoost model')
    parser.add_argument('--model', type=str, default='models/xgboost_model.pkl',
                       help='Path to trained model')
    parser.add_argument('--repo-id', type=str, 
                       default='PatZhang0214/business-risk-prediction-dataset',
                       help='HuggingFace repository ID')
    parser.add_argument('--filename', type=str,
                       default='v3/yelp_fred_merged.parquet',
                       help='Filename within the HuggingFace repo')
    parser.add_argument('--features', type=str, nargs='+', 
                       help='List of feature names')
    parser.add_argument('--features-file', type=str, 
                       help='Path to text file with feature names (one per line)')
    parser.add_argument('--output-dir', type=str, default='reports',
                       help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.model)
    
    # Get features
    if args.features:
        features = args.features
    elif args.features_file:
        print(f"📄 Loading features from {args.features_file}...")
        with open(args.features_file, 'r') as f:
            features = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(features)} features")
    else:
        # Try to infer from model
        print("⚠️  No features provided. Please specify with --features or --features-file flag")
        print("Example: --features rating pcpi poverty_rate unemployment_rate ...")
        print("Or: --features-file features.txt")
        return
    
    # Load test data
    X_test, y_test, df_test = load_test_data(args.repo_id, args.filename, features)
    
    # Generate predictions
    print("\n🔮 Generating predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    print("\n📊 Calculating metrics...")
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    # Print metrics
    print_metrics(metrics)
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    plot_confusion_matrix(y_test, y_pred, f'{args.output_dir}/confusion_matrix.png')
    plot_roc_curve(y_test, y_pred_proba, f'{args.output_dir}/roc_curve.png')
    plot_precision_recall_curve(y_test, y_pred_proba, f'{args.output_dir}/precision_recall_curve.png')
    plot_feature_importance(model, features, f'{args.output_dir}/feature_importance.png')
    plot_prediction_distribution(y_pred_proba, f'{args.output_dir}/prediction_distribution.png')
    
    # Generate reports
    print("\n📝 Generating reports...")
    generate_classification_report(y_test, y_pred, f'{args.output_dir}/classification_report.txt')
    save_metrics_json(metrics, features, f'{args.output_dir}/evaluation_metrics.json')
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
    print(f"All outputs saved to: {args.output_dir}/")
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - precision_recall_curve.png")
    print("  - feature_importance.png")
    print("  - prediction_distribution.png")
    print("  - classification_report.txt")
    print("  - evaluation_metrics.json")


if __name__ == "__main__":
    main()