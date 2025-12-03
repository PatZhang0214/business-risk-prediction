# Business Risk Prediction - Data Pipeline

This data pipeline preprocesses merged Yelp and FRED economic data to prepare it for training a business closure prediction model.

## 📁 Project Structure

```
data_pipeline/
├─ configs/
│  └─ process.yaml          # Pipeline configuration
├─ src/
│  ├─ validate_schema.py    # Pydantic schema validation
│  ├─ preprocess.py         # Main pipeline script
│  ├─ scale_features.py     # Feature scaling with StandardScaler
│  └─ push_to_hub.py        # Upload to Hugging Face Hub
├─ dataset/
│  └─ yelp_fred_merged.parquet  # Raw merged dataset
├─ outputs/                 # Generated: processed data
├─ artifacts/               # Generated: scaler.pkl
├─ requirements.txt
└─ README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Ensure your raw data is located at:
```
dataset/yelp_fred_merged.parquet
```

The dataset should contain these columns:
- `business_id` (str)
- `rating` (float)
- `pcpi` (float) - Per Capita Personal Income
- `poverty_rate` (float)
- `median_household_income` (float)
- `unemployment_rate` (float)
- `avg_weekly_wages` (float)
- `is_open` (int) - Target variable (1=open, 0=closed)

### 3. Run the Pipeline

From the **root** of the repository:

```bash
python src/preprocess.py
```

### 4. Verify Outputs

After successful execution, you should see:

```
outputs/
└─ processed.parquet        # Cleaned, validated, and scaled data

artifacts/
└─ scaler.pkl              # Fitted StandardScaler (needed for inference)
```

## ⚙️ Configuration

Edit `configs/process.yaml` to customize the pipeline:

```yaml
data:
  raw_path: "dataset/yelp_fred_merged.parquet"
  output_path: "outputs/processed.parquet"
  scaler_path: "artifacts/scaler.pkl"

preprocess:
  drop_missing_threshold: 0.3
  normalize: true

features:
  numeric_cols:
    - rating
    - pcpi
    - poverty_rate
    - median_household_income
    - unemployment_rate
    - avg_weekly_wages
```

## 🔧 What the Pipeline Does

1. **Loads raw data** from the specified parquet file
2. **Drops null values** in numeric feature columns
3. **Validates schema** using Pydantic to ensure data quality
4. **Scales features** using StandardScaler (mean=0, std=1)
5. **Saves processed data** and the fitted scaler for model training

## 📊 Pipeline Outputs

### `outputs/processed.parquet`
- Cleaned and scaled dataset ready for model training
- All numeric features standardized
- No missing values in feature columns

### `artifacts/scaler.pkl`
- Fitted StandardScaler object
- **Critical for production**: Must be used to transform new data during inference
- Load with: `scaler = joblib.load('artifacts/scaler.pkl')`

## 🐛 Troubleshooting

### Error: `FileNotFoundError: dataset/yelp_fred_merged.parquet`

**Solution**: Ensure you're running from the repository root and the dataset exists:
```bash
ls dataset/yelp_fred_merged.parquet
python src/preprocess.py
```

### Error: `No such file or directory: 'artifacts/scaler.pkl'`

**Solution**: The `artifacts/` directory is auto-created. If you still see this error, manually create it:
```bash
mkdir -p artifacts outputs
python src/preprocess.py
```

### Validation errors

If Pydantic validation fails, check your data for:
- Missing required columns
- Incorrect data types
- Out-of-range values

## 🚢 Optional: Push to Hugging Face Hub

To share your processed dataset:

```python
from src.push_to_hub import push_to_hub

push_to_hub(
    output_path="outputs/processed.parquet",
    repo_id="your-username/yelp-business-risk"
)
```

Or use the command line:
```bash
python -c "from src.push_to_hub import push_to_hub; push_to_hub('outputs/processed.parquet', 'your-username/yelp-business-risk')"
```

## 📦 Using DVC (Data Version Control)

Track your data pipeline with DVC:

```bash
# Initialize DVC (first time only)
dvc init

# Track raw data
dvc add dataset/yelp_fred_merged.parquet

# Create DVC pipeline
dvc stage add -n preprocess \
  -d dataset/yelp_fred_merged.parquet \
  -d src/preprocess.py \
  -d src/validate_schema.py \
  -d src/scale_features.py \
  -o outputs/processed.parquet \
  -o artifacts/scaler.pkl \
  python src/preprocess.py

# Run pipeline
dvc repro

# Commit changes
git add dvc.yaml dvc.lock .gitignore
git commit -m "Add data preprocessing pipeline"
```

## 🔄 Next Steps

After running this pipeline:

1. **Split data**: Create train/validation/test sets
2. **Train model**: Use `outputs/processed.parquet` for model training
3. **Evaluate**: Test model performance
4. **Deploy**: Use `artifacts/scaler.pkl` in your inference pipeline

## 📝 Dependencies

- `polars` - Fast DataFrame library
- `pydantic` - Data validation
- `hydra-core` - Configuration management
- `scikit-learn` - Feature scaling
- `joblib` - Artifact serialization
- `datasets` & `huggingface_hub` - Optional HF integration
- `dvc` - Data version control

## 🤝 Contributing

When making changes to the pipeline:

1. Update `configs/process.yaml` for configuration changes
2. Add validation rules in `validate_schema.py`
3. Test on a small sample first
4. Document any new dependencies in `requirements.txt`

---

**Questions?** Open an issue or contact the maintainer.