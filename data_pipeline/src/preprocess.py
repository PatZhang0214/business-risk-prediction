import hydra
from omegaconf import DictConfig
import polars as pl
from pathlib import Path

from validate_schema import validate_records
from scale_features import scale_numeric_features
from data_loader import load_dataset


@hydra.main(config_path="../configs", config_name="process", version_base=None)
def main(cfg: DictConfig):

    print("📥 Loading raw dataset...")
    # Use the new loader that handles HF download
    df = load_dataset(
        local_path=cfg.data.raw_path,
        hf_repo_id=cfg.data.hf_repo_id,
        hf_filename=cfg.data.hf_filename
    )
    
    print(f"📊 Dataset shape: {df.shape}")

    # Drop rows with too many missing values
    df = df.drop_nulls(subset=cfg.features.numeric_cols)
    print(f"📊 After dropping nulls: {df.shape}")

    print("✅ Validating schema...")
    validate_records(df)

    # Scale numeric features
    if cfg.preprocess.normalize:
        print("📊 Scaling numeric features...")
        scaled = scale_numeric_features(
            df, 
            cfg.features.numeric_cols,
            cfg.data.scaler_path
        )

        # Replace original cols with scaled versions
        for i, col in enumerate(cfg.features.numeric_cols):
            df = df.with_columns(pl.Series(col, scaled[:, i]))

    # Create output directory if it doesn't exist
    Path(cfg.data.output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save processed file
    print(f"💾 Saving processed dataset to {cfg.data.output_path}")
    df.write_parquet(cfg.data.output_path)

    print("🎉 Preprocessing pipeline complete!")


if __name__ == "__main__":
    main()