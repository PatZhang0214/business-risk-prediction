import hydra
from omegaconf import DictConfig
import polars as pl

from validate_schema import validate_records
from scale_features import scale_numeric_features

@hydra.main(config_path="../configs", config_name="process", version_base=None)
def main(cfg: DictConfig):

    print("📥 Loading raw dataset...")
    df = pl.read_parquet(cfg.data.raw_path)

    # Drop rows with too many missing values
    df = df.drop_nulls(subset=cfg.features.numeric_cols)

    print("✅ Validating schema...")
    validate_records(df)

    # Scale numeric features
    if cfg.preprocess.normalize:
        print("📊 Scaling numeric features...")
        scaled = scale_numeric_features(df, cfg.features.numeric_cols)

        # Replace original cols with scaled versions
        for i, col in enumerate(cfg.features.numeric_cols):
            df = df.with_columns(pl.Series(col, scaled[:, i]))

    # Save processed file
    print(f"💾 Saving processed dataset to {cfg.data.output_path}")
    df.write_parquet(cfg.data.output_path)

    print("🎉 Preprocessing pipeline complete!")

if __name__ == "__main__":
    main()
