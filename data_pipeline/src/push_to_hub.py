"""
Push processed dataset to Hugging Face Hub with versioning
Usage: python push_dataset.py --version v2
"""
from datasets import Dataset
import polars as pl
from huggingface_hub import HfApi
import argparse

def push_to_hub_versioned(output_path, repo_id, version=None, commit_message=None):
    """
    Push dataset with version tagging
    
    Args:
        output_path: Path to processed parquet file
        repo_id: HuggingFace repo ID (e.g., "PatZhang0214/business-risk-prediction-dataset")
        version: Version tag (e.g., "v2", "10k-entries")
        commit_message: Custom commit message
    """
    print(f"📥 Loading dataset from {output_path}...")
    df = pl.read_parquet(output_path)
    print(f"   Dataset shape: {df.shape}")
    
    print(f"🔄 Converting to Hugging Face Dataset format...")
    hf_ds = Dataset.from_pandas(df.to_pandas())
    
    # Create commit message
    if commit_message is None:
        commit_message = f"Add dataset with {len(df):,} entries"
        if version:
            commit_message += f" ({version})"
    
    print(f"📤 Pushing to {repo_id}...")
    print(f"   Commit message: {commit_message}")
    
    # Push to hub
    hf_ds.push_to_hub(
        repo_id,
        commit_message=commit_message
    )
    
    print(f"✅ Successfully pushed to https://huggingface.co/datasets/{repo_id}")
    
    # Create version tag if specified
    if version:
        print(f"🏷️  Creating version tag: {version}")
        api = HfApi()
        try:
            api.create_tag(
                repo_id=repo_id,
                tag=version,
                repo_type="dataset",
                tag_message=f"Release {version} with {len(df):,} entries"
            )
            print(f"✅ Tag '{version}' created")
            print(f"   Access this version: load_dataset('{repo_id}', revision='{version}')")
        except Exception as e:
            print(f"⚠️  Warning: Could not create tag: {e}")
    
    print("\n📊 Version Summary:")
    print(f"   Latest (main): {len(df):,} entries")
    if version:
        print(f"   Tagged as: {version}")
    print(f"   Previous version (v1/older commit): 967 entries (still accessible)")

def main():
    parser = argparse.ArgumentParser(description="Push dataset to HuggingFace with versioning")
    parser.add_argument(
        "--output-path",
        default="outputs/processed.parquet",
        help="Path to processed parquet file"
    )
    parser.add_argument(
        "--repo-id",
        default="PatZhang0214/business-risk-prediction-dataset",
        help="HuggingFace repository ID"
    )
    parser.add_argument(
        "--version",
        default="v2-10k",
        help="Version tag (e.g., v2, v2-10k, 2024-12-26)"
    )
    parser.add_argument(
        "--message",
        help="Custom commit message"
    )
    
    args = parser.parse_args()
    
    push_to_hub_versioned(
        output_path=args.output_path,
        repo_id=args.repo_id,
        version=args.version,
        commit_message=args.message
    )

if __name__ == "__main__":
    main()