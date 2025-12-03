from pathlib import Path
import polars as pl
from huggingface_hub import hf_hub_download


def load_dataset(local_path: str, hf_repo_id: str, hf_filename: str) -> pl.DataFrame:
    """
    Load dataset from local path if it exists, otherwise download from Hugging Face.
    
    Args:
        local_path: Path to local parquet file
        hf_repo_id: Hugging Face dataset repository ID (e.g., "username/dataset-name")
        hf_filename: Name of the file in the HF repository
    
    Returns:
        Polars DataFrame containing the dataset
    """
    local_file = Path(local_path)
    
    # Check if file exists locally
    if local_file.exists():
        print(f"📂 Loading dataset from local path: {local_path}")
        return pl.read_parquet(local_path)
    
    # File doesn't exist locally, download from Hugging Face
    print(f"🤗 Local file not found. Downloading from Hugging Face: {hf_repo_id}")
    
    try:
        # Create parent directory if it doesn't exist
        local_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Download file from Hugging Face Hub
        downloaded_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=hf_filename,
            repo_type="dataset",
            local_dir=str(local_file.parent),
            local_dir_use_symlinks=False
        )
        
        print(f"✅ Successfully downloaded to: {downloaded_path}")
        
        # Load and return the dataset
        return pl.read_parquet(downloaded_path)
        
    except Exception as e:
        print(f"❌ Error downloading from Hugging Face: {e}")
        print(f"Please manually download from: https://huggingface.co/datasets/{hf_repo_id}")
        raise