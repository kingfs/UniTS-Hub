import os
import argparse
from huggingface_hub import snapshot_download

# Define models to be built-in
MODELS = {
    "timesfm": "google/timesfm-2.5-200m-pytorch",
    "chronos": "amazon/chronos-2"
}

BASE_DIR = os.getenv("MODELS_DIR", "/app/models")

def download_models(model_name: str = None):
    # Ensure base directory exists
    os.makedirs(BASE_DIR, exist_ok=True)
    
    models_to_download = {model_name: MODELS[model_name]} if model_name else MODELS
    
    for name, repo_id in models_to_download.items():
        print(f"⬇️ Downloading {name} from {repo_id}...")
        save_path = os.path.join(BASE_DIR, name)
        
        # snapshot_download for full directory
        snapshot_download(
            repo_id=repo_id,
            local_dir=save_path,
            local_dir_use_symlinks=False,  # Important for Docker consistency
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"]  # Skip non-PyTorch weights
        )
        print(f"✅ {name} saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model weights for UniTS-Hub")
    parser.add_argument("--model", type=str, choices=["timesfm", "chronos"], help="Specific model to download")
    args = parser.parse_args()
    
    download_models(args.model)
