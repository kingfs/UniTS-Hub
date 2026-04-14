import argparse
import os

from huggingface_hub import snapshot_download


MODEL_REPOS = {
    "timesfm": "google/timesfm-2.5-200m-pytorch",
    "chronos": "amazon/chronos-2",
    "kronos": "NeoQuasar/Kronos-base",
}

TOKENIZER_REPOS = {
    "kronos": "NeoQuasar/Kronos-Tokenizer-base",
}

BASE_DIR = os.getenv("MODELS_DIR", "/app/models")


def _download(name: str, repo_id: str) -> None:
    os.makedirs(BASE_DIR, exist_ok=True)
    save_path = os.path.join(BASE_DIR, name)
    print(f"Downloading {name} from {repo_id} to {save_path}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=save_path,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
    )
    print(f"Saved {name} to {save_path}")


def download_model_bundle(model_name: str) -> None:
    if model_name not in MODEL_REPOS:
        raise ValueError(f"Unsupported model [{model_name}]")
    _download(model_name, MODEL_REPOS[model_name])
    tokenizer_repo = TOKENIZER_REPOS.get(model_name)
    if tokenizer_repo:
        _download(f"{model_name}_tokenizer", tokenizer_repo)


def download_models(model_name: str | None = None) -> None:
    if model_name:
        download_model_bundle(model_name)
        return
    for name in MODEL_REPOS:
        download_model_bundle(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model weights for UniTS-Hub")
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(MODEL_REPOS.keys()),
        help="Specific model bundle to download",
    )
    args = parser.parse_args()
    download_models(args.model)
