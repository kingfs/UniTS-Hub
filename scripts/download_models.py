import argparse
import os

from huggingface_hub import snapshot_download


MODEL_REPOS = {
    "timesfm": "google/timesfm-2.5-200m-pytorch",
    "chronos": "amazon/chronos-2",
    "kronos": "NeoQuasar/Kronos-base",
}

TOKENIZER_REPOS = {
    "kronos_tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
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


def download_models(model_name: str | None = None) -> None:
    if model_name:
        if model_name in MODEL_REPOS:
            _download(model_name, MODEL_REPOS[model_name])
            if model_name == "kronos":
                _download("kronos_tokenizer", TOKENIZER_REPOS["kronos_tokenizer"])
            return
        if model_name in TOKENIZER_REPOS:
            _download(model_name, TOKENIZER_REPOS[model_name])
            return
        raise ValueError(f"Unsupported model [{model_name}]")

    for name, repo_id in MODEL_REPOS.items():
        _download(name, repo_id)
    for name, repo_id in TOKENIZER_REPOS.items():
        _download(name, repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model weights for UniTS-Hub")
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted({*MODEL_REPOS.keys(), *TOKENIZER_REPOS.keys()}),
        help="Specific model or tokenizer to download",
    )
    args = parser.parse_args()
    download_models(args.model)
