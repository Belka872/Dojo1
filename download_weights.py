from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "facebook/nllb-200-distilled-600M"
WEIGHTS_DIR = Path("base_model/nllb-200-distilled-600M")


def main() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=WEIGHTS_DIR,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "onnx/*", "flax_model*"],
    )
    print(f"Downloaded {MODEL_ID} to {WEIGHTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
