"""Upload staged IndicMix AI model files to Hugging Face Hub."""

import argparse
import os
import sys

try:
    from huggingface_hub import HfApi, login
except ImportError:
    print("Error: huggingface_hub is not installed. Install with `pip install huggingface_hub`.")
    sys.exit(1)


def upload_model(
    repo_id: str,
    token: str = None,
    staging_dir: str = None,
    private: bool = False
):
    if not staging_dir:
        staging_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "hf_hub"))

    if not os.path.exists(staging_dir):
        print(f"Error: Staging directory not found at {staging_dir}.")
        print("Please run `python scripts/prepare_hf_model.py` first.")
        sys.exit(1)

    effective_token = token or os.environ.get("HF_TOKEN")
    
    # Check .env file if token not yet set
    env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if not effective_token and os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    effective_token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if effective_token:
        login(token=effective_token)

    api = HfApi(token=effective_token)

    try:
        user_info = api.whoami()
        username = user_info.get("name") or user_info.get("username", "Authenticated User")
        print(f"Authenticated as Hugging Face user: {username}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("\nPlease provide a valid Hugging Face write token:")
        print("1. Set HF_TOKEN environment variable: $env:HF_TOKEN='hf_...'")
        print("2. Or run: huggingface-cli login")
        print("3. Or pass: python scripts/upload_to_hf.py --token hf_... --repo-id <username>/<model-name>")
        sys.exit(1)

    print(f"\nTarget Hugging Face Model Repository: {repo_id}")
    print(f"Source Directory: {staging_dir}")

    # Ensure repository exists
    print(f"Creating / verifying model repository '{repo_id}' on Hugging Face...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True
    )

    print(f"Uploading files to https://huggingface.co/{repo_id}...")
    api.upload_folder(
        folder_path=staging_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial release of IndicMix AI: Code-mixed Indic NLP models & INT8 ONNX exports"
    )

    print("\nUpload successfully completed!")
    print(f"Model URL: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Upload IndicMix AI to Hugging Face Hub")
    parser.add_argument("--repo-id", default="Adityapande403/IndicMix-AI", help="Target HF repository ID (e.g. username/repo-name)")
    parser.add_argument("--token", default=None, help="Hugging Face access token with write permission")
    parser.add_argument("--staging-dir", default=None, help="Directory containing staged model files")
    parser.add_argument("--private", action="store_true", help="Create a private model repository")
    args = parser.parse_args()

    upload_model(
        repo_id=args.repo_id,
        token=args.token,
        staging_dir=args.staging_dir,
        private=args.private
    )


if __name__ == "__main__":
    main()
