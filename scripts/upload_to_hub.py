"""
Upload the fine-tuned model to HuggingFace Hub for deployment.

Usage:
    python scripts/upload_to_hub.py --model_dir model --hub_name your-username/radiology-summarizer

You will be prompted for your HuggingFace token, or set HF_TOKEN env variable.
Get your token at: https://huggingface.co/settings/tokens
"""

import argparse
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from huggingface_hub import HfApi


def upload_model(model_dir: str, hub_name: str, private: bool = False):
    api = HfApi()
    token = os.getenv("HF_TOKEN")

    print(f"Creating repository: {hub_name} (private={private})")
    api.create_repo(hub_name, private=private, exist_ok=True, token=token)

    print(f"Loading model from '{model_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    print(f"Uploading tokenizer to {hub_name}...")
    tokenizer.push_to_hub(hub_name, token=token)

    print(f"Uploading model to {hub_name} (this may take a few minutes)...")
    model.push_to_hub(hub_name, token=token)

    print(f"\nModel uploaded successfully!")
    print(f"View at: https://huggingface.co/{hub_name}")
    print(f"\nTo use in deployment, set environment variable:")
    print(f"  HF_MODEL_NAME={hub_name}")


def main():
    parser = argparse.ArgumentParser(description="Upload fine-tuned model to HuggingFace Hub")
    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument(
        "--hub_name",
        type=str,
        required=True,
        help="HuggingFace Hub repo name, e.g. your-username/radiology-summarizer",
    )
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    args = parser.parse_args()

    upload_model(args.model_dir, args.hub_name, args.private)


if __name__ == "__main__":
    main()
