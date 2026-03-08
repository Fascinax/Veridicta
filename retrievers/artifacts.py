"""Download Veridicta index artifacts from HuggingFace Hub when running on Streamlit Cloud.

The heavy files (FAISS index, BM25 index, chunks.jsonl) are gitignored and must be
fetched at startup if not present locally.

HuggingFace repo: https://huggingface.co/datasets/Fascinax/veridicta-index
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

HF_REPO_ID = "Fascinax/veridicta-index"
HF_REPO_TYPE = "dataset"

# Map: local relative path -> filename in HF Hub repo
_ARTIFACTS = {
    "data/index/veridicta.faiss": "index/veridicta.faiss",
    "data/index/bm25_corpus.pkl": "index/bm25_corpus.pkl",
    "data/processed/chunks.jsonl": "processed/chunks.jsonl",
}


def _hf_token() -> str | None:
    """Resolve HuggingFace token from env or Streamlit secrets."""
    token = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token:
        return token
    try:
        import streamlit as st  # noqa: PLC0415
        return st.secrets.get("HF_API_TOKEN") or st.secrets.get("HUGGINGFACE_TOKEN")
    except Exception:
        return None


def ensure_artifacts(root: str | Path = ".") -> None:
    """Download any missing artifacts from HuggingFace Hub.

    Safe to call on every startup — skips files that already exist.
    Raises RuntimeError if a download fails and the file is still missing.
    """
    root = Path(root)
    missing = [
        (local, remote)
        for local, remote in _ARTIFACTS.items()
        if not (root / local).exists()
    ]
    if not missing:
        logger.info("All artifacts present — skipping download.")
        return

    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub not installed. Run: pip install huggingface_hub"
        ) from exc

    token = _hf_token()
    if not token:
        logger.warning(
            "HF_API_TOKEN not set. Attempting anonymous download from %s "
            "(will fail if repo is private).",
            HF_REPO_ID,
        )

    for local, remote in missing:
        dest = root / local
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s from HuggingFace Hub ...", remote)
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=remote,
                repo_type=HF_REPO_TYPE,
                token=token,
                local_dir=str(root),
            )
            logger.info("Saved to %s", dest)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download artifact '{remote}' from {HF_REPO_ID}: {exc}"
            ) from exc

    logger.info("All artifacts ready.")


def upload_artifacts(root: str | Path = ".", token: str | None = None) -> None:
    """Upload local artifacts to HuggingFace Hub (run once locally to seed the repo).

    Usage:
        python -c "from retrievers.artifacts import upload_artifacts; upload_artifacts()"
    Or:
        python -m retrievers.artifacts --upload
    """
    from huggingface_hub import HfApi  # noqa: PLC0415

    root = Path(root)
    token = token or _hf_token()
    api = HfApi()

    # Create repo if it does not exist
    try:
        api.create_repo(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            exist_ok=True,
            token=token,
        )
        logger.info("HF repo ready: %s", HF_REPO_ID)
    except Exception as exc:
        raise RuntimeError(f"Cannot create HF repo: {exc}") from exc

    for local, remote in _ARTIFACTS.items():
        src = root / local
        if not src.exists():
            logger.warning("Skipping missing file: %s", src)
            continue
        size_mb = src.stat().st_size / 1024 / 1024
        logger.info("Uploading %s (%.1f MB) -> %s ...", src, size_mb, remote)
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=remote,
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            token=token,
        )
        logger.info("  Done.")

    logger.info("All artifacts uploaded to %s", HF_REPO_ID)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Manage Veridicta HuggingFace artifacts.")
    parser.add_argument("--upload", action="store_true", help="Upload local artifacts to HF Hub.")
    parser.add_argument("--download", action="store_true", help="Download missing artifacts from HF Hub.")
    parser.add_argument("--token", default=None, help="HuggingFace API token (overrides HF_API_TOKEN env).")
    args = parser.parse_args()

    if args.upload:
        upload_artifacts(token=args.token)
    elif args.download:
        if args.token:
            os.environ["HF_API_TOKEN"] = args.token
        ensure_artifacts()
    else:
        parser.print_help()
