"""Download Veridicta index artifacts from HuggingFace Hub when running on Streamlit Cloud.

The heavy files (FAISS index, BM25 index, chunks.jsonl) are gitignored and must be
fetched at startup if not present locally.

HuggingFace repo: https://huggingface.co/datasets/Fascinax/veridicta-index

BM25 artifacts are optional at download time because they can be rebuilt locally
from `chunks_map.jsonl` if only FAISS + chunks are present.
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
    "data/index/chunks_map.jsonl": "index/chunks_map.jsonl",
    "data/index/embedding_config.json": "index/embedding_config.json",
    "data/index/bm25s_index/data.csc.index.npy": "index/bm25s_index/data.csc.index.npy",
    "data/index/bm25s_index/indices.csc.index.npy": "index/bm25s_index/indices.csc.index.npy",
    "data/index/bm25s_index/indptr.csc.index.npy": "index/bm25s_index/indptr.csc.index.npy",
    "data/index/bm25s_index/params.index.json": "index/bm25s_index/params.index.json",
    "data/index/bm25s_index/vocab.index.json": "index/bm25s_index/vocab.index.json",
    "data/processed/chunks.jsonl": "processed/chunks.jsonl",
}

_OPTIONAL_ARTIFACTS = {
    "data/index/bm25s_index/data.csc.index.npy",
    "data/index/bm25s_index/indices.csc.index.npy",
    "data/index/bm25s_index/indptr.csc.index.npy",
    "data/index/bm25s_index/params.index.json",
    "data/index/bm25s_index/vocab.index.json",
}

# Map: local relative directory path -> HF Hub path prefix (for directory artifacts)
_DIR_ARTIFACTS: dict[str, str] = {
    "data/index/lancedb": "index/lancedb",
}

# Optional directory artifacts — download failure is non-fatal
_OPTIONAL_DIR_ARTIFACTS: set[str] = {"data/index/lancedb"}


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


def _dir_artifact_present(root: Path, local_dir: str) -> bool:
    """Return True if a directory artifact exists and contains at least one file."""
    p = root / local_dir
    return p.is_dir() and any(f for f in p.rglob("*") if f.is_file())


def _download_dir_artifact(
    root: Path,
    local_dir: str,
    remote_prefix: str,
    token: str | None,
) -> None:
    """Download all files under *remote_prefix* in the HF Hub repo into *local_dir*."""
    from huggingface_hub import hf_hub_download, list_repo_files  # noqa: PLC0415

    remote_files = [
        f
        for f in list_repo_files(HF_REPO_ID, repo_type=HF_REPO_TYPE, token=token)
        if f.startswith(remote_prefix + "/")
    ]
    if not remote_files:
        raise FileNotFoundError(
            f"No files found under '{remote_prefix}' in {HF_REPO_ID}. "
            "Upload the LanceDB index first: python -m retrievers.artifacts --upload"
        )
    for remote_file in remote_files:
        logger.info("Downloading %s from HuggingFace Hub ...", remote_file)
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=remote_file,
            repo_type=HF_REPO_TYPE,
            token=token,
            local_dir=str(root / "data"),
        )
    logger.info("Dir artifact '%s' ready (%d files).", local_dir, len(remote_files))


def ensure_artifacts(root: str | Path = ".") -> None:
    """Download any missing artifacts from HuggingFace Hub.

    Safe to call on every startup — skips files that already exist.
    Raises RuntimeError if a download fails and the file is still missing.
    """
    root = Path(root)
    missing_files = [
        (local, remote)
        for local, remote in _ARTIFACTS.items()
        if not (root / local).exists()
    ]
    missing_dirs = [
        (local_dir, remote_prefix)
        for local_dir, remote_prefix in _DIR_ARTIFACTS.items()
        if not _dir_artifact_present(root, local_dir)
    ]
    if not missing_files and not missing_dirs:
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

    for local, remote in missing_files:
        dest = root / local
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s from HuggingFace Hub ...", remote)
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=remote,
                repo_type=HF_REPO_TYPE,
                token=token,
                local_dir=str(root / "data"),
            )
            logger.info("Saved to %s", dest)
        except Exception as exc:
            if local in _OPTIONAL_ARTIFACTS:
                logger.warning(
                    "Optional artifact missing on HF Hub: %s (%s). "
                    "bm25s will be rebuilt locally if needed.",
                    remote,
                    exc,
                )
                continue
            raise RuntimeError(
                f"Failed to download artifact '{remote}' from {HF_REPO_ID}: {exc}"
            ) from exc

    for local_dir, remote_prefix in missing_dirs:
        try:
            _download_dir_artifact(root, local_dir, remote_prefix, token)
        except Exception as exc:
            if local_dir in _OPTIONAL_DIR_ARTIFACTS:
                logger.warning(
                    "Optional dir artifact missing on HF Hub: %s (%s). "
                    "Build locally: python -m retrievers.lancedb_rag --build-from-faiss",
                    remote_prefix,
                    exc,
                )
            else:
                raise RuntimeError(
                    f"Failed to download dir artifact '{remote_prefix}' from {HF_REPO_ID}: {exc}"
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

    for local_dir, remote_prefix in _DIR_ARTIFACTS.items():
        src_dir = root / local_dir
        if not src_dir.is_dir() or not any(src_dir.rglob("*")):
            logger.warning("Skipping missing dir artifact: %s", src_dir)
            continue
        logger.info("Uploading dir %s -> %s ...", src_dir, remote_prefix)
        api.upload_folder(
            folder_path=str(src_dir),
            path_in_repo=remote_prefix,
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
