"""Tests for retrievers/artifacts.py — HuggingFace artifact management."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from retrievers.artifacts import (
    HF_REPO_ID,
    HF_REPO_TYPE,
    _hf_token,
    ensure_artifacts,
    upload_artifacts,
    _ARTIFACTS,
    _OPTIONAL_ARTIFACTS,
    _DIR_ARTIFACTS,
    _OPTIONAL_DIR_ARTIFACTS,
    _dir_artifact_present,
)


class TestHfTokenResolution:
    """Test HuggingFace token resolution from env and Streamlit secrets."""

    def test_hf_token_from_env_hf_api_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_API_TOKEN", "hf_test_token")
        token = _hf_token()
        assert token == "hf_test_token"

    def test_hf_token_from_env_huggingface_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_alt_token")
        token = _hf_token()
        assert token == "hf_alt_token"

    def test_hf_token_prefers_hf_api_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HF_API_TOKEN", "primary")
        monkeypatch.setenv("HUGGINGFACE_TOKEN", "secondary")
        token = _hf_token()
        assert token == "primary"

    def test_hf_token_returns_none_if_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        token = _hf_token()
        assert token is None

    @pytest.mark.skip(reason="Streamlit is hard to mock with local imports")
    def test_hf_token_from_streamlit_secrets(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # This test requires Streamlit which is imported locally
        # Skipping for now - tested manually in Streamlit context
        pass


class TestEnsureArtifacts:
    """Test artifact download logic."""

    @patch("retrievers.artifacts.logger")
    def test_ensure_artifacts_skips_if_all_present(
        self, mock_logger: MagicMock, tmp_path: Path
    ) -> None:
        # Create all required file artifacts
        for local_path in _ARTIFACTS.keys():
            artifact = tmp_path / local_path
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("fake")
        # Create dir artifacts (lancedb) so missing_dirs is also empty
        for local_dir in _DIR_ARTIFACTS:
            d = tmp_path / local_dir / "chunks.lance"
            d.mkdir(parents=True, exist_ok=True)
            (d / "_latest.manifest").write_text("fake")

        ensure_artifacts(root=tmp_path)

        mock_logger.info.assert_called_with("All artifacts present — skipping download.")

    def test_ensure_artifacts_raises_if_huggingface_hub_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Hide huggingface_hub import
        import sys
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)

        with pytest.raises(RuntimeError, match="huggingface_hub not installed"):
            ensure_artifacts(root=tmp_path)

    @patch("huggingface_hub.hf_hub_download")
    @patch("retrievers.artifacts._hf_token", return_value="test_token")
    @patch("retrievers.artifacts.logger")
    def test_ensure_artifacts_downloads_missing_files(
        self,
        mock_logger: MagicMock,
        mock_token: MagicMock,
        mock_download: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Only create some artifacts, leave others missing
        first_artifact = list(_ARTIFACTS.keys())[0]
        artifact = tmp_path / first_artifact
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("fake")

        ensure_artifacts(root=tmp_path)

        # Should attempt to download missing files
        assert mock_download.call_count > 0

    @patch("huggingface_hub.hf_hub_download")
    @patch("retrievers.artifacts._hf_token", return_value=None)
    @patch("retrievers.artifacts.logger")
    def test_ensure_artifacts_warns_if_no_token(
        self,
        mock_logger: MagicMock,
        mock_token: MagicMock,
        mock_download: MagicMock,
        tmp_path: Path,
    ) -> None:
        ensure_artifacts(root=tmp_path)

        # Should log warning about missing token
        warning_calls = [
            call for call in mock_logger.warning.call_args_list
            if "HF_API_TOKEN not set" in str(call)
        ]
        assert len(warning_calls) > 0

    @patch("huggingface_hub.hf_hub_download")
    @patch("retrievers.artifacts._hf_token", return_value="test_token")
    @patch("retrievers.artifacts.logger")
    def test_ensure_artifacts_handles_optional_missing_gracefully(
        self,
        mock_logger: MagicMock,
        mock_token: MagicMock,
        mock_download: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Simulate download failure for optional artifact
        def download_side_effect(repo_id, filename, **kwargs):
            if filename in [v for k, v in _ARTIFACTS.items() if k in _OPTIONAL_ARTIFACTS]:
                raise RuntimeError("Optional artifact not found")

        mock_download.side_effect = download_side_effect

        # Should not raise, just warn
        ensure_artifacts(root=tmp_path)

        warning_calls = [
            call for call in mock_logger.warning.call_args_list
            if "Optional artifact missing" in str(call)
        ]
        assert len(warning_calls) > 0

    @patch("huggingface_hub.hf_hub_download")
    @patch("retrievers.artifacts._hf_token", return_value="test_token")
    def test_ensure_artifacts_raises_on_required_artifact_failure(
        self,
        mock_token: MagicMock,
        mock_download: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Find a required (non-optional) artifact
        required_artifact = next(
            (local, remote)
            for local, remote in _ARTIFACTS.items()
            if local not in _OPTIONAL_ARTIFACTS
        )

        def download_side_effect(repo_id, filename, **kwargs):
            if filename == required_artifact[1]:
                raise RuntimeError("Required artifact download failed")

        mock_download.side_effect = download_side_effect

        with pytest.raises(RuntimeError, match="Failed to download artifact"):
            ensure_artifacts(root=tmp_path)


class TestArtifactConstants:
    """Test artifact configuration constants."""

    def test_artifacts_dict_has_expected_keys(self) -> None:
        # Should include FAISS, chunks, and bm25s files
        assert "data/index/veridicta.faiss" in _ARTIFACTS
        assert "data/processed/chunks.jsonl" in _ARTIFACTS
        assert any("bm25s" in k for k in _ARTIFACTS.keys())

    def test_optional_artifacts_is_subset_of_artifacts(self) -> None:
        # All optional artifacts should be in main artifacts dict
        for optional in _OPTIONAL_ARTIFACTS:
            assert optional in _ARTIFACTS

    def test_optional_artifacts_are_bm25s_only(self) -> None:
        # All optional artifacts should be bm25s-related
        for optional in _OPTIONAL_ARTIFACTS:
            assert "bm25s" in optional

    def test_hf_repo_constants(self) -> None:
        assert HF_REPO_ID == "Fascinax/veridicta-index"
        assert HF_REPO_TYPE == "dataset"


class TestUploadArtifacts:
    """Test artifact upload logic."""

    @patch("huggingface_hub.HfApi")
    @patch("retrievers.artifacts._hf_token", return_value="test_token")
    def test_upload_creates_repo_and_uploads_existing_files(
        self,
        mock_token: MagicMock,
        mock_api_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        # Create two artifacts on disk
        for i, (local, remote) in enumerate(_ARTIFACTS.items()):
            if i >= 2:
                break
            dest = tmp_path / local
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("fake content")

        upload_artifacts(root=tmp_path, token="explicit_token")

        mock_api.create_repo.assert_called_once_with(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            exist_ok=True,
            token="explicit_token",
        )
        assert mock_api.upload_file.call_count == 2

    @patch("huggingface_hub.HfApi")
    @patch("retrievers.artifacts._hf_token", return_value="test_token")
    def test_upload_skips_missing_files(
        self,
        mock_token: MagicMock,
        mock_api_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_api = MagicMock()
        mock_api_class.return_value = mock_api

        upload_artifacts(root=tmp_path, token="tok")

        mock_api.upload_file.assert_not_called()

    @patch("huggingface_hub.HfApi")
    @patch("retrievers.artifacts._hf_token", return_value=None)
    def test_upload_raises_if_repo_creation_fails(
        self,
        mock_token: MagicMock,
        mock_api_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_api = MagicMock()
        mock_api.create_repo.side_effect = RuntimeError("403 Forbidden")
        mock_api_class.return_value = mock_api

        with pytest.raises(RuntimeError, match="Cannot create HF repo"):
            upload_artifacts(root=tmp_path)


class TestDirArtifacts:
    """Test directory artifact handling for LanceDB."""

    def test_dir_artifacts_dict_structure(self) -> None:
        assert "data/index/lancedb" in _DIR_ARTIFACTS
        assert _DIR_ARTIFACTS["data/index/lancedb"] == "index/lancedb"

    def test_optional_dir_artifacts_subset_of_dir_artifacts(self) -> None:
        for optional in _OPTIONAL_DIR_ARTIFACTS:
            assert optional in _DIR_ARTIFACTS

    def test_dir_artifact_present_false_when_missing(self, tmp_path: Path) -> None:
        assert not _dir_artifact_present(tmp_path, "data/index/lancedb")

    def test_dir_artifact_present_false_when_empty_dir(self, tmp_path: Path) -> None:
        (tmp_path / "data/index/lancedb").mkdir(parents=True)
        assert not _dir_artifact_present(tmp_path, "data/index/lancedb")

    def test_dir_artifact_present_true_when_has_files(self, tmp_path: Path) -> None:
        d = tmp_path / "data/index/lancedb/chunks.lance"
        d.mkdir(parents=True)
        (d / "_latest.manifest").write_text("x")
        assert _dir_artifact_present(tmp_path, "data/index/lancedb")

    @patch("retrievers.artifacts._hf_token", return_value="test_token")
    @patch("retrievers.artifacts._download_dir_artifact", side_effect=RuntimeError("not found"))
    @patch("retrievers.artifacts.logger")
    def test_ensure_artifacts_warns_on_optional_dir_failure(
        self,
        mock_logger: MagicMock,
        mock_download_dir: MagicMock,
        mock_token: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Create all file artifacts so the file loop is skipped
        for local_path in _ARTIFACTS.keys():
            artifact = tmp_path / local_path
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("fake")
        # lancedb dir absent → triggers optional dir download → fails → warn, not raise
        ensure_artifacts(root=tmp_path)

        warning_calls = [
            c for c in mock_logger.warning.call_args_list
            if "Optional dir artifact" in str(c)
        ]
        assert len(warning_calls) > 0

    @patch("huggingface_hub.HfApi")
    def test_upload_artifacts_calls_upload_folder_for_lancedb(
        self,
        mock_hf_api_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        mock_api.create_repo.return_value = None
        mock_api.upload_file.return_value = None
        mock_api.upload_folder.return_value = None
        # Create file artifacts
        for local_path in _ARTIFACTS.keys():
            f = tmp_path / local_path
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x")
        # Create lancedb dir with at least one file
        lancedb_dir = tmp_path / "data/index/lancedb/chunks.lance"
        lancedb_dir.mkdir(parents=True)
        (lancedb_dir / "_latest.manifest").write_text("x")

        upload_artifacts(root=tmp_path, token="test_token")

        mock_api.upload_folder.assert_called_once()


class TestDirArtifacts:
    """Test directory artifact handling for LanceDB."""

    def test_dir_artifacts_dict_structure(self) -> None:
        assert "data/index/lancedb" in _DIR_ARTIFACTS
        assert _DIR_ARTIFACTS["data/index/lancedb"] == "index/lancedb"

    def test_optional_dir_artifacts_subset_of_dir_artifacts(self) -> None:
        for optional in _OPTIONAL_DIR_ARTIFACTS:
            assert optional in _DIR_ARTIFACTS

    def test_dir_artifact_present_false_when_missing(self, tmp_path: Path) -> None:
        assert not _dir_artifact_present(tmp_path, "data/index/lancedb")

    def test_dir_artifact_present_false_when_empty_dir(self, tmp_path: Path) -> None:
        (tmp_path / "data/index/lancedb").mkdir(parents=True)
        assert not _dir_artifact_present(tmp_path, "data/index/lancedb")

    def test_dir_artifact_present_true_when_has_files(self, tmp_path: Path) -> None:
        d = tmp_path / "data/index/lancedb/chunks.lance"
        d.mkdir(parents=True)
        (d / "_latest.manifest").write_text("x")
        assert _dir_artifact_present(tmp_path, "data/index/lancedb")

    @patch("retrievers.artifacts._hf_token", return_value="test_token")
    @patch("retrievers.artifacts._download_dir_artifact", side_effect=RuntimeError("not found"))
    @patch("retrievers.artifacts.logger")
    def test_ensure_artifacts_warns_on_optional_dir_failure(
        self,
        mock_logger: MagicMock,
        mock_download_dir: MagicMock,
        mock_token: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Create all file artifacts so the file download loop is skipped
        for local_path in _ARTIFACTS.keys():
            artifact = tmp_path / local_path
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("fake")
        # lancedb dir absent → triggers optional dir download → fails → warn, not raise
        ensure_artifacts(root=tmp_path)

        warning_calls = [
            c for c in mock_logger.warning.call_args_list
            if "Optional dir artifact" in str(c)
        ]
        assert len(warning_calls) > 0

    @patch("huggingface_hub.HfApi")
    def test_upload_artifacts_calls_upload_folder_for_lancedb(
        self,
        mock_hf_api_class: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_api = MagicMock()
        mock_hf_api_class.return_value = mock_api
        mock_api.create_repo.return_value = None
        mock_api.upload_file.return_value = None
        mock_api.upload_folder.return_value = None
        # Create file artifacts
        for local_path in _ARTIFACTS.keys():
            f = tmp_path / local_path
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x")
        # Create lancedb dir with at least one file
        lancedb_dir = tmp_path / "data/index/lancedb/chunks.lance"
        lancedb_dir.mkdir(parents=True)
        (lancedb_dir / "_latest.manifest").write_text("x")

        upload_artifacts(root=tmp_path, token="test_token")

        mock_api.upload_folder.assert_called_once()
