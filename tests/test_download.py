"""download.py: snapshot_download wrapper. We patch subprocess.run +
subprocess.Popen to avoid hitting any real SSH/network."""
from __future__ import annotations
from unittest.mock import MagicMock, patch

import pytest

from e2e.config import ServerTarget
from e2e.download import _DOWNLOAD_SCRIPT, _ensure_hf_tools, download_model


@pytest.fixture
def target() -> ServerTarget:
    return ServerTarget(ssh_host="u@h", ssh_port=2222, gpu_name="t")


class TestEnsureHfTools:
    def test_runs_pip_install_via_ssh(self, target):
        with patch("e2e.download.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            _ensure_hf_tools(target)
            args = mock_run.call_args[0][0]
            # First three are ssh args
            assert args[:4] == ["ssh", "-p", "2222", "u@h"]
            cmd = args[-1]
            assert "huggingface_hub" in cmd
            assert "hf_transfer" in cmd
            # Uses dedicated venv (portable across distros + pip versions)
            assert ".e2e-venv" in cmd
            assert "python3 -m venv" in cmd


class TestDownloadScript:
    def test_script_format_substitutes_repo_path_patterns(self):
        script = _DOWNLOAD_SCRIPT.format(
            repo="Org/Model", local_dir="/data/m",
            patterns=["*.safetensors", "*.json"],
        )
        assert "'Org/Model'" in script
        assert "'/data/m'" in script
        assert "*.safetensors" in script
        assert "snapshot_download" in script
        assert "HF_HUB_ENABLE_HF_TRANSFER" in script


class TestDownloadModel:
    def test_invokes_ssh_with_python_one_liner(self, target):
        """End-to-end orchestration: ssh→pip→ssh→python3 -c snapshot_download."""
        with patch("e2e.download.subprocess.run") as mock_run, \
             patch("e2e.download.subprocess.Popen") as mock_popen:
            mock_run.return_value = MagicMock(returncode=0)
            fake_proc = MagicMock()
            fake_proc.stdout.readline.side_effect = [
                "downloading...\n", "DOWNLOADED in 60s\n", ""
            ]
            fake_proc.wait.return_value = 0
            mock_popen.return_value = fake_proc

            download_model(target, "Org/Model", "/data/m")

            # pip install happened
            assert mock_run.called
            # ssh + python3 -c happened
            assert mock_popen.called
            popen_args = mock_popen.call_args[0][0]
            assert popen_args[:4] == ["ssh", "-p", "2222", "u@h"]
            joined = " ".join(popen_args[4:])
            assert "python3 -c" in joined
            assert "Org/Model" in joined
            assert "/data/m" in joined

    def test_nonzero_exit_raises(self, target):
        with patch("e2e.download.subprocess.run") as mock_run, \
             patch("e2e.download.subprocess.Popen") as mock_popen:
            mock_run.return_value = MagicMock(returncode=0)
            fake_proc = MagicMock()
            fake_proc.stdout.readline = MagicMock(side_effect=["", ""])
            fake_proc.wait.return_value = 1
            mock_popen.return_value = fake_proc
            with pytest.raises(RuntimeError, match="exit code 1"):
                download_model(target, "Org/Model", "/data/m")

    def test_custom_patterns_propagated(self, target):
        with patch("e2e.download.subprocess.run") as mock_run, \
             patch("e2e.download.subprocess.Popen") as mock_popen:
            mock_run.return_value = MagicMock(returncode=0)
            fake_proc = MagicMock()
            fake_proc.stdout.readline = MagicMock(side_effect=["", ""])
            fake_proc.wait.return_value = 0
            mock_popen.return_value = fake_proc

            download_model(target, "Org/Model", "/data/m",
                           allow_patterns=["*.bin"])

            popen_args = mock_popen.call_args[0][0]
            joined = " ".join(popen_args[4:])
            assert "*.bin" in joined
            assert "*.safetensors" not in joined
