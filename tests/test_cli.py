"""cli.py: arg parsing helpers + argparse smoke checks."""
from __future__ import annotations
import subprocess
import sys

import pytest

from e2e.cli import _parse_csv, DEFAULT_INFERENCE_SET


class TestDefaultInferenceSet:
    def test_default_set_is_the_default_subdir(self):
        # `infer` runs inferences/default/ unless --inferences-dir overrides it.
        assert DEFAULT_INFERENCE_SET.name == "default"
        assert DEFAULT_INFERENCE_SET.parent.name == "inferences"


class TestParseCsv:
    def test_none_input(self):
        assert _parse_csv(None) is None

    def test_empty_string(self):
        assert _parse_csv("") is None

    def test_whitespace_only(self):
        assert _parse_csv("   ,  , ") is None

    def test_single_item(self):
        assert _parse_csv("foo") == ["foo"]

    def test_multi_strips_whitespace(self):
        assert _parse_csv(" foo, bar ,baz ") == ["foo", "bar", "baz"]

    def test_skips_empty_segments(self):
        assert _parse_csv("foo,,bar,") == ["foo", "bar"]


# -------------------------------------------------------------------------
# argparse end-to-end: actually launch `python -m e2e ...` and check it
# parses our flag matrix without ImportError / typo crashes.
# -------------------------------------------------------------------------

def _run_cli(*args, cwd) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "e2e", *args],
                          cwd=cwd, capture_output=True, text=True, timeout=30)


@pytest.fixture
def framework_root():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent


class TestCliArgparse:
    def test_top_level_help(self, framework_root):
        r = _run_cli("--help", cwd=framework_root)
        assert r.returncode == 0
        for word in ("deploy", "poc", "infer", "validate"):
            assert word in r.stdout

    def test_deploy_requires_args(self, framework_root):
        r = _run_cli("deploy", cwd=framework_root)
        assert r.returncode != 0
        # argparse complains about missing required args in stderr
        for arg in ("--ssh-host", "--model-name", "--gpu-name",
                    "--host-model-path"):
            assert arg in r.stderr, f"{arg!r} expected in stderr:\n{r.stderr}"

    def test_validate_requires_executor_run_id(self, framework_root):
        r = _run_cli("validate", "--ssh-host", "u@h",
                     "--model-name", "M", "--gpu-name", "g",
                     cwd=framework_root)
        assert r.returncode != 0
        assert "--executor-run-id" in r.stderr

    def test_infer_help_exposes_inferences_filter(self, framework_root):
        r = _run_cli("infer", "--help", cwd=framework_root)
        assert r.returncode == 0
        assert "--inferences" in r.stdout

    def test_infer_help_exposes_inferences_dir(self, framework_root):
        r = _run_cli("infer", "--help", cwd=framework_root)
        assert r.returncode == 0
        assert "--inferences-dir" in r.stdout

    def test_validate_help_default_pass_value_is_0_9(self, framework_root):
        r = _run_cli("validate", "--help", cwd=framework_root)
        assert r.returncode == 0
        assert "0.9" in r.stdout

    def test_artifacts_dir_flag_is_gone(self, framework_root):
        """User explicitly requested removing this — guard against regression."""
        r = _run_cli("infer", "--help", cwd=framework_root)
        assert "--artifacts-dir" not in r.stdout

    def test_run_id_flag_is_gone(self, framework_root):
        """Replaced by --date + --run-name with new artifacts layout."""
        for cmd in ("deploy", "poc", "infer"):
            r = _run_cli(cmd, "--help", cwd=framework_root)
            assert "--run-id" not in r.stdout, \
                f"--run-id still in `{cmd} --help`"

    def test_date_and_run_name_flags(self, framework_root):
        for cmd in ("deploy", "poc", "infer", "validate"):
            r = _run_cli(cmd, "--help", cwd=framework_root)
            assert "--date" in r.stdout
            assert "--run-name" in r.stdout

    def test_vllm_url_flag_is_gone(self, framework_root):
        """vLLM URL is always derived from an SSH forward tunnel now —
        users no longer specify (and should never expose) a public URL."""
        for cmd in ("deploy", "poc", "infer", "validate"):
            r = _run_cli(cmd, "--help", cwd=framework_root)
            assert "--vllm-url" not in r.stdout, \
                f"--vllm-url still in `{cmd} --help`"

    def test_tp_alias_works(self, framework_root):
        r = _run_cli("deploy", "--help", cwd=framework_root)
        assert "--tp" in r.stdout
        assert "--pp" in r.stdout

    def test_logprobs_mode_has_choices(self, framework_root):
        r = _run_cli("deploy", "--help", cwd=framework_root)
        for choice in ("processed_logprobs", "raw_logprobs",
                       "processed_logits", "raw_logits"):
            assert choice in r.stdout
