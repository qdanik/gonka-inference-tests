"""ServerTarget, ModelSpec, RunPaths, make_run_name + make_date_str."""
from __future__ import annotations
import re
from pathlib import Path

import pytest

from e2e.config import (
    ModelSpec, RunPaths, ServerTarget,
    make_date_str, make_run_name,
)


class TestServerTarget:
    def test_ssh_cmd_inserts_port(self):
        t = ServerTarget(ssh_host="u@h", ssh_port=7722, gpu_name="b300")
        assert t.ssh_cmd("ls -la") == ["ssh", "-p", "7722", "u@h", "ls -la"]

    def test_defaults_match_documented(self):
        t = ServerTarget(ssh_host="u@h", gpu_name="b300")
        assert t.tensor_parallel_size == 1
        assert t.pipeline_parallel_size == 1
        assert t.gpu_memory_utilization == 0.95
        assert t.max_model_len == 16384
        assert t.max_num_seqs == 128
        assert t.enforce_eager is False
        assert t.logprobs_mode == "processed_logprobs"
        assert t.docker_image == "ghcr.io/kaitakuai/vllm:0.20.0-pocv2"


class TestModelSpec:
    def test_minimal_construction(self):
        m = ModelSpec(name="A/B", hf_repo="A/B")
        assert m.container_mount == "/model"
        assert m.extra_args == []
        assert m.extra_env == {}


class TestMakeDateStr:
    def test_format_yyyy_mm_dd(self):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", make_date_str())


class TestMakeRunName:
    def test_basename_plus_gpu(self):
        assert make_run_name("MiniMaxAI/MiniMax-M2.7", "2xb200") == \
               "MiniMax-M2.7-2xb200"

    def test_model_without_org_prefix(self):
        assert make_run_name("Qwen3-235B", "4xh100") == "Qwen3-235B-4xh100"

    def test_sanitizes_spaces_and_slashes_in_gpu(self):
        assert make_run_name("A/B", "h100 tp/4") == "B-h100_tp_4"


def _rp(base: Path, name: str = "MiniMax-M2.7-2xb200") -> RunPaths:
    return RunPaths(base_dir=base, date_str="2026-06-05", run_name=name)


class TestRunPaths:
    def test_root_includes_date_and_run_name(self, tmp_artifacts: Path):
        rp = _rp(tmp_artifacts)
        assert rp.root == tmp_artifacts / "2026-06-05" / "MiniMax-M2.7-2xb200"
        assert rp.poc == rp.root / "_poc"

    def test_run_id_is_path_form(self, tmp_artifacts: Path):
        rp = _rp(tmp_artifacts)
        assert rp.run_id == "2026-06-05/MiniMax-M2.7-2xb200"

    def test_ensure_creates_both_levels(self, tmp_artifacts: Path):
        rp = _rp(tmp_artifacts); rp.ensure()
        assert rp.root.is_dir() and rp.poc.is_dir()
        # Parent date dir exists too
        assert (tmp_artifacts / "2026-06-05").is_dir()

    def test_label_dir_creates_on_access(self, tmp_artifacts: Path):
        rp = _rp(tmp_artifacts); rp.ensure()
        d = rp.label_dir("sys_math_en")
        assert d.is_dir() and d.name == "sys_math_en"

    def test_next_index_starts_at_one_on_empty(self, tmp_artifacts: Path):
        rp = _rp(tmp_artifacts); rp.ensure()
        assert rp.next_index("foo", "inference") == 1

    def test_next_index_auto_increments(self, tmp_artifacts: Path):
        rp = _rp(tmp_artifacts); rp.ensure()
        d = rp.label_dir("foo")
        (d / "inference-1.json").write_text("{}")
        (d / "inference-2.json").write_text("{}")
        assert rp.next_index("foo", "inference") == 3

    def test_next_index_handles_gaps(self, tmp_artifacts: Path):
        rp = _rp(tmp_artifacts); rp.ensure()
        d = rp.label_dir("foo")
        (d / "inference-1.json").write_text("{}")
        (d / "inference-5.json").write_text("{}")
        assert rp.next_index("foo", "inference") == 6

    def test_next_index_isolated_per_prefix(self, tmp_artifacts: Path):
        rp = _rp(tmp_artifacts); rp.ensure()
        d = rp.label_dir("foo")
        (d / "inference-1.json").write_text("{}")
        (d / "validated-by-h100-1.json").write_text("{}")
        (d / "validated-by-h100-2.json").write_text("{}")
        assert rp.next_index("foo", "inference") == 2
        assert rp.next_index("foo", "validated-by-h100") == 3
        assert rp.next_index("foo", "validated-by-rtx") == 1

    def test_next_index_isolated_per_label(self, tmp_artifacts: Path):
        rp = _rp(tmp_artifacts); rp.ensure()
        (rp.label_dir("foo") / "inference-1.json").write_text("{}")
        (rp.label_dir("foo") / "inference-2.json").write_text("{}")
        assert rp.next_index("bar", "inference") == 1

    def test_next_index_ignores_unrelated_files(self, tmp_artifacts: Path):
        rp = _rp(tmp_artifacts); rp.ensure()
        d = rp.label_dir("foo")
        (d / "inference-1.json").write_text("{}")
        (d / "inference-notanumber.json").write_text("{}")
        (d / "inference-2.json").write_text("{}")
        assert rp.next_index("foo", "inference") == 3

    def test_same_day_same_model_same_gpu_shares_directory(self, tmp_artifacts: Path):
        """Two `infer` runs on the same day to the same (model, gpu) accumulate
        inference-N files in the SAME directory — different N values."""
        rp1 = _rp(tmp_artifacts); rp1.ensure()
        (rp1.label_dir("foo") / "inference-1.json").write_text("{}")

        rp2 = _rp(tmp_artifacts); rp2.ensure()    # identical date+run_name
        assert rp2.next_index("foo", "inference") == 2
