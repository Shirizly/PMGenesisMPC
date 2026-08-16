"""Tests for run_collection.resolve_env_counts — where a collection run gets
its per-object-count environment counts from.

Genesis-free: run_collection imports only stdlib + yaml at module level.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Genesis"))

from run_collection import resolve_env_counts  # noqa: E402

MATERIAL = {"shape": "cube", "particle_size": 0.005, "particle_friction": 0.3,
            "particle_density": 1000.0, "box_friction": 0.3}


def _write_optimal(tmp_path, n_envs, measured_under=None, name="optimal.yaml"):
    blob = {"n_envs": n_envs}
    if measured_under is not None:
        blob["measured_under"] = measured_under
    p = tmp_path / name
    p.write_text(yaml.safe_dump(blob))
    return p


def test_literal_mapping_is_used_as_is():
    counts, provenance, warnings = resolve_env_counts(
        {"n_envs": {20: 8, 50: 4}}, MATERIAL)

    assert counts == {20: 8, 50: 4}
    assert "literal" in provenance
    assert warnings == []


def test_reads_counts_from_a_benchmark_file(tmp_path):
    p = _write_optimal(tmp_path, {20: 128, 200: 1},
                       measured_under={**MATERIAL, "measured_at": "2026-01-01T00:00:00",
                                       "gpu": "Test GPU"})

    counts, provenance, warnings = resolve_env_counts({"n_envs": str(p)}, MATERIAL)

    assert counts == {20: 128, 200: 1}
    assert "2026-01-01" in provenance and "Test GPU" in provenance
    assert warnings == []


def test_material_mismatch_is_reported_not_swallowed(tmp_path):
    """A throughput optimum is specific to the material it was measured on."""
    stale = {**MATERIAL, "particle_size": 0.012, "particle_density": 5000.0}
    p = _write_optimal(tmp_path, {20: 128}, measured_under=stale)

    counts, _, warnings = resolve_env_counts({"n_envs": str(p)}, MATERIAL)

    assert counts == {20: 128}, "counts are still returned; the user decides"
    joined = " ".join(warnings)
    assert "particle_size" in joined and "particle_density" in joined


def test_warns_when_measured_without_shared_travel_distance(tmp_path):
    p = _write_optimal(tmp_path, {20: 2},
                       measured_under={**MATERIAL, "shared_travel_distance": False})

    _, _, warnings = resolve_env_counts({"n_envs": str(p)}, MATERIAL)

    assert any("shared travel distance" in w for w in warnings)


def test_missing_benchmark_file_explains_how_to_produce_it(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_env_counts({"n_envs": str(tmp_path / "absent.yaml")}, MATERIAL)

    assert "benchmark_throughput" in str(excinfo.value)


def test_cli_override_beats_the_plan(tmp_path):
    p = _write_optimal(tmp_path, {20: 64}, measured_under=MATERIAL)

    counts, _, _ = resolve_env_counts(
        {"n_envs": {20: 1}}, MATERIAL, cli_override=str(p))

    assert counts == {20: 64}


def test_keys_are_ints_even_when_yaml_gives_strings(tmp_path):
    p = tmp_path / "o.yaml"
    p.write_text('n_envs:\n  "20": "128"\n')

    counts, _, _ = resolve_env_counts({"n_envs": str(p)}, MATERIAL)

    assert counts == {20: 128}
    assert all(isinstance(k, int) and isinstance(v, int) for k, v in counts.items())
