"""Offline tests for append-lhe-sumw: the Runs-tree reader is monkeypatched."""
import json

import numpy as np
import pytest
from click.testing import CliRunner

from pocket_coffea.scripts.dataset import append_lhe_sumw as script


def _file_sums(genEventSumw, genEventCount, scale_ratios, pdf_ratios=None):
    """Per-file sums as returned by read_runs_sumw, built from normalised ratios."""
    sums = {
        "genEventSumw": genEventSumw,
        "genEventCount": genEventCount,
        "LHEScaleSumw": np.asarray(scale_ratios) * genEventSumw,
    }
    if pdf_ratios is not None:
        sums["LHEPdfSumw"] = np.asarray(pdf_ratios) * genEventSumw
    return sums


def test_combine_runs_sumw_weights_files_by_genEventSumw():
    per_file_sums = [
        _file_sums(1.0, 10, [1.2, 1.0], [1.0, 1.1]),
        _file_sums(3.0, 30, [0.8, 1.0], [1.0, 0.9]),
    ]
    inclusive_ratios, total_genEventSumw, total_genEventCount = script.combine_runs_sumw(per_file_sums)
    assert total_genEventSumw == 4.0
    assert total_genEventCount == 40
    assert np.allclose(inclusive_ratios["LHEScaleSumw"], [(1.2 + 3 * 0.8) / 4, 1.0])
    assert np.allclose(inclusive_ratios["LHEPdfSumw"], [1.0, (1.1 + 3 * 0.9) / 4])


def test_combine_runs_sumw_skips_branch_missing_in_some_file():
    per_file_sums = [_file_sums(1.0, 1, [1.0], [1.0]), _file_sums(1.0, 1, [1.0])]
    inclusive_ratios, _, _ = script.combine_runs_sumw(per_file_sums)
    assert list(inclusive_ratios) == ["LHEScaleSumw"]


def test_combine_runs_sumw_inconsistent_lengths_raise():
    per_file_sums = [_file_sums(1.0, 1, [1.0] * 9), _file_sums(1.0, 1, [1.0] * 8)]
    with pytest.raises(ValueError, match="different length"):
        script.combine_runs_sumw(per_file_sums)


def test_cli_writes_metadata_of_target(tmp_path, monkeypatch):
    unskimmed = {
        "sig_2024": {"files": ["a.root", "b.root"], "metadata": {"isMC": "True", "nevents": "20"}},
        "data_2024": {"files": ["d.root"], "metadata": {"isMC": "False"}},
    }
    skimmed = {
        "sig_2024": {"files": ["skim.root"], "metadata": {"isMC": "True", "isSkim": "True"}},
        "data_2024": {"files": ["dskim.root"], "metadata": {"isMC": "False"}},
    }
    unskimmed_path = tmp_path / "unskimmed.json"
    skimmed_path = tmp_path / "skimmed.json"
    unskimmed_path.write_text(json.dumps(unskimmed))
    skimmed_path.write_text(json.dumps(skimmed))

    sums_by_file = {
        "a.root": _file_sums(1.0, 10, [1.1, 1.0], [1.0, 1.2]),
        "b.root": _file_sums(1.0, 10, [0.9, 1.0], [1.0, 1.0]),
    }
    monkeypatch.setattr(script, "read_runs_sumw", lambda path: sums_by_file[path])

    result = CliRunner().invoke(
        script.append_lhe_sumw, ["-u", str(unskimmed_path), "-t", str(skimmed_path)]
    )
    assert result.exit_code == 0, result.output

    written = json.loads(skimmed_path.read_text())
    assert np.allclose(written["sig_2024"]["metadata"]["LHEScaleSumw"], [1.0, 1.0])
    assert np.allclose(written["sig_2024"]["metadata"]["LHEPdfSumw"], [1.0, 1.1])
    assert written["sig_2024"]["files"] == ["skim.root"]
    assert "LHEScaleSumw" not in written["data_2024"]["metadata"]
