"""Offline unit tests for the LHE scale and PDF variation weights."""
import awkward as ak
import numpy as np
import pytest

from pocket_coffea.lib.weights.common.common import (
    LHEPdfVariations,
    LHEScaleVariations,
    get_inclusive_lhe_ratios,
)


class FakeEvents(dict):
    """Minimal stand-in for NanoEvents: branches by key plus a metadata dict."""

    def __init__(self, branches, metadata):
        super().__init__(branches)
        self.metadata = metadata


def _scale_weights():
    # two events, 9 scale weights each: [i] = 1 + 0.1 * (i - 4) for event 0,
    # [i] = 1 + 0.2 * (i - 4) for event 1 ([4] = 1 = nominal)
    index = np.arange(9)
    return ak.Array([list(1 + 0.1 * (index - 4)), list(1 + 0.2 * (index - 4))])


def test_lhe_scale_variations_without_normalisation():
    events = FakeEvents({"LHEScaleWeight": _scale_weights()}, {"dataset": "dset"})
    out = LHEScaleVariations(params=None, metadata={}).compute(events, 2, "nominal")

    assert out.variations == ["muR", "muF", "muRmuF"]
    assert np.allclose(out.nominal, [1.0, 1.0])
    muR_up, muF_up, muRmuF_up = out.up
    muR_down, muF_down, muRmuF_down = out.down
    assert np.allclose(muR_up, [1.3, 1.6])  # index 7
    assert np.allclose(muR_down, [0.7, 0.4])  # index 1
    assert np.allclose(muF_up, [1.1, 1.2])  # index 5
    assert np.allclose(muF_down, [0.9, 0.8])  # index 3
    assert np.allclose(muRmuF_up, [1.4, 1.8])  # index 8
    assert np.allclose(muRmuF_down, [0.6, 0.2])  # index 0


def test_lhe_scale_variations_normalised_to_inclusive_ratios():
    inclusive_scale_ratios = [0.5, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, 1.5]
    events = FakeEvents(
        {"LHEScaleWeight": _scale_weights()},
        {"dataset": "dset", "LHEScaleSumw": inclusive_scale_ratios},
    )
    out = LHEScaleVariations(params=None, metadata={}).compute(events, 2, "nominal")
    muR_up = out.up[0]
    muR_down = out.down[0]
    assert np.allclose(muR_up, np.array([1.3, 1.6]) / 1.25)
    assert np.allclose(muR_down, np.array([0.7, 0.4]) / 0.8)


def test_lhe_scale_variations_only_nominal_for_shape_variations():
    events = FakeEvents({"LHEScaleWeight": _scale_weights()}, {"dataset": "dset"})
    out = LHEScaleVariations(params=None, metadata={}).compute(events, 2, "JES_TotalUp")
    assert out.up is None and out.down is None
    assert np.allclose(out.nominal, [1.0, 1.0])


def test_lhe_scale_variations_need_nine_weights():
    events = FakeEvents({"LHEScaleWeight": ak.Array([[1.0] * 8])}, {"dataset": "dset"})
    with pytest.raises(ValueError, match="at least 9"):
        LHEScaleVariations(params=None, metadata={}).compute(events, 1, "nominal")


def test_lhe_pdf_variations():
    # 103 members (central, 100 eigenvectors, 2 alpha_s): central != 1 on purpose
    n_members = 103
    pdf_weights = ak.Array([
        [1.1] + [1.1 * (1 + 0.01 * member) for member in range(1, n_members)],
        [0.9] * n_members,
    ])
    inclusive_pdf_ratios = [1.05] + [1.05 * 1.02] * (n_members - 1)
    events = FakeEvents(
        {"LHEPdfWeight": pdf_weights},
        {"dataset": "dset", "LHEPdfSumw": ",".join(map(str, inclusive_pdf_ratios))},
    )
    out = LHEPdfVariations(params=None, metadata={}).compute(events, 2, "nominal")

    assert len(out.variations) == 100
    assert out.variations[0] == "eigenvector1" and out.variations[-1] == "eigenvector100"
    # relative to the central member, divided by the inclusive ratio (1.02)
    assert np.allclose(out.up[0], [1.01 / 1.02, 1.0 / 1.02])
    assert np.allclose(out.up[99], [2.0 / 1.02, 1.0 / 1.02])
    assert all(np.allclose(down, [1.0, 1.0]) for down in out.down)


def test_get_inclusive_lhe_ratios():
    assert get_inclusive_lhe_ratios({}, "LHEPdfSumw") is None
    assert np.allclose(get_inclusive_lhe_ratios({"LHEPdfSumw": "1,2.5"}, "LHEPdfSumw"), [1.0, 2.5])
    assert np.allclose(get_inclusive_lhe_ratios({"LHEPdfSumw": [1, 2.5]}, "LHEPdfSumw"), [1.0, 2.5])
