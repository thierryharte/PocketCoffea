"""Append the inclusive LHE scale/PDF ratios to the metadata of a dataset definition.

For every MC dataset, the NanoAOD ``Runs`` trees of the original (unskimmed)
files are read and combined into

    LHEScaleSumw[i] = sum(genWeight * LHEScaleWeight[i]) / sum(genWeight)
    LHEPdfSumw[i]   = sum(genWeight * LHEPdfWeight[i])   / sum(genWeight)

over the full dataset (same definition as the ``Runs`` branches of a single
file). They are written into the metadata of the target dataset definition,
typically the skimmed one, whose files no longer carry a ``Runs`` tree.
The ``lhe_scale`` and ``lhe_pdf`` weights use them to normalise the variations
to their inclusive effect (see ``pocket_coffea.lib.weights.common.common``).
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import click
import numpy as np

LHE_SUMW_BRANCHES = ["LHEScaleSumw", "LHEPdfSumw"]


def read_runs_sumw(file_path):
    """Read the ``Runs`` tree of one NanoAOD file and return the per-file totals:
    ``genEventSumw``, ``genEventCount`` and, for each available LHE branch, the
    *un-normalised* sum ``sum(genWeight * w_i)`` (the stored value times
    ``genEventSumw``, summed over the runs of the file)."""
    import uproot  # lazy import: keeps the module importable without uproot

    with uproot.open(file_path, timeout=120) as nanoaod_file:
        runs_tree = nanoaod_file["Runs"]
        available_lhe_branches = [b for b in LHE_SUMW_BRANCHES if b in runs_tree]
        runs = runs_tree.arrays(
            ["genEventSumw", "genEventCount"] + available_lhe_branches, library="np"
        )
    per_file_sums = {
        "genEventSumw": float(np.sum(runs["genEventSumw"])),
        "genEventCount": int(np.sum(runs["genEventCount"])),
    }
    for branch in available_lhe_branches:
        per_run_ratios = np.stack(runs[branch])  # (n_runs, n_weights)
        per_file_sums[branch] = (per_run_ratios * runs["genEventSumw"][:, np.newaxis]).sum(axis=0)
    return per_file_sums


def combine_runs_sumw(per_file_sums):
    """Combine the output of :func:`read_runs_sumw` for all files of a dataset.

    Returns ``(inclusive_ratios, total_genEventSumw, total_genEventCount)`` where
    ``inclusive_ratios`` maps each LHE branch present in *every* file to the
    list ``sum(genWeight * w_i) / sum(genWeight)``."""
    total_genEventSumw = sum(sums["genEventSumw"] for sums in per_file_sums)
    total_genEventCount = sum(sums["genEventCount"] for sums in per_file_sums)
    inclusive_ratios = {}
    for branch in LHE_SUMW_BRANCHES:
        if not all(branch in sums for sums in per_file_sums):
            continue
        n_weights = {len(sums[branch]) for sums in per_file_sums}
        if len(n_weights) != 1:
            raise ValueError(f"{branch} has a different length in different files: {sorted(n_weights)}")
        weighted_sum = np.sum([sums[branch] for sums in per_file_sums], axis=0)
        inclusive_ratios[branch] = (weighted_sum / total_genEventSumw).tolist()
    return inclusive_ratios, total_genEventSumw, total_genEventCount


def is_mc(metadata):
    return str(metadata.get("isMC", "True")).lower() == "true"


@click.command()
@click.option(
    "-u", "--unskimmed",
    required=True,
    type=click.Path(exists=True),
    help="Dataset definition json with the original (unskimmed) NanoAOD files",
)
@click.option(
    "-t", "--target",
    type=click.Path(exists=True),
    help="Dataset definition json to add the metadata to (e.g. the skimmed one). Defaults to --unskimmed",
)
@click.option(
    "-o", "--output",
    type=click.Path(),
    help="Output json. Defaults to overwriting --target",
)
@click.option(
    "-d", "--datasets",
    multiple=True,
    help="Only process these datasets (can be repeated). Defaults to all MC datasets of --target",
)
@click.option("-j", "--workers", default=8, show_default=True, help="Number of files read in parallel")
@click.option("--indent", default=2, show_default=True, help="Indentation of the output json")
def append_lhe_sumw(unskimmed, target, output, datasets, workers, indent):
    """Append the inclusive LHEScaleSumw and LHEPdfSumw of each MC dataset to its metadata."""
    target = target or unskimmed
    output = output or target
    with open(unskimmed) as f:
        unskimmed_datasets = json.load(f)
    with open(target) as f:
        target_datasets = json.load(f)

    selected_datasets = list(datasets) or [
        name for name, definition in target_datasets.items() if is_mc(definition["metadata"])
    ]
    missing_datasets = [name for name in selected_datasets if name not in unskimmed_datasets]
    if missing_datasets:
        raise click.ClickException(f"Datasets not found in {unskimmed}: {missing_datasets}")

    for dataset_name in selected_datasets:
        original_files = unskimmed_datasets[dataset_name]["files"]
        print(f"{dataset_name}: reading the Runs tree of {len(original_files)} files")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            per_file_sums = list(pool.map(read_runs_sumw, original_files))
        inclusive_ratios, total_genEventSumw, total_genEventCount = combine_runs_sumw(per_file_sums)

        expected_nevents = int(unskimmed_datasets[dataset_name]["metadata"].get("nevents", total_genEventCount))
        if expected_nevents != total_genEventCount:
            logging.warning(
                f"{dataset_name}: genEventCount of the files ({total_genEventCount}) differs from the "
                f"nevents in the metadata ({expected_nevents}): are some files missing?"
            )
        if not inclusive_ratios:
            logging.warning(f"{dataset_name}: no LHE sums in the Runs tree, nothing added")
            continue

        target_metadata = target_datasets[dataset_name]["metadata"]
        target_metadata.update(inclusive_ratios)
        summary = ", ".join(
            f"{branch}: {len(ratios)} entries, range [{min(ratios):.4f}, {max(ratios):.4f}]"
            for branch, ratios in inclusive_ratios.items()
        )
        print(f"  genEventSumw = {total_genEventSumw:.6g}; {summary}")

    with open(output, "w") as f:
        json.dump(target_datasets, f, indent=indent)
    print(f"Written {output}")


if __name__ == "__main__":
    append_lhe_sumw()
