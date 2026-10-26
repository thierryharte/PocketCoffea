"""Utilities to inject artificial shape variations into pocket_coffea histograms.

These helpers operate on the histogram dictionaries stored in the coffea output
(``df["variables"][variable] -> {sample: {dataset: hist.Hist}}``) *before* they
are handed to :class:`pocket_coffea.utils.stat.combine.Datacard`, by adding new
entries to the ``variation`` StrCategory axis. The new entries can then be
declared as regular ``shape`` :class:`SystematicUncertainty` in the datacard.

Two flavors are provided:

- :func:`add_norm_variation`: Up = nominal scaled by one factor per category
  (pure normalization effect, all bins including under/overflow).
- :func:`add_binwise_variation`: Up = nominal reweighted bin by bin with a
  per-category weight array (generic shape reweighting); a scalar entry behaves
  like a normalization factor.

Two more combine existing variations into a new one:

- :func:`add_envelope_variation`: Up/Down = bin-by-bin max/min over a list of
  variations (e.g. the 7-point QCD scale envelope).
- :func:`add_quadrature_variation`: Up/Down = nominal +- the quadrature sum of
  the deviations of a list of variations (e.g. Hessian PDF eigenvectors).

:func:`rescale_histograms` instead changes the histograms themselves: it
multiplies the nominal *and* every variation of the selected samples/datasets
by one scalar (a plain yield rescaling, no new variation entry).
"""

import hist
import numpy as np


def _resolve_bin_weights(weight_spec, variable_axis, category):
    """Normalize a weight specification to an array over the ``flow=True`` bins
    of the (last) variable axis.

    - A scalar broadcasts to every bin, under/overflow included (pure
      normalization of the category).
    - An array of length ``variable_axis.size`` weights the in-range bins;
      under/overflow keep weight 1.0.
    - An array of length ``variable_axis.extent`` also weights the flow slots
      explicitly (underflow first, when the axis has one).
    """
    if np.ndim(weight_spec) == 0:
        return np.full(variable_axis.extent, float(weight_spec))
    weights = np.asarray(weight_spec, dtype=float)
    if weights.ndim != 1:
        raise ValueError(
            f"Weights for category {category!r} must be a scalar or a 1D array, "
            f"got shape {weights.shape}"
        )
    if len(weights) == variable_axis.extent:
        return weights
    if len(weights) == variable_axis.size:
        underflow = int(variable_axis.traits.underflow)
        overflow = int(variable_axis.traits.overflow)
        return np.concatenate([np.ones(underflow), weights, np.ones(overflow)])
    raise ValueError(
        f"Weights for category {category!r} have length {len(weights)}, expected "
        f"{variable_axis.size} (in-range bins of axis {variable_axis.name!r}, "
        f"flow bins keep weight 1) or {variable_axis.extent} (flow slots included)"
    )


def _select_datasets(histograms, samples, datasets):
    """Return the ``{(sample, dataset)}`` pairs targeted by ``samples`` (all must
    exist in ``histograms``) and ``datasets`` (None = every dataset of those
    samples). Datasets in the list that are absent from the histograms are
    ignored (e.g. empty datasets), but a list matching nothing at all raises."""
    missing_samples = sorted(set(samples) - set(histograms))
    if missing_samples:
        raise ValueError(
            f"Samples {missing_samples} not found in histograms "
            f"(available: {sorted(histograms)})"
        )
    dataset_filter = None if datasets is None else set(datasets)
    selected = {
        (sample, dataset)
        for sample in set(samples)
        for dataset in histograms[sample]
        if dataset_filter is None or dataset in dataset_filter
    }
    if dataset_filter is not None and not selected:
        raise ValueError(
            f"None of the requested datasets {sorted(dataset_filter)} found in "
            f"the histograms of samples {sorted(set(samples))}"
        )
    return selected


def _with_extended_variation_axis(histogram, variation_name, sample, dataset):
    """Return a copy of ``histogram`` whose ``variation`` axis has two more
    entries, ``{variation_name}Up`` and ``{variation_name}Down`` (left empty),
    after checking that the histogram is a Weight-storage MC histogram with
    ``cat``/``variation`` axes, a ``nominal`` entry and no such variation yet."""
    axis_names = [axis.name for axis in histogram.axes]
    if "cat" not in axis_names or "variation" not in axis_names:
        raise ValueError(
            f"Histogram for sample {sample}, dataset {dataset} has no "
            f"'cat'/'variation' axes (axes: {axis_names}); "
            "is this a data histogram?"
        )
    try:
        storage_type = histogram.storage_type
    except AttributeError:  # hist < 2.8
        storage_type = histogram._storage_type
    if storage_type() != hist.storage.Weight():
        raise NotImplementedError(
            f"Histogram for sample {sample}, dataset {dataset} does not "
            "use Weight storage"
        )
    variation_index = axis_names.index("variation")
    variation_axis = histogram.axes["variation"]
    up_name = f"{variation_name}Up"
    down_name = f"{variation_name}Down"
    if up_name in variation_axis or down_name in variation_axis:
        raise ValueError(
            f"Variation {variation_name!r} already present in histogram "
            f"for sample {sample}, dataset {dataset}"
        )
    if "nominal" not in variation_axis:
        raise ValueError(
            f"No 'nominal' variation in histogram for sample {sample}, "
            f"dataset {dataset}"
        )

    new_variation_axis = hist.axis.StrCategory(
        list(variation_axis) + [up_name, down_name],
        name="variation",
        label=variation_axis.label,
    )
    new_axes = list(histogram.axes)
    new_axes[variation_index] = new_variation_axis
    new_histogram = hist.Hist(*new_axes, name=histogram.name, storage=storage_type())

    # Copy the existing variations (same order, first entries of the new
    # axis). Categorical axes have a trailing overflow slot in the
    # flow=True view, so slice the variation axis on both sides to drop
    # it; the (empty) overflow slots of the new view stay zero.
    copy_index = [slice(None)] * histogram.ndim
    copy_index[variation_index] = slice(0, len(variation_axis))
    copy_index = tuple(copy_index)
    new_histogram.view(flow=True)[copy_index] = histogram.view(flow=True)[copy_index]
    return new_histogram


def _combine_source_variations(
    histograms, variation_name, samples, source_variations, datasets, combine
):
    """Shared driver of :func:`add_envelope_variation` and
    :func:`add_quadrature_variation`: for each selected histogram, call
    ``combine(sources, nominal)`` with the ``value``/``variance`` views of the
    source variations (stacked on axis 0) and of the nominal, and store the
    returned ``(up, down)`` views as ``{variation_name}Up/Down``."""
    if len(source_variations) == 0:
        raise ValueError(f"No source variations given for {variation_name!r}")
    selected = _select_datasets(histograms, samples, datasets)

    new_histograms = dict(histograms)
    for sample in set(samples):
        new_histograms[sample] = {}
        for dataset, histogram in histograms[sample].items():
            if (sample, dataset) not in selected:
                new_histograms[sample][dataset] = histogram
                continue
            new_histogram = _with_extended_variation_axis(
                histogram, variation_name, sample, dataset
            )
            variation_axis = histogram.axes["variation"]
            missing_sources = [v for v in source_variations if v not in variation_axis]
            if missing_sources:
                raise ValueError(
                    f"Source variations {missing_sources} not found in histogram for "
                    f"sample {sample}, dataset {dataset} "
                    f"(available: {list(variation_axis)})"
                )
            variation_index = [axis.name for axis in histogram.axes].index("variation")
            # Move the variation axis first: view[i] is then variation i
            old_view = np.moveaxis(histogram.view(flow=True), variation_index, 0)
            new_view = np.moveaxis(new_histogram.view(flow=True), variation_index, 0)
            sources = old_view[[variation_axis.index(v) for v in source_variations]]
            nominal = old_view[variation_axis.index("nominal")]

            up, down = combine(sources, nominal)
            new_variation_axis = new_histogram.axes["variation"]
            new_view[new_variation_axis.index(f"{variation_name}Up")] = up
            new_view[new_variation_axis.index(f"{variation_name}Down")] = down
            new_histograms[sample][dataset] = new_histogram

    return new_histograms


def add_binwise_variation(
    histograms: dict[str, dict[str, hist.Hist]],
    variation_name: str,
    samples: list[str],
    weights_by_category: dict,
    default_weight=1.0,
    down_mode: str = "nominal",
    preserve_norm: bool = False,
    datasets: list[str] | None = None,
) -> dict[str, dict[str, hist.Hist]]:
    """Add an artificial per-bin reweighting variation to the variation axis.

    For every histogram of the requested ``samples``, two new entries are
    appended to the ``variation`` axis:

    - ``{variation_name}Up``: the nominal content multiplied, in each category,
      bin by bin along the (last) variable axis by the weights given in
      ``weights_by_category`` (variances scale with the squared weight).
    - ``{variation_name}Down``: by default an exact copy of the nominal
      (``down_mode="nominal"``, i.e. a one-sided variation). With
      ``down_mode="mirror"`` the nominal is instead divided by the same weights
      (log-mirror, matching the lnN convention; requires strictly positive
      weights).

    The weights refer to the histogram's own binning **before** any
    ``bins_edges`` rebinning done by the ``Datacard`` (rebinning then merges the
    reweighted contents). See :func:`_resolve_bin_weights` for the accepted
    weight formats: scalar (all bins, flow included), in-range array (flow
    untouched), or flow-inclusive array. Categories not listed get
    ``default_weight`` (default 1.0: Up = nominal there).

    Samples not listed in ``samples`` are left untouched (their histograms keep
    the original variation axis; :meth:`Datacard.rearrange_histograms` falls
    back to the nominal for them). The input dictionary is not modified: a new
    outer dict is returned, sharing the untouched histogram objects with the
    input.

    Typical use — reweight the ttbb shape as a function of the fit variable in
    the signal region only::

        histograms = add_binwise_variation(
            histograms,
            variation_name="ttbb_shape_rw",
            samples=[s for s in histograms if s.startswith("TTBB")],
            weights_by_category={"SR": weights_array},  # len = n bins of the variable
        )

    and declare ``SystematicUncertainty(name="ttbb_shape_rw", typ="shape", ...)``.

    .. note::
       If the affected process has a free rateParam and the datacard is built
       with ``shape_only_for_rateparam=True``, the overall yield change of the
       reweighting is removed with the common factor summed over all
       ``rateparam_norm_categories``: only the bin-by-bin shape and the
       relative differences between categories survive.

    .. warning::
       When several per-category datacards are combined, the shape-only factor
       is only identical across cards if the varied *total yield per category*
       is the same in every card's variable — automatic for scalar weights, but
       for per-bin weights it requires the arrays applied to the different
       variables to represent the same underlying event-level reweighting. Use
       ``preserve_norm=True`` to sidestep this entirely: the varied totals then
       equal the nominal ones per category everywhere, the rateParam factor is
       exactly 1 in every card, and the variation need not be injected into the
       other cards' variables at all (they fall back to nominal, which is the
       same thing). Otherwise, if the weights are known only versus one
       observable, apply at least the resulting per-category yield ratio (a
       scalar) to the other cards' histograms so the category totals agree.

    :param histograms: pocket_coffea histograms, ``{sample: {dataset: hist.Hist}}``
        with axes ``(cat, variation, ..., variable)``; weights act on the last axis.
    :param variation_name: name of the new variation; entries ``{name}Up`` and
        ``{name}Down`` are created. Must match the ``name`` (or
        ``coffea_name_alias``) of the SystematicUncertainty declared for it.
    :param samples: samples to which the variation is added. Must all be present
        in ``histograms``.
    :param weights_by_category: mapping ``category -> weights`` (scalar or
        per-bin array) applied to the nominal to build the Up template. Every
        key must exist on the ``cat`` axis of the histograms.
    :param default_weight: weight specification for categories not listed in
        ``weights_by_category``, defaults to 1.0.
    :param down_mode: ``"nominal"`` (default, one-sided: Down = nominal) or
        ``"mirror"`` (Down = nominal / weights).
    :param preserve_norm: if True, rescale each varied template — per dataset
        and per category, flow bins included — so its total yield matches the
        nominal one, making the variation a pure within-category shape effect
        with no normalization component (the common rescaling is applied to all
        bins, flow included, like an event-level weight would be). Guarantees
        yield neutrality exactly for every dataset, which hand-tuned weight
        arrays cannot (neutrality depends on the nominal distribution they
        multiply). Scalar weight entries then become no-ops by construction.
        Defaults to False.
    :param datasets: optionally restrict the injection to these datasets
        (within the selected samples); histograms of other datasets are carried
        over untouched. Use it to apply different weights per year by calling
        the function once per year with that year's datasets::

            for year, weights in weights_by_year.items():
                year_datasets = [
                    d
                    for s in samples
                    for d in datasets_metadata["by_datataking_period"][year].get(s, [])
                ]
                histograms = add_binwise_variation(
                    histograms, "rw", samples, {"SR": weights},
                    preserve_norm=True, datasets=year_datasets,
                )

        Repeated calls with the same ``variation_name`` are allowed as long as
        they touch disjoint datasets (overlaps raise "already present").
        Datasets in the list that are absent from the histogram dicts are
        ignored (e.g. empty datasets), but if nothing matches at all a
        ValueError is raised. Datasets never covered by any call simply fall
        back to nominal in the Datacard. Defaults to None (all datasets).
    :return: new ``{sample: {dataset: hist.Hist}}`` dict with the extended
        variation axis on the requested samples.
    """
    if down_mode not in ("nominal", "mirror"):
        raise ValueError(
            f"Unknown down_mode {down_mode!r}, expected 'nominal' or 'mirror'"
        )

    selected = _select_datasets(histograms, samples, datasets)

    up_name = f"{variation_name}Up"
    down_name = f"{variation_name}Down"

    new_histograms = dict(histograms)
    for sample in set(samples):
        new_histograms[sample] = {}
        for dataset, histogram in histograms[sample].items():
            if (sample, dataset) not in selected:
                # carried over untouched; a later call (same variation_name,
                # different weights) can cover it
                new_histograms[sample][dataset] = histogram
                continue
            new_histogram = _with_extended_variation_axis(
                histogram, variation_name, sample, dataset
            )
            axis_names = [axis.name for axis in histogram.axes]
            cat_index = axis_names.index("cat")
            variation_index = axis_names.index("variation")
            cat_axis = histogram.axes["cat"]
            variation_axis = histogram.axes["variation"]
            new_variation_axis = new_histogram.axes["variation"]
            variable_axis = histogram.axes[-1]

            unknown_categories = sorted(set(weights_by_category) - set(cat_axis))
            if unknown_categories:
                raise ValueError(
                    f"Categories {unknown_categories} not found on the 'cat' axis "
                    f"of sample {sample}, dataset {dataset} "
                    f"(available: {list(cat_axis)})"
                )

            old_view = histogram.view(flow=True)
            new_view = new_histogram.view(flow=True)

            def indexer(cat_i, variation_i, ndim=old_view.ndim):
                index = [slice(None)] * ndim
                index[cat_index] = cat_i
                index[variation_index] = variation_i
                return tuple(index)

            nominal_i = variation_axis.index("nominal")
            up_i = new_variation_axis.index(up_name)
            down_i = new_variation_axis.index(down_name)
            for category in cat_axis:
                #breakpoint()
                weights = _resolve_bin_weights(
                    weights_by_category.get(category, default_weight),
                    variable_axis,
                    category,
                )
                if down_mode == "mirror" and np.any(weights <= 0):
                    raise ValueError(
                        f"down_mode='mirror' requires positive weights, got "
                        f"minimum {weights.min()} for category {category!r}"
                    )
                cat_i = cat_axis.index(category)
                # weights broadcast along the last (variable) axis
                nominal = old_view[indexer(cat_i, nominal_i)]

                def effective_weights(raw_weights):
                    """Per-bin weights, rescaled to preserve the category yield
                    of this dataset (flow included) when preserve_norm is on."""
                    if not preserve_norm:
                        return raw_weights
                    varied_total = (nominal["value"] * raw_weights).sum()
                    nominal_total = nominal["value"].sum()
                    if varied_total == 0:
                        return raw_weights
                    return raw_weights * (nominal_total / varied_total)

                up_weights = effective_weights(weights)
                up = new_view[indexer(cat_i, up_i)]
                up["value"] = nominal["value"] * up_weights
                up["variance"] = nominal["variance"] * up_weights**2
                down = new_view[indexer(cat_i, down_i)]
                if down_mode == "nominal":
                    down["value"] = nominal["value"]
                    down["variance"] = nominal["variance"]
                else:
                    down_weights = effective_weights(1.0 / weights)
                    down["value"] = nominal["value"] * down_weights
                    down["variance"] = nominal["variance"] * down_weights**2

            new_histograms[sample][dataset] = new_histogram

    return new_histograms


def add_norm_variation(
    histograms: dict[str, dict[str, hist.Hist]],
    variation_name: str,
    samples: list[str],
    scale_by_category: dict[str, float],
    default_scale: float = 1.0,
    down_mode: str = "nominal",
    datasets: list[str] | None = None,
) -> dict[str, dict[str, hist.Hist]]:
    """Add an artificial per-category normalization variation to the variation axis.

    Special case of :func:`add_binwise_variation` with one scalar factor per
    category: ``{variation_name}Up`` is the nominal scaled by
    ``scale_by_category[category]`` in every bin of that category (under/overflow
    included; ``default_scale`` for categories not listed), so the bin-by-bin
    shape within a category is untouched. ``{variation_name}Down`` is the
    nominal itself by default (one-sided), or nominal divided by the factor with
    ``down_mode="mirror"``.

    Typical use — model a 4FS vs 5FS normalization difference on the ttbb
    samples::

        histograms = add_norm_variation(
            histograms,
            variation_name="4Fvs5F",
            samples=[s for s in histograms if s.startswith("TTBB")],
            scale_by_category={"CR": 1.0081, "SR": 0.9143, "CR_ttlf": 1.1309},
        )

    and declare ``SystematicUncertainty(name="4Fvs5F", typ="shape", ...)``.

    .. note::
       If the affected process has a free rateParam and the datacard is built
       with ``shape_only_for_rateparam=True``, the overall normalization of this
       variation is absorbed by the rateParam rescaling: only the *relative*
       differences between the per-category factors survive. Equal factors in
       all fit categories then make the variation a no-op.

    See :func:`add_binwise_variation` for the parameter documentation.
    """
    for category, scale in scale_by_category.items():
        if np.ndim(scale) != 0:
            raise ValueError(
                f"scale_by_category[{category!r}] must be a scalar, got "
                f"{scale!r}; use add_binwise_variation for per-bin weights"
            )
    return add_binwise_variation(
        histograms,
        variation_name=variation_name,
        samples=samples,
        weights_by_category=scale_by_category,
        default_weight=default_scale,
        down_mode=down_mode,
        datasets=datasets,
    )


def add_envelope_variation(
    histograms: dict[str, dict[str, hist.Hist]],
    variation_name: str,
    samples: list[str],
    source_variations: list[str],
    include_nominal: bool = False,
    datasets: list[str] | None = None,
) -> dict[str, dict[str, hist.Hist]]:
    """Add the bin-by-bin envelope of several existing variations as a new variation.

    ``{variation_name}Up`` is, in every bin (flow included) and category, the
    maximum over ``source_variations`` and ``{variation_name}Down`` the minimum;
    each bin keeps the variance of the variation it was taken from. With
    ``include_nominal=True`` the nominal takes part in the envelope, so that
    Down <= nominal <= Up everywhere.

    Typical use — the 7-point QCD scale envelope from the ``lhe_scale`` weight
    (:class:`pocket_coffea.lib.weights.common.common.LHEScaleVariations`)::

        histograms = add_envelope_variation(
            histograms,
            variation_name="CMS_bbbb_hadronic_QCDScaleacc_ggHH",
            samples=ggHH_samples,
            source_variations=[
                f"lhe_scale_{var}{shift}"
                for var in ["muR", "muF", "muRmuF"]
                for shift in ["Up", "Down"]
            ],
        )

    and declare ``SystematicUncertainty(name="CMS_bbbb_hadronic_QCDScaleacc_ggHH", typ="shape", ...)``.

    :param histograms: pocket_coffea histograms, ``{sample: {dataset: hist.Hist}}``.
    :param variation_name: name of the new variation (``{name}Up``/``{name}Down``).
    :param samples: samples to which the variation is added.
    :param source_variations: existing entries of the ``variation`` axis to
        build the envelope from (full names, including ``Up``/``Down``). They
        must exist in every selected histogram.
    :param include_nominal: also include the nominal in the envelope. Defaults to False.
    :param datasets: optionally restrict to these datasets, see
        :func:`add_binwise_variation`.
    :return: new ``{sample: {dataset: hist.Hist}}`` dict.
    """
    if include_nominal:
        source_variations = list(source_variations) + ["nominal"]

    def envelope(sources, nominal):
        values = sources["value"]
        variances = sources["variance"]
        up = np.empty_like(nominal)
        down = np.empty_like(nominal)
        for out, chosen in ((up, values.argmax(axis=0)), (down, values.argmin(axis=0))):
            out["value"] = np.take_along_axis(values, chosen[np.newaxis], axis=0)[0]
            out["variance"] = np.take_along_axis(variances, chosen[np.newaxis], axis=0)[0]
        return up, down

    return _combine_source_variations(
        histograms, variation_name, samples, source_variations, datasets, envelope
    )


def add_quadrature_variation(
    histograms: dict[str, dict[str, hist.Hist]],
    variation_name: str,
    samples: list[str],
    source_variations: list[str],
    datasets: list[str] | None = None,
) -> dict[str, dict[str, hist.Hist]]:
    """Add the quadrature sum of the deviations of several variations as a new variation.

    In every bin (flow included) and category the deviation
    ``delta = sqrt(sum_i (source_i - nominal)**2)`` is computed, and
    ``{variation_name}Up = nominal + delta``,
    ``{variation_name}Down = nominal - delta`` (negative bins are left to the
    clipping of the ``Datacard``). Both keep the nominal variances.
    This is the prescription for the eigenvectors of a Hessian PDF set.

    Typical use — the PDF uncertainty from the ``lhe_pdf`` weight
    (:class:`pocket_coffea.lib.weights.common.common.LHEPdfVariations`)::

        histograms = add_quadrature_variation(
            histograms,
            variation_name="CMS_bbbb_hadronic_PDFacc_ggHH",
            samples=ggHH_samples,
            source_variations=[f"lhe_pdf_eigenvector{i}Up" for i in range(1, 101)],
        )

    and declare ``SystematicUncertainty(name="CMS_bbbb_hadronic_PDFacc_ggHH", typ="shape", ...)``.

    :param histograms: pocket_coffea histograms, ``{sample: {dataset: hist.Hist}}``.
    :param variation_name: name of the new variation (``{name}Up``/``{name}Down``).
    :param samples: samples to which the variation is added.
    :param source_variations: existing entries of the ``variation`` axis whose
        deviations from the nominal are summed in quadrature (full names,
        including ``Up``/``Down``). They must exist in every selected histogram.
    :param datasets: optionally restrict to these datasets, see
        :func:`add_binwise_variation`.
    :return: new ``{sample: {dataset: hist.Hist}}`` dict.
    """

    def quadrature(sources, nominal):
        delta = np.sqrt(((sources["value"] - nominal["value"]) ** 2).sum(axis=0))
        up = nominal.copy()
        down = nominal.copy()
        up["value"] = nominal["value"] + delta
        down["value"] = nominal["value"] - delta
        return up, down

    return _combine_source_variations(
        histograms, variation_name, samples, source_variations, datasets, quadrature
    )


def rescale_histograms(
    histograms: dict[str, dict[str, hist.Hist]],
    samples: list[str],
    scale: float,
    datasets: list[str] | None = None,
) -> dict[str, dict[str, hist.Hist]]:
    """Rescale the normalization of whole histograms by a constant factor.

    Every histogram of the requested ``samples`` (optionally restricted to
    ``datasets``) is multiplied by ``scale`` in all its bins: every category,
    the nominal *and* every variation, under/overflow included. Bin variances
    scale with ``scale**2``. Unlike :func:`add_norm_variation` no variation
    entry is created: the histograms themselves change, so the effect is a
    plain change of the sample yield (e.g. a k-factor or a cross-section fix
    for one year's datasets).

    Typical use — apply a per-year factor to a sample::

        for year, factor in factor_by_year.items():
            year_datasets = datasets_metadata["by_datataking_period"][year].get(
                "TTBB", []
            )
            histograms = rescale_histograms(
                histograms, ["TTBB"], factor, datasets=year_datasets
            )

    The input dictionary is not modified: a new outer dict is returned,
    sharing the untouched histogram objects with the input.

    :param histograms: pocket_coffea histograms, ``{sample: {dataset: hist.Hist}}``.
    :param samples: samples to rescale. Must all be present in ``histograms``.
    :param scale: scalar multiplied into every bin (variances by its square).
    :param datasets: optionally restrict the rescaling to these datasets
        (within the selected samples); absent datasets are ignored, but a
        list matching nothing raises. Defaults to None (all datasets).
    :return: new ``{sample: {dataset: hist.Hist}}`` dict.
    """
    if np.ndim(scale) != 0:
        raise ValueError(f"scale must be a scalar, got {scale!r}")
    selected = _select_datasets(histograms, samples, datasets)
    new_histograms = dict(histograms)
    for sample in set(samples):
        new_histograms[sample] = {
            dataset: histogram * scale if (sample, dataset) in selected else histogram
            for dataset, histogram in histograms[sample].items()
        }
    return new_histograms
