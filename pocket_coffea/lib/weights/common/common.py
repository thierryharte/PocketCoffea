from ..weights import WeightWrapper, WeightLambda, WeightData, WeightDataMultiVariation
import logging
import numpy as np
import awkward as ak

from pocket_coffea.lib.scale_factors import (
    sf_ele_reco,
    sf_ele_id,
    sf_ele_promptmva,
    sf_photon,
    sf_ele_trigger,
    sf_mu,
    sf_mu_promptmva,
    sf_btag,
    sf_btag_calib,
    sf_ctag,
    sf_ctag_calib,
    sf_jet_puId,
    sf_L1prefiring,
    sf_pileup_reweight,
    sf_partonshower_isr,
    sf_partonshower_fsr
)


### Simple WeightWrappers from functions defined in the lib

genWeight = WeightLambda.wrap_func(
    name="genWeight",
    function=lambda params, metadata, events, size, shape_variations:
            events.genWeight,
    has_variations=False
    )

signOfGenWeight = WeightLambda.wrap_func(
    name="signOf_genWeight",
    function=lambda params, metadata, events, size, shape_variations:
       np.sign(events.genWeight),
    has_variations=False
    )

lumi = WeightLambda.wrap_func(
    name="lumi",
    function=lambda params, metadata, events, size, shape_variations:
        np.ones(len(events)) * params.lumi.picobarns[metadata["year"]]["tot"],
    has_variations=False
    )


XS = WeightLambda.wrap_func(
    name = "XS",
    function = lambda params, metadata, events, size, shape_variations:
        np.ones(len(events)) * float(metadata["xsec"]),
    has_variations=False
)


pileup = WeightLambda.wrap_func(
    name="pileup",
    function=lambda params, metadata, events, size, shape_variations:
        sf_pileup_reweight(params, events, metadata["year"]),
    has_variations=True  # no list of variations it means only up and down
    )


SF_ele_reco = WeightLambda.wrap_func(
    name="sf_ele_reco",
    function=lambda params, metadata, events, size, shape_variations:
        sf_ele_reco(params, events, metadata["year"]),
    has_variations=True
    )

SF_ele_id = WeightLambda.wrap_func(
    name="sf_ele_id",
    function=lambda params, metadata, events, size, shape_variations:
        sf_ele_id(params, events, metadata["year"]),
    has_variations=True
    )

SF_ele_promptMVA = WeightLambda.wrap_func(
    name="sf_ele_promptMVA",
    function=lambda params, metadata, events, size, shape_variations:
        sf_ele_promptmva(params, events, metadata["year"]),
    has_variations=True
    )

SF_ele_trigger = WeightLambda.wrap_func(
    name="sf_ele_trigger",
    function=lambda params, metadata, events, size, shape_variations: sf_ele_trigger(
        params, events, metadata["year"]
    ),
    has_variations=True,
)

SF_mu_id = WeightLambda.wrap_func(
    name="sf_mu_id",
    function=lambda params, metadata, events, size, shape_variations:
        sf_mu(params, events, metadata["year"], 'id'),
    has_variations=True
    )

SF_mu_iso = WeightLambda.wrap_func(
    name="sf_mu_iso",
    function=lambda params, metadata, events, size, shape_variations:
        sf_mu(params, events, metadata["year"], 'iso'),
    has_variations=True
    )

SF_mu_trigger = WeightLambda.wrap_func(
    name="sf_mu_trigger",
    function=lambda params, metadata, events, size, shape_variations:
        sf_mu(params, events, metadata["year"], 'trigger'),
    has_variations=True
    )

SF_mu_promptMVA = WeightLambda.wrap_func(
    name="sf_mu_promptMVA",
    function=lambda params, metadata, events, size, shape_variations:
        sf_mu_promptmva(params, events, metadata["year"], 'promptMVA'),
    has_variations=True
    )


SF_L1prefiring = WeightLambda.wrap_func(
    name="sf_L1prefiring",
    function=lambda params, metadata, events, size, shape_variations:
        sf_L1prefiring(events),
    has_variations=True
    )


SF_PSWeight_isr = WeightLambda.wrap_func(
    name="sf_partonshower_isr",
    function=lambda params, metadata, events, size, shape_variations:
        sf_partonshower_isr(events),
    has_variations=True
    )

SF_PSWeight_fsr = WeightLambda.wrap_func(
    name="sf_partonshower_fsr",
    function=lambda params, metadata, events, size, shape_variations:
        sf_partonshower_fsr(events),
    has_variations=True
    )


SF_pho_pxseed = WeightLambda.wrap_func(
    name="sf_pho_pxseed",
    function=lambda params, metadata, events, size, shape_variations:
        sf_photon(params, events, metadata["year"], 'pxseed'),
    has_variations=True
    )

SF_pho_id = WeightLambda.wrap_func(
    name="sf_pho_id",
    function=lambda params, metadata, events, size, shape_variations:
        sf_photon(params, events, metadata["year"], 'id'),
    has_variations=True
    )

########################################
# Btag scale factors have weights depending on the shape_variation

class SF_btag(WeightWrapper):
    name = "sf_btag"
    has_variations = True

    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        # Getting the variations from the parameters depending on the year
        self._variations = params.systematic_variations.weight_variations.sf_btag[metadata["year"]]
        self.jet_coll = params.jet_scale_factors.jet_collection.btag

    def compute(self, events, size, shape_variation):
        
        if shape_variation == "nominal":
            out = sf_btag(self._params,
                          events[self.jet_coll],
                          self._metadata["year"],
                          # Assuming n*JetCollection* is defined
                          njets=events[f"n{self.jet_coll}"],
                          variations=["central"] + self._variations,
                          )
            # This is a dict with variation: [nom, up, down]
            return WeightDataMultiVariation(
                name = self.name,
                nominal = out["central"][0],
                variations = self._variations,
                up = [out[var][1] for var in self._variations],
                down = [out[var][2] for var in self._variations]
            )


        elif shape_variation.startswith("JES"):
            out = sf_btag(self._params,
                                events[self.jet_coll],
                                self._metadata["year"],
                                # Assuming n*JetCollection* is defined
                                njets=events[f"n{self.jet_coll}"],
                                variations=[shape_variation],
                                )
            return WeightData(
                name = self.name,
                nominal = out["central"][0]
                )       
            
        else:
            out = sf_btag(self._params,
                          events[self.jet_coll],
                          self._metadata["year"],
                          # Assuming n*JetCollection* is defined
                          njets=events[f"n{self.jet_coll}"],
                          variations=["central"],
                          )
            return WeightData(
                name = self.name,
                nominal = out["central"][0]
            )

class SF_btag_calib(WeightWrapper):
    name = "sf_btag_calib"
    has_variations = False

    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        self.jet_coll = params.jet_scale_factors.jet_collection.btag

    def compute(self, events, size, shape_variation):
        jetsHt = ak.sum(events[self.jet_coll].pt, axis=1)
        out = sf_btag_calib(self._params,
                            sample=self._metadata["sample"],
                            year=self._metadata["year"],
                            # Assuming n*JetCollection* is defined
                            njets=events[f"n{self.jet_coll}"],
                            jetsHt=jetsHt
                            )
        return WeightData(
            name = self.name,
            nominal = out, #out[0] only if has_variations = True
            #up = out[1],
            #down = out[2]
            )


########################################
# Ctag scale factors have weights depending on the shape_variation

class SF_ctag(WeightWrapper):
    name = "sf_ctag"
    has_variations = True

    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        # Getting the variations from the parameters depending on the year
        self._variations = params.systematic_variations.weight_variations.sf_ctag[metadata["year"]]
        self.jet_coll = params.jet_scale_factors.jet_collection.ctag

    def compute(self, events, size, shape_variation):
        
        if shape_variation == "nominal":
            out = sf_ctag(self._params,
                          events[self.jet_coll],
                          self._metadata["year"],
                          # Assuming n*JetCollection* is defined
                          njets=events[f"n{self.jet_coll}"],
                          variations=["central"] + self._variations,
                          )
            # This is a dict with variation: [nom, up, down]
            return WeightDataMultiVariation(
                name = self.name,
                nominal = out["central"][0],
                variations = self._variations,
                up = [out[var][1] for var in self._variations],
                down = [out[var][2] for var in self._variations]
            )
        else:
            #only the nominal weight for shape variations != nominal
            out = sf_ctag(self._params,
                          events[self.jet_coll],
                          self._metadata["year"],
                          # Assuming n*JetCollection* is defined
                          njets=events[f"n{self.jet_coll}"],
                          variations=["central"],
                          )
            return WeightData(
                name = self.name,
                nominal = out["central"][0]
            )

class SF_ctag_calib(WeightWrapper):
    name = "sf_ctag_calib"
    has_variations = True

    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        self.jet_coll = params.jet_scale_factors.jet_collection.ctag

    def compute(self, events, size, shape_variation):
        jetsHt = ak.sum(events[self.jet_coll].pt, axis=1)
        out = sf_ctag_calib(self._params,
                            dataset=self._metadata["dataset"],
                            year=self._metadata["year"],
                            # Assuming n*JetCollection* is defined
                            njets=events[f"n{self.jet_coll}"],
                            jetsHt=jetsHt
                            )
        return WeightData(
            name = self.name,
            nominal = out[0],
            up = out[1],
            down = out[2]
            )

#############################################################
class SF_jet_puId(WeightWrapper):
    name = "sf_jet_puId"
    has_variations = True

    def __init__(self, params, metadata):
        super().__init__(params, metadata)
        self.jet_coll = params.jet_scale_factors.jet_collection.puId

    def compute(self, events, size, shape_variation):
        out = sf_jet_puId(self._params,
                          events[self.jet_coll],
                          year=self._metadata["year"],
                          # Assuming n*JetCollection* is defined
                          njets=events[f"n{self.jet_coll}"],
                          )
        return WeightData(
            name = self.name,
            nominal = out[0],
            up = out[1],
            down = out[2]
            )


#############################################################
# LHE theory weights: QCD scale (muR, muF) and PDF variations.
# NanoAOD stores them per event as w_var / w_nominal. The variations can be
# normalised to the pre-skim inclusive effect of each variation, so that only the
# acceptance/shape effect is kept (the cross-section effect is usually covered by
# separate normalisation nuisances, e.g. QCDscale_ggHH, pdf_Higgs_ggHH).
# The inclusive ratios are read from the dataset metadata keys "LHEScaleSumw" and
# "LHEPdfSumw", which have the same definition as the NanoAOD Runs branches:
# sum(genWeight * w_var) / sum(genWeight) over the full (unskimmed) dataset.

# LHEScaleWeight index = 3 * i_muR + i_muF, with i = 0, 1, 2 for a factor 0.5, 1, 2
LHE_SCALE_NOMINAL_INDEX = 4
# variation name -> (index of the Up variation, index of the Down variation)
LHE_SCALE_VARIATION_INDICES = {
    "muR": (7, 1),
    "muF": (5, 3),
    "muRmuF": (8, 0),
}
# Hessian PDF sets: LHEPdfWeight[0] is the central member, [1..100] the eigenvectors.
# Further members (e.g. the alpha_s variations [101], [102]) are not used here.
LHE_PDF_N_EIGENVECTORS = 100


def get_inclusive_lhe_ratios(metadata, key):
    '''Return the inclusive LHE ratios stored in the dataset metadata under `key`
    ("LHEScaleSumw" or "LHEPdfSumw") as a numpy array, or None if missing.
    A comma-separated string is accepted as well as a list.'''
    if key not in metadata:
        return None
    inclusive_ratios = metadata[key]
    if isinstance(inclusive_ratios, str):
        inclusive_ratios = inclusive_ratios.split(",")
    return np.asarray(inclusive_ratios, dtype=float)


def _lhe_weights_as_numpy(events, branch, min_entries):
    weights_per_event = ak.to_numpy(ak.to_regular(events[branch], axis=1))
    if weights_per_event.shape[1] < min_entries:
        raise ValueError(
            f"{branch} has {weights_per_event.shape[1]} entries per event, "
            f"at least {min_entries} are needed"
        )
    return weights_per_event


class LHEScaleVariations(WeightWrapper):
    '''QCD renormalisation (muR) and factorisation (muF) scale variations from
    LHEScaleWeight, as three Up/Down pairs: muR, muF and muR+muF varied together
    (the anti-correlated points [2] and [6] are not used).
    The 7-point envelope is built from these 6 variations at the datacard step,
    see `pocket_coffea.utils.stat.shape_manipulation.add_envelope_variation`.

    If the dataset metadata contains "LHEScaleSumw", each variation is divided by
    its inclusive ratio, so that only the acceptance/shape effect is kept.
    The nominal weight is 1: this weight only adds variations.'''
    name = "lhe_scale"
    has_variations = True
    _variations = list(LHE_SCALE_VARIATION_INDICES)

    def compute(self, events, size, shape_variation):
        nominal = np.ones(size)
        if shape_variation != "nominal":
            return WeightData(name=self.name, nominal=nominal)

        scale_weights = _lhe_weights_as_numpy(events, "LHEScaleWeight", min_entries=9)
        inclusive_ratios = get_inclusive_lhe_ratios(events.metadata, "LHEScaleSumw")
        if inclusive_ratios is None:
            logging.warning(
                f"No LHEScaleSumw in the metadata of dataset {events.metadata['dataset']}: "
                "the lhe_scale variations are not normalised to the inclusive cross-section"
            )
            inclusive_ratios = np.ones(scale_weights.shape[1])

        def relative_to_nominal(index):
            per_event_ratio = scale_weights[:, index] / scale_weights[:, LHE_SCALE_NOMINAL_INDEX]
            inclusive_ratio = inclusive_ratios[index] / inclusive_ratios[LHE_SCALE_NOMINAL_INDEX]
            return per_event_ratio / inclusive_ratio

        return WeightDataMultiVariation(
            name=self.name,
            nominal=nominal,
            variations=self._variations,
            up=[relative_to_nominal(LHE_SCALE_VARIATION_INDICES[var][0]) for var in self._variations],
            down=[relative_to_nominal(LHE_SCALE_VARIATION_INDICES[var][1]) for var in self._variations],
        )


class LHEPdfVariations(WeightWrapper):
    '''Hessian PDF eigenvector variations from LHEPdfWeight, relative to the
    central member [0]: one variation per eigenvector ("eigenvector1" ...
    "eigenvector100"). The Up variation is the eigenvector member, the Down
    variation is the nominal (Hessian deviations are symmetrised at the datacard
    step by summing them in quadrature, see
    `pocket_coffea.utils.stat.shape_manipulation.add_quadrature_variation`).

    If the dataset metadata contains "LHEPdfSumw", each variation is divided by
    its inclusive ratio, so that only the acceptance/shape effect is kept.
    The nominal weight is 1: this weight only adds variations.'''
    name = "lhe_pdf"
    has_variations = True
    _variations = [f"eigenvector{i}" for i in range(1, LHE_PDF_N_EIGENVECTORS + 1)]

    def compute(self, events, size, shape_variation):
        nominal = np.ones(size)
        if shape_variation != "nominal":
            return WeightData(name=self.name, nominal=nominal)

        pdf_weights = _lhe_weights_as_numpy(events, "LHEPdfWeight", min_entries=LHE_PDF_N_EIGENVECTORS + 1)
        inclusive_ratios = get_inclusive_lhe_ratios(events.metadata, "LHEPdfSumw")
        if inclusive_ratios is None:
            logging.warning(
                f"No LHEPdfSumw in the metadata of dataset {events.metadata['dataset']}: "
                "the lhe_pdf variations are not normalised to the inclusive cross-section"
            )
            inclusive_ratios = np.ones(pdf_weights.shape[1])

        def relative_to_central(member):
            per_event_ratio = pdf_weights[:, member] / pdf_weights[:, 0]
            inclusive_ratio = inclusive_ratios[member] / inclusive_ratios[0]
            return per_event_ratio / inclusive_ratio

        return WeightDataMultiVariation(
            name=self.name,
            nominal=nominal,
            variations=self._variations,
            up=[relative_to_central(member) for member in range(1, LHE_PDF_N_EIGENVECTORS + 1)],
            down=[nominal] * LHE_PDF_N_EIGENVECTORS,
        )


# List with default classes for common weights
common_weights = [
    genWeight,
    signOfGenWeight,
    lumi,
    XS,
    pileup,
    SF_ele_reco,
    SF_ele_id,
    SF_ele_promptMVA,
    SF_ele_trigger,
    SF_pho_pxseed,
    SF_pho_id,
    SF_mu_id,
    SF_mu_iso,
    SF_mu_promptMVA,
    SF_mu_trigger,
    SF_btag,
    SF_btag_calib,
    SF_ctag,
    SF_ctag_calib,
    SF_jet_puId,
    SF_PSWeight_isr,
    SF_PSWeight_fsr,
    LHEScaleVariations,
    LHEPdfVariations,
]



