"""
Shared configuration for the EEG pipeline.
"""

import csv
import os

# paths ------------------------------------------------------------------------
DATA_ROOT = os.environ.get("PED_EEG_DATA", "/home/jure/work/ped_neuro_eeg_data")
BIDS_ROOT = os.path.join(DATA_ROOT, "bids")
DERIV_ROOT = os.path.join(BIDS_ROOT, "derivatives", "clean")
RAW_GROUPS = ["test", "control"]

# repository paths, independent of the working directory
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, "data")
CONNECTOME_DIR = os.path.join(DATA_DIR, "connectomes")
QC_DIR = os.path.join(REPO_ROOT, "qc")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# study sample -----------------------------------------------------------------
# children aged 2-9 years in completed years, the age range of the main and key
# secondary analyses; recordings outside it are not analysed (amendment of
# 24 October 2026, see analysis_spec.md)
AGE_RANGE = (2, 9)


def study_ids() -> set[str]:
    """
    IDs of the participants in the study age range (data/demographics_*.csv).
    """
    ids = set()
    for group in RAW_GROUPS:
        with open(os.path.join(DATA_DIR, f"demographics_{group}.csv"), encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if AGE_RANGE[0] <= int(row["age"]) <= AGE_RANGE[1]:
                    ids.add(row["id"])
    return ids

# reproducibility ---------------------------------------------------------------
SEED = 42

# recording --------------------------------------------------------------------
SFREQ = 250.0
# the exported text files are in mV, but were stored in BIDS as if in V
UNIT_SCALE = 1e-3

SCALP_CHANNELS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
]
EAR_CHANNELS = ["A1", "A2"]
ECG_CHANNEL = "ECG"

# segment selection ------------------------------------------------------------
EPOCH_DURATION = 1.0
N_EPOCHS = 100
MIN_EPOCHS = 30
# Autoreject is fitted on the candidate epochs nearest the recording midpoint
CANDIDATE_POOL = 400
EDGE_EXCLUDE_S = 30.0
# margin around breaks: at least half the length of the band-pass FIR filter
BREAK_MARGIN_S = 3.5
PHOTIC_MARGIN_S = 60.0
# hyperventilation (detected from the respiration channel) is excluded together
# with the 2 minutes after it
HV_MARGIN_AFTER_S = 120.0

# filtering --------------------------------------------------------------------
L_FREQ = 1.0
H_FREQ = 40.0
ICA_L_FREQ = 1.0

# bad channels (fixed before any group comparison) -----------------------------
MAX_BAD_CHANNELS = 3

# vigilance --------------------------------------------------------------------
SPINDLE_FREQ = (12, 15)
SPINDLE_CHANNELS = ["C3", "C4", "Cz"]
SPINDLE_RATE_FLAG = 0.5  # spindles per minute

# connectivity -----------------------------------------------------------------
MT_BANDWIDTH = 2.0
BANDS = {
    "alpha": (8.0, 13.0),
    "theta": (4.0, 7.0),
}
IAF_HALF_WIDTH = 2.0
IAF_FLOOR = 4.0

# posterior dominant rhythm ----------------------------------------------------
POSTERIOR_CHANNELS = ["O1", "O2", "P3", "P4", "Pz"]
PDR_RANGE = (5.0, 13.0)
PDR_MIN_HEIGHT_DB = 1.0
# peaks wider than this are not accepted as a PDR
PDR_MAX_WIDTH = 5.0
# search window around the age-expected PDR (Hz)
PDR_WINDOW = 3.0
# age-expected peak alpha frequency (meta-analysis of Freschl et al. 2022):
# (age in years, frequency in Hz), linearly interpolated, constant outside
PDR_EXPECTED = [(0.5, 6.1), (5.0, 8.4), (13.0, 9.7), (18.0, 10.1)]

# electrode groups -------------------------------------------------------------
LEFT_CHANNELS = ["Fp1", "F7", "F3", "T3", "C3", "T5", "P3", "O1"]
RIGHT_CHANNELS = ["Fp2", "F8", "F4", "T4", "C4", "T6", "P4", "O2"]
LAQ_CHANNELS = ["Fp1", "F3", "F7"]
LPQ_CHANNELS = ["P3", "O1", "T5"]
RAQ_CHANNELS = ["Fp2", "F4", "F8"]
RPQ_CHANNELS = ["P4", "O2", "T6"]
# frontal-pole and anterior temporal electrodes: ocular references for ICA and
# the electrodes left out of interhemispheric strength in S11
FRONTAL_CHANNELS = ["Fp1", "Fp2", "F7", "F8"]
