"""
Connectomes calculation script: debiased weighted phase lag index (dbWPLI)
between all pairs of the 19 scalp channels, estimated from the 100 kept 1-s
epochs of each recording with multitaper spectra (+/- 1 Hz smoothing).
"""

import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import mne
from mne_connectivity import spectral_connectivity_epochs

from config import (
    DERIV_ROOT,
    DATA_DIR,
    CONNECTOME_DIR,
    SCALP_CHANNELS,
    BANDS,
    MT_BANDWIDTH,
)

mne.set_log_level("ERROR")


def connectome(epochs: mne.Epochs, fmin: float, fmax: float) -> np.ndarray:
    """
    Band-averaged dbWPLI connectome.

    Args:
        epochs (mne.Epochs): Kept epochs.
        fmin (float): Lower band edge in Hz.
        fmax (float): Upper band edge in Hz.
    """
    result = spectral_connectivity_epochs(
        epochs,
        method="wpli2_debiased",
        mode="multitaper",
        mt_bandwidth=MT_BANDWIDTH,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        n_jobs=1,
        verbose=False,
    )
    con = result.get_data(output="dense")[:, :, 0]

    return con + con.T


def process_subject(row: dict):
    """
    Compute and store all connectomes of one recording.

    Args:
        row (dict): Subject id, group and individual band edges.
    """
    sub_id = row["id"]
    epochs = mne.read_epochs(
        os.path.join(
            DERIV_ROOT, f"sub-{sub_id}", "eeg", f"sub-{sub_id}_task-rest_desc-clean_epo.fif"
        )
    )
    epochs.reorder_channels(SCALP_CHANNELS)

    bands = dict(BANDS)
    bands["iaf"] = (row["iaf_low"], row["iaf_high"])

    for band, (fmin, fmax) in bands.items():
        conn_matrix = connectome(epochs, fmin, fmax)
        output_path = os.path.join(
            CONNECTOME_DIR, row["group"], band, f"{sub_id}_{band}_task-rest_connectome_eeg.csv"
        )
        np.savetxt(output_path, conn_matrix, delimiter=",")

    print(f"---> Connectomes saved for {sub_id}")


if __name__ == "__main__":
    pdr = pd.read_csv(os.path.join(DATA_DIR, "pdr.csv"))

    # start from an empty directory so that no stale connectomes remain
    if os.path.exists(CONNECTOME_DIR):
        shutil.rmtree(CONNECTOME_DIR)
    for group in ["test", "control"]:
        for band in list(BANDS) + ["iaf"]:
            os.makedirs(os.path.join(CONNECTOME_DIR, group, band), exist_ok=True)

    rows = pdr[["id", "group", "iaf_low", "iaf_high"]].to_dict("records")
    with ProcessPoolExecutor(max_workers=16) as executor:
        list(executor.map(process_subject, rows))

    print()
    print("---> All connectomes have been generated!")
