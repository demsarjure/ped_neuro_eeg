"""
Post hoc checks of the alpha-power measure, not part of the analysis plan
(25 October 2026). Higher 20-40 Hz power (muscle activity) could flatten the
aperiodic fit over 1.5-35 Hz (pdr.py) and so lower the aperiodic-adjusted alpha
power. Alpha power is therefore also computed with the aperiodic component fitted
over 1.5-20 Hz, and as absolute power, at the posterior electrodes.
"""

import os

import numpy as np
import pandas as pd
import mne
from fooof import FOOOF

from config import DERIV_ROOT, DATA_DIR, SCALP_CHANNELS, POSTERIOR_CHANNELS
from pdr import epoch_spectrum, ALPHA

mne.set_log_level("ERROR")

FIT_RANGE_SHORT = (1.5, 20.0)


def alpha_measures(sub_id: str) -> dict:
    """
    Posterior alpha power with the shorter aperiodic fit, and absolute alpha power.

    Args:
        sub_id (str): Subject identifier.
    """
    epochs = mne.read_epochs(
        os.path.join(DERIV_ROOT, f"sub-{sub_id}", "eeg", f"sub-{sub_id}_task-rest_desc-clean_epo.fif")
    )
    epochs.reorder_channels(SCALP_CHANNELS)
    f, psd = epoch_spectrum(epochs)
    posterior = psd[[epochs.ch_names.index(c) for c in POSTERIOR_CHANNELS]].mean(axis=0)

    fm = FOOOF(peak_width_limits=[1.0, 8.0], max_n_peaks=6, min_peak_height=0.0, aperiodic_mode="fixed", verbose=False)
    fm.fit(f, posterior, list(FIT_RANGE_SHORT))
    alpha = (fm.freqs >= ALPHA[0]) & (fm.freqs <= ALPHA[1])
    in_band = (f >= ALPHA[0]) & (f <= ALPHA[1])

    return {
        "id": sub_id,
        "alpha_adj_db_20hz": float(10 * fm._spectrum_flat[alpha].mean()),
        # absolute posterior alpha power, dB relative to 1 uV^2/Hz
        "alpha_abs_db": float(10 * np.log10(posterior[in_band].mean() / 1e-12)),
    }


if __name__ == "__main__":
    pdr = pd.read_csv(os.path.join(DATA_DIR, "pdr.csv"))
    out = pd.DataFrame([alpha_measures(i) for i in pdr["id"]]).merge(pdr[["id", "group"]], on="id")
    out.to_csv(os.path.join(DATA_DIR, "spectral_checks.csv"), index=False)
    print(out.groupby("group")[["alpha_adj_db_20hz", "alpha_abs_db"]].median().round(2))
