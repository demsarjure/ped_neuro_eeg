"""
Post hoc checks on the analysed epochs, not part of the analysis plan
(25 October 2026).

1. Epochs matched for alpha power. In each child, the 50 of the 100 analysed
   epochs with the lowest posterior alpha power (control children) or the highest
   (children with CTNNB1 syndrome) were kept, so that alpha power is closer
   between the groups in real data, and the alpha-band connectome was recomputed.
2. Frequency-resolved IHS and posterior power, 2-30 Hz in 1-Hz bins, for a
   descriptive figure.
"""

import os
import multiprocessing

import numpy as np
import pandas as pd
import mne
from joblib import Parallel, delayed
from mne_connectivity import spectral_connectivity_epochs
from scipy.signal import welch

from config import DERIV_ROOT, DATA_DIR, SCALP_CHANNELS, POSTERIOR_CHANNELS, LEFT_CHANNELS, RIGHT_CHANNELS, BANDS, SFREQ
from connectomes import connectome

mne.set_log_level("ERROR")

N_MATCHED = 50
FREQ_RANGE = (2.0, 30.0)


def ihs(con: np.ndarray) -> float:
    """
    Interhemispheric strength of a connectome (mean over the 64 left-right pairs).

    Args:
        con (np.ndarray): Connectome, channels x channels.
    """
    left = [SCALP_CHANNELS.index(c) for c in LEFT_CHANNELS]
    right = [SCALP_CHANNELS.index(c) for c in RIGHT_CHANNELS]
    return float(con[np.ix_(left, right)].mean())


def process_subject(sub_id: str, group: str) -> tuple[dict, pd.DataFrame]:
    """
    Alpha-matched IHS and the frequency-resolved IHS and power of one child.

    Args:
        sub_id (str): Subject identifier.
        group (str): "test" or "control".
    """
    epochs = mne.read_epochs(
        os.path.join(DERIV_ROOT, f"sub-{sub_id}", "eeg", f"sub-{sub_id}_task-rest_desc-clean_epo.fif")
    )
    epochs.reorder_channels(SCALP_CHANNELS)
    data = epochs.get_data()
    posterior = [SCALP_CHANNELS.index(c) for c in POSTERIOR_CHANNELS]

    # posterior alpha power of each epoch (dB relative to 1 uV^2/Hz)
    f, pxx = welch(data[:, posterior], fs=SFREQ, nperseg=data.shape[-1], nfft=2 * data.shape[-1])
    alpha = (f >= BANDS["alpha"][0]) & (f <= BANDS["alpha"][1])
    epoch_alpha = 10 * np.log10(pxx[:, :, alpha].mean(axis=(1, 2)) / 1e-12)

    order = np.argsort(epoch_alpha)
    keep = order[:N_MATCHED] if group == "control" else order[-N_MATCHED:]
    matched = epochs[np.sort(keep)]

    # frequency-resolved dbWPLI and posterior power (2-s Welch over the epochs)
    result = spectral_connectivity_epochs(
        epochs, method="wpli2_debiased", mode="multitaper", mt_bandwidth=2.0,
        fmin=FREQ_RANGE[0], fmax=FREQ_RANGE[1], faverage=False, n_jobs=1, verbose=False,
    )
    dense = result.get_data(output="dense")
    freqs = np.array(result.freqs)
    left = [SCALP_CHANNELS.index(c) for c in LEFT_CHANNELS]
    right = [SCALP_CHANNELS.index(c) for c in RIGHT_CHANNELS]
    sym = dense + dense.transpose(1, 0, 2)
    ihs_by_freq = sym[np.ix_(left, right)].mean(axis=(0, 1))
    power = 10 * np.log10(pxx[:, :, :].mean(axis=(0, 1)) / 1e-12)
    by_freq = pd.DataFrame({"id": sub_id, "group": group, "freq": freqs, "ihs": ihs_by_freq})
    by_freq["power_db"] = np.interp(freqs, f, power)

    return (
        {
            "id": sub_id,
            "group": group,
            "alpha_all_db": float(np.median(epoch_alpha)),
            "alpha_matched_db": float(np.median(epoch_alpha[keep])),
            "lh_rh_alpha_matched_epochs": ihs(connectome(matched, *BANDS["alpha"])),
        },
        by_freq,
    )


if __name__ == "__main__":
    pdr = pd.read_csv(os.path.join(DATA_DIR, "pdr.csv"))
    results = Parallel(n_jobs=min(16, multiprocessing.cpu_count()), backend="loky")(
        delayed(process_subject)(row.id, row.group) for row in pdr.itertuples()
    )
    matched = pd.DataFrame([r[0] for r in results])
    matched.to_csv(os.path.join(DATA_DIR, "epochs_alpha_matched.csv"), index=False)
    pd.concat([r[1] for r in results]).to_csv(os.path.join(DATA_DIR, "ihs_by_frequency.csv"), index=False)

    print(matched.groupby("group")[["alpha_all_db", "alpha_matched_db", "lh_rh_alpha_matched_epochs"]].median().round(4))
