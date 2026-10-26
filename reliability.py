"""
Additional checks of the IHS estimate (analysis_spec.md, Additional
analyses).

1. Split-half reliability: IHS, and posterior alpha power, from the odd and from
   the even analysed epochs of each child, correlated across children in each
   group (Spearman-Brown corrected).
2. Chance level: IHS recomputed after the right-hemisphere electrodes were
   shuffled across epochs (the same permutation for all of them, so their mutual
   timing is kept), 200 times per child; a child's IHS is above chance if it
   exceeds the 95th percentile of these values.
"""

import os
import multiprocessing

import numpy as np
import pandas as pd
import mne
from joblib import Parallel, delayed

from scipy.signal import welch

from config import DERIV_ROOT, DATA_DIR, SCALP_CHANNELS, LEFT_CHANNELS, RIGHT_CHANNELS, POSTERIOR_CHANNELS, BANDS, SEED, SFREQ
from connectomes import connectome

mne.set_log_level("ERROR")

N_SHUFFLES = 200


def ihs(con: np.ndarray) -> float:
    """
    Interhemispheric strength of a connectome (mean over the 64 left-right pairs).

    Args:
        con (np.ndarray): Connectome, channels x channels.
    """
    left = [SCALP_CHANNELS.index(c) for c in LEFT_CHANNELS]
    right = [SCALP_CHANNELS.index(c) for c in RIGHT_CHANNELS]
    return float(con[np.ix_(left, right)].mean())


def process_subject(sub_id: str, seed: int) -> dict:
    """
    Split-half IHS and the chance level of IHS for one child.

    Args:
        sub_id (str): Subject identifier.
        seed (int): Seed of the shuffles.
    """
    epochs = mne.read_epochs(
        os.path.join(DERIV_ROOT, f"sub-{sub_id}", "eeg", f"sub-{sub_id}_task-rest_desc-clean_epo.fif")
    )
    epochs.reorder_channels(SCALP_CHANNELS)
    band = BANDS["alpha"]

    odd = ihs(connectome(epochs[0::2], *band))
    even = ihs(connectome(epochs[1::2], *band))

    # posterior alpha power of the odd and even epochs (dB relative to 1 uV^2/Hz)
    x = epochs.get_data()[:, [SCALP_CHANNELS.index(c) for c in POSTERIOR_CHANNELS]]
    f, pxx = welch(x, fs=SFREQ, nperseg=x.shape[-1], nfft=2 * x.shape[-1])
    per_epoch = pxx[:, :, (f >= band[0]) & (f <= band[1])].mean(axis=(1, 2))

    rng = np.random.default_rng(seed)
    data = epochs.get_data()
    right = [SCALP_CHANNELS.index(c) for c in RIGHT_CHANNELS]
    null = []
    for _ in range(N_SHUFFLES):
        shuffled = data.copy()
        shuffled[:, right] = data[rng.permutation(len(data))][:, right]
        null.append(ihs(connectome(mne.EpochsArray(shuffled, epochs.info, verbose=False), *band)))
    observed = ihs(connectome(epochs, *band))

    return {
        "id": sub_id,
        "ihs_odd": odd,
        "ihs_even": even,
        "alpha_odd": float(10 * np.log10(per_epoch[0::2].mean() / 1e-12)),
        "alpha_even": float(10 * np.log10(per_epoch[1::2].mean() / 1e-12)),
        "ihs": observed,
        "ihs_chance_95": float(np.percentile(null, 95)),
        "above_chance": bool(observed > np.percentile(null, 95)),
    }


if __name__ == "__main__":
    pdr = pd.read_csv(os.path.join(DATA_DIR, "pdr.csv"))[["id", "group"]]
    rows = Parallel(n_jobs=min(16, multiprocessing.cpu_count()), backend="loky")(
        delayed(process_subject)(row.id, SEED + i) for i, row in enumerate(pdr.itertuples())
    )
    out = pd.DataFrame(rows).merge(pdr, on="id")
    out.to_csv(os.path.join(DATA_DIR, "ihs_reliability.csv"), index=False)

    for g, d in out.groupby("group"):
        r = d["ihs_odd"].corr(d["ihs_even"], method="spearman")
        a = d["alpha_odd"].corr(d["alpha_even"], method="spearman")
        print(
            f"{g}: IHS split-half Spearman {r:.2f} (Spearman-Brown {2 * r / (1 + r):.2f}), "
            f"alpha power {a:.2f} ({2 * a / (1 + a):.2f}); above chance {d['above_chance'].sum()} of {len(d)}"
        )
