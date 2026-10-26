"""
Additional test of the signal-to-noise explanation (analysis_spec.md,
Additional analyses).

dbWPLI is lower when the signal-to-noise ratio is lower, even if the coupling is
unchanged, and children with CTNNB1 syndrome had lower alpha power. Independent
1/f noise (band-passed like the data) was therefore added to the analysed epochs
of each matched control child until its aperiodic-adjusted posterior alpha power
(pdr.py) fell to the mean of the children with CTNNB1 syndrome of the same sex
and age, and the alpha-band connectome was recomputed (connectomes.py). If lower
alpha power alone produced the lower IHS, these control children would lose the
difference.
"""

import os
import multiprocessing

import numpy as np
import pandas as pd
import mne
from joblib import Parallel, delayed

from config import DERIV_ROOT, DATA_DIR, SCALP_CHANNELS, BANDS, LEFT_CHANNELS, RIGHT_CHANNELS, L_FREQ, H_FREQ, SEED
from pdr import epoch_spectrum, parameterise
from connectomes import connectome

mne.set_log_level("ERROR")

# precision of the alpha-power match (dB) and the largest noise tried (SD of the
# noise relative to that of each channel)
TOLERANCE_DB = 0.05
MAX_SCALE = 20.0


def pink_noise(rng: np.random.Generator, n_channels: int, n_times: int, sfreq: float) -> np.ndarray:
    """
    Independent 1/f noise for each channel, band-passed like the data, unit SD.

    Args:
        rng (np.random.Generator): Random number generator.
        n_channels (int): Number of channels.
        n_times (int): Number of samples.
        sfreq (float): Sampling frequency.
    """
    f = np.fft.rfftfreq(n_times, 1 / sfreq)
    f[0] = f[1]
    spectrum = (rng.standard_normal((n_channels, f.size)) + 1j * rng.standard_normal((n_channels, f.size))) / np.sqrt(f)
    x = np.fft.irfft(spectrum, n=n_times)
    x = mne.filter.filter_data(x, sfreq, l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)

    return x / x.std(axis=1, keepdims=True)


def with_noise(epochs: mne.Epochs, noise: np.ndarray, scale: float) -> mne.EpochsArray:
    """
    The epochs with noise added to each channel (scale x the channel's SD).

    Args:
        epochs (mne.Epochs): The analysed epochs.
        noise (np.ndarray): Unit-SD noise, channels x (epochs x samples), in epoch order.
        scale (float): SD of the noise relative to that of each channel.
    """
    data = epochs.get_data()
    n_epochs, n_channels, n_times = data.shape
    sd = data.transpose(1, 0, 2).reshape(n_channels, -1).std(axis=1)
    added = (noise * (scale * sd)[:, None]).reshape(n_channels, n_epochs, n_times).transpose(1, 0, 2)

    return mne.EpochsArray(data + added, epochs.info, events=epochs.events, tmin=epochs.tmin, verbose=False)


def alpha_power(epochs: mne.Epochs, age: float) -> float:
    """
    Aperiodic-adjusted posterior alpha power (dB), as in pdr.py.

    Args:
        epochs (mne.Epochs): Epochs.
        age (float): Age in years.
    """
    f, psd = epoch_spectrum(epochs)
    return parameterise(f, psd, epochs.ch_names, age)["alpha_adj_db"]


def process_subject(sub_id: str, age: float, target: float, seed: int) -> dict:
    """
    Degrade one control recording to the target alpha power and recompute IHS.

    Args:
        sub_id (str): Subject identifier.
        age (float): Age in years.
        target (float): Target alpha power (dB).
        seed (int): Seed of the noise.
    """
    epochs = mne.read_epochs(
        os.path.join(DERIV_ROOT, f"sub-{sub_id}", "eeg", f"sub-{sub_id}_task-rest_desc-clean_epo.fif")
    )
    epochs.reorder_channels(SCALP_CHANNELS)
    before = alpha_power(epochs, age)

    scale = 0.0
    if before > target:
        rng = np.random.default_rng(seed)
        n_epochs, n_channels, n_times = epochs.get_data().shape
        noise = pink_noise(rng, n_channels, n_epochs * n_times, epochs.info["sfreq"])
        low, high = 0.0, MAX_SCALE
        for _ in range(40):
            scale = (low + high) / 2
            achieved = alpha_power(with_noise(epochs, noise, scale), age)
            if abs(achieved - target) <= TOLERANCE_DB:
                break
            if achieved > target:
                low = scale
            else:
                high = scale
        epochs = with_noise(epochs, noise, scale)

    con = connectome(epochs, *BANDS["alpha"])
    left = [SCALP_CHANNELS.index(c) for c in LEFT_CHANNELS]
    right = [SCALP_CHANNELS.index(c) for c in RIGHT_CHANNELS]

    return {
        "id": sub_id,
        "alpha_before_db": before,
        "alpha_target_db": target,
        "alpha_after_db": alpha_power(epochs, age),
        "noise_scale": scale,
        "lh_rh_noise": float(con[np.ix_(left, right)].mean()),
    }


if __name__ == "__main__":
    metrics = pd.read_csv(os.path.join(DATA_DIR, "connectome_metrics_alpha.csv"))
    pdr = pd.read_csv(os.path.join(DATA_DIR, "pdr.csv"))[["id", "alpha_adj_db"]]
    df = metrics[["id", "group", "sex", "age"]].merge(pdr, on="id")

    # target: the mean alpha power of the children with CTNNB1 syndrome of the same sex and age
    targets = df[df["group"] == "test"].groupby(["sex", "age"])["alpha_adj_db"].mean().rename("target")
    controls = df[df["group"] == "control"].join(targets, on=["sex", "age"], how="inner")

    rows = Parallel(n_jobs=min(16, multiprocessing.cpu_count()), backend="loky")(
        delayed(process_subject)(row.id, float(row.age), float(row.target), SEED + i)
        for i, row in enumerate(controls.itertuples())
    )
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(DATA_DIR, "noise_injection.csv"), index=False)

    print(f"---> {len(out)} matched control children, {int((out['noise_scale'] > 0).sum())} with noise added")
    print(out[["alpha_before_db", "alpha_target_db", "alpha_after_db", "noise_scale"]].describe().round(2))
