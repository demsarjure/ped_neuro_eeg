"""
Posterior dominant rhythm (PDR), individual frequency bands and an objective
background-slowing proxy.

Spectra are estimated from runs of consecutive kept epochs (2-s Hann windows),
averaged over the posterior (O1, O2, P3, P4, Pz) and the central (C3, C4)
electrodes, and parameterised with FOOOF (1.5-35 Hz, fixed aperiodic
component). The PDR is the strongest posterior peak that is narrow (<= 5 Hz),
rises at least 1 dB above the aperiodic component, lies within +/- 3 Hz of the
PDR expected for age (Freschl et al. 2022; window clipped to 5-13 Hz) and,
within +/- 1 Hz of its frequency, has a local maximum of the aperiodic-adjusted
spectrum and more power above the aperiodic component posteriorly than
centrally.
Age norms are estimated from controls with a clear PDR; a child is flagged as
having background slowing if no clear PDR is found, or if the PDR or the
aperiodic-adjusted theta/alpha ratio falls outside the controls' age-adjusted
95% prediction interval in the slowing direction.
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
import mne
import statsmodels.api as sm
from scipy.signal import welch

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from fooof import FOOOF

from config import (
    DERIV_ROOT,
    DATA_DIR,
    SFREQ,
    EPOCH_DURATION,
    POSTERIOR_CHANNELS,
    PDR_RANGE,
    PDR_MIN_HEIGHT_DB,
    PDR_MAX_WIDTH,
    PDR_WINDOW,
    PDR_EXPECTED,
    IAF_HALF_WIDTH,
    IAF_FLOOR,
)

mne.set_log_level("ERROR")

FIT_RANGE = (1.5, 35.0)
CENTRAL_CHANNELS = ["C3", "C4"]
# half-width (Hz) around the peak frequency for the local-maximum and
# posterior-dominance tests
PEAK_HALF_WIDTH = 1.0
THETA = (4.0, 7.0)
ALPHA = (8.0, 13.0)
PI_LEVEL = 0.95


def epoch_spectrum(epochs: mne.Epochs) -> tuple[np.ndarray, np.ndarray]:
    """
    Power spectrum from 2-s Hann windows over runs of consecutive epochs.

    Args:
        epochs (mne.Epochs): The kept 1-s epochs.
    """
    width = int(EPOCH_DURATION * SFREQ)
    onsets = epochs.events[:, 0]
    data = epochs.get_data()

    # split the epochs into runs of consecutive epochs
    runs = np.split(np.arange(len(onsets)), np.where(np.diff(onsets) != width)[0] + 1)

    psd_sum, n_windows = 0, 0
    for run in runs:
        segment = np.concatenate(data[run], axis=1)
        if segment.shape[1] < 2 * width:
            continue
        f, pxx = welch(segment, fs=SFREQ, nperseg=2 * width, noverlap=width)
        n = 1 + (segment.shape[1] - 2 * width) // width
        psd_sum = psd_sum + pxx * n
        n_windows += n

    if n_windows == 0:
        # no consecutive epochs: zero-padded 1-s epochs
        f, pxx = welch(data, fs=SFREQ, nperseg=width, nfft=2 * width)
        return f, pxx.mean(axis=0)

    return f, psd_sum / n_windows


def expected_pdr(age: float) -> float:
    """
    PDR expected for age (Freschl et al. 2022), linearly interpolated.

    Args:
        age (float): Age in years.
    """
    ages, freqs = zip(*PDR_EXPECTED)
    return float(np.interp(age, ages, freqs))


def local_maximum(f: np.ndarray, spectrum: np.ndarray, cf: float, half_width: float = PEAK_HALF_WIDTH) -> bool:
    """
    Whether the aperiodic-adjusted (flattened) spectrum has a local maximum
    within +/- half_width of cf, i.e. the fitted peak corresponds to a real
    bump rather than a fitting artefact. A bin is a local maximum if it is
    higher than both of its neighbours in the full spectrum.

    Args:
        f (np.ndarray): Frequencies.
        spectrum (np.ndarray): Flattened power spectrum.
        cf (float): Peak frequency.
        half_width (float): Search half-width in Hz.
    """
    # bins within the search window that have a neighbour on both sides
    sel = np.where(np.abs(f - cf) <= half_width)[0]
    sel = sel[(sel > 0) & (sel < len(f) - 1)]
    return bool(np.any((spectrum[sel] > spectrum[sel - 1]) & (spectrum[sel] > spectrum[sel + 1])))


def fit_spectrum(f: np.ndarray, spectrum: np.ndarray) -> FOOOF:
    """
    FOOOF model of a power spectrum with a fixed aperiodic component (no knee).

    Args:
        f (np.ndarray): Frequencies.
        spectrum (np.ndarray): Power spectrum.
    """
    fm = FOOOF(
        peak_width_limits=[1.0, 8.0],
        max_n_peaks=6,
        min_peak_height=0.0,
        aperiodic_mode="fixed",
        verbose=False,
    )
    fm.fit(f, spectrum, list(FIT_RANGE))
    return fm


def parameterise(f: np.ndarray, psd: np.ndarray, ch_names: list[str], age: float) -> dict:
    """
    Find the PDR and spectral background measures of one child.

    Args:
        f (np.ndarray): Frequencies.
        psd (np.ndarray): Power spectra, channels x frequencies.
        ch_names (list[str]): Channel names.
        age (float): Age in years.
    """
    posterior = psd[[ch_names.index(c) for c in POSTERIOR_CHANNELS]].mean(axis=0)
    central = psd[[ch_names.index(c) for c in CENTRAL_CHANNELS]].mean(axis=0)

    # aperiodic-adjusted (flattened) posterior spectrum, and the power above the
    # aperiodic component (the rhythm itself) posteriorly and centrally
    fm = fit_spectrum(f, posterior)
    fm_central = fit_spectrum(f, central)
    freqs = fm.freqs
    flat = fm._spectrum_flat
    rhythm = 10**fm.power_spectrum - 10**fm._ap_fit
    rhythm_central = 10**fm_central.power_spectrum - 10**fm_central._ap_fit

    # candidate peaks: high enough, narrow, within the window around the PDR
    # expected for age, posterior-dominant (mean power above the aperiodic
    # component within +/- 1 Hz) and a local maximum of the flattened spectrum
    expected = expected_pdr(age)
    low = max(PDR_RANGE[0], expected - PDR_WINDOW)
    high = min(PDR_RANGE[1], expected + PDR_WINDOW)
    result = {
        "pdr_hz": np.nan,
        "pdr_height_db": np.nan,
        "pdr_clear": False,
        "pdr_age_expected_hz": expected,
    }
    peaks = fm.get_params("peak_params")
    peaks = np.atleast_2d(peaks) if np.size(peaks) else np.empty((0, 3))
    candidates = []
    for cf, pw, bw in peaks:
        near = np.abs(freqs - cf) <= PEAK_HALF_WIDTH
        if (
            low <= cf <= high
            and 10 * pw >= PDR_MIN_HEIGHT_DB
            and bw <= PDR_MAX_WIDTH
            and rhythm[near].mean() > rhythm_central[near].mean()
            and local_maximum(freqs, flat, cf)
        ):
            candidates.append((cf, 10 * pw))

    if candidates:
        # the strongest qualifying peak
        cf, height_db = max(candidates, key=lambda c: c[1])
        result.update({"pdr_hz": float(cf), "pdr_height_db": float(height_db), "pdr_clear": True})

    # aperiodic-adjusted power (dB) in theta and alpha
    theta = flat[(freqs >= THETA[0]) & (freqs <= THETA[1])].mean()
    alpha = flat[(freqs >= ALPHA[0]) & (freqs <= ALPHA[1])].mean()
    result.update(
        {
            "alpha_adj_db": float(10 * alpha),
            "theta_alpha_db": float(10 * (theta - alpha)),
            "aperiodic_exponent": float(fm.get_params("aperiodic_params", "exponent")),
            "fit_r2": float(fm.r_squared_),
        }
    )

    return result


def prediction_interval(fit, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean and 95% prediction interval of an OLS fit on log(age + 1).

    Args:
        fit: Fitted statsmodels OLS results.
        x (np.ndarray): Ages in years.
    """
    X = sm.add_constant(np.log(x + 1), has_constant="add")
    pred = fit.get_prediction(X).summary_frame(alpha=1 - PI_LEVEL)
    return pred["mean"].to_numpy(), pred["obs_ci_lower"].to_numpy(), pred["obs_ci_upper"].to_numpy()


if __name__ == "__main__":
    dem = pd.concat(
        [
            pd.read_csv(os.path.join(DATA_DIR, "demographics_test.csv")),
            pd.read_csv(os.path.join(DATA_DIR, "demographics_control.csv")),
        ],
        ignore_index=True,
    )
    qc = pd.read_csv(os.path.join(DATA_DIR, "qc_summary.csv"))
    analysable = qc.loc[~qc["excluded"], "id"].tolist()

    ages = dem.set_index("id")["age"]

    rows = []
    for sub_id in analysable:
        epochs = mne.read_epochs(
            os.path.join(DERIV_ROOT, f"sub-{sub_id}", "eeg", f"sub-{sub_id}_task-rest_desc-clean_epo.fif")
        )
        f, psd = epoch_spectrum(epochs)
        rows.append({"id": sub_id, **parameterise(f, psd, epochs.ch_names, float(ages[sub_id]))})
        print(f"    - {sub_id}: PDR {rows[-1]['pdr_hz']:.1f} Hz (clear: {rows[-1]['pdr_clear']})")

    df = pd.DataFrame(rows).merge(dem, on="id", how="left")
    df = df.merge(qc[["id", "group", "spindle_flag"]], on="id", how="left")

    # possible sleep: spindles without a posterior dominant rhythm
    df["sleep_flag"] = df["spindle_flag"] & ~df["pdr_clear"]

    # age norms from controls with a clear PDR (awake by definition)
    norm = df[(df["group"] == "control") & df["pdr_clear"]]
    X = sm.add_constant(np.log(norm["age"] + 1))
    fit_pdr = sm.OLS(norm["pdr_hz"], X).fit()
    fit_tar = sm.OLS(norm["theta_alpha_db"], X).fit()

    pdr_mean, pdr_lower, _ = prediction_interval(fit_pdr, df["age"].to_numpy())
    _, _, tar_upper = prediction_interval(fit_tar, df["age"].to_numpy())

    df["pdr_expected_hz"] = pdr_mean
    df["slowing_flag"] = (
        ~df["pdr_clear"]
        | (df["pdr_hz"] < pdr_lower)
        | (df["theta_alpha_db"] > tar_upper)
    )

    # individual band centred on the PDR (or the PDR expected from the control
    # norms), rounded to the 1-Hz frequency resolution of 1-s epochs
    centre = np.round(np.where(df["pdr_clear"], df["pdr_hz"], df["pdr_expected_hz"]))
    df["iaf_low"] = np.maximum(IAF_FLOOR, centre - IAF_HALF_WIDTH)
    df["iaf_high"] = centre + IAF_HALF_WIDTH

    df.to_csv(os.path.join(DATA_DIR, "pdr.csv"), index=False)
    with open(os.path.join(DATA_DIR, "pdr_norms.json"), "w") as f:
        json.dump(
            {
                "n_controls": int(len(norm)),
                "pdr": {"params": fit_pdr.params.to_dict(), "resid_sd": float(np.sqrt(fit_pdr.scale))},
                "theta_alpha_db": {"params": fit_tar.params.to_dict(), "resid_sd": float(np.sqrt(fit_tar.scale))},
            },
            f,
            indent=2,
        )

    print()
    print(f"---> PDR norms from {len(norm)} controls")
    print(df.groupby("group")[["pdr_clear", "slowing_flag", "spindle_flag", "sleep_flag"]].sum())
