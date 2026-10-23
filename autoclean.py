"""
The cleanup script for EEG data.

For each recording: exclusion of recording breaks, photic stimulation (+/- 60 s)
and hyperventilation (and the 2 min after it); the candidate pool of the 400
1-s epochs nearest to the recording midpoint; bad-channel detection on the pool
and interpolation; 1-40 Hz band-pass, common average reference, ICA removal of
ocular and cardiac components, Autoreject (local, fitted on the pool) and
selection of up to 100 artefact-free 1-s epochs nearest to the midpoint.
"""

import os
import json
import multiprocessing

import numpy as np
import pandas as pd
import mne
import yasa
from joblib import Parallel, delayed
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import ICA
from autoreject import AutoReject
from scipy.signal import welch, find_peaks

from config import (
    BIDS_ROOT,
    DERIV_ROOT,
    DATA_DIR,
    SEED,
    UNIT_SCALE,
    SCALP_CHANNELS,
    ECG_CHANNEL,
    FRONTAL_CHANNELS,
    EPOCH_DURATION,
    N_EPOCHS,
    MIN_EPOCHS,
    CANDIDATE_POOL,
    EDGE_EXCLUDE_S,
    BREAK_MARGIN_S,
    PHOTIC_MARGIN_S,
    HV_MARGIN_AFTER_S,
    L_FREQ,
    H_FREQ,
    ICA_L_FREQ,
    MAX_BAD_CHANNELS,
    SPINDLE_FREQ,
    SPINDLE_CHANNELS,
    SPINDLE_RATE_FLAG,
)

mne.set_log_level("ERROR")

# bad-channel criteria (fixed before any group comparison) ---------------------
# judged in each epoch of the candidate pool, in the recording reference
# flat channel: 1-40 Hz amplitude (RMS) below 1 uV
BAD_FLAT_RMS = 1e-6
# poor contact or unconnected electrode: both the 48-52 Hz mains amplitude and
# the 1-40 Hz amplitude at least 5 times those of the epoch's lower-quartile
# channel (physiologically large channels have normal mains amplitude)
BAD_CONTACT_RATIO = 5.0
# a channel is bad if it is flat or poorly connected in more than 20% of the
# pool epochs
BAD_EPOCH_FRACTION = 0.2

# ICA ----------------------------------------------------------------------------
# references: the frontal electrodes (ocular; scored at 1-10 Hz) and the ECG
# electrode when connected (scored at 8-16 Hz)
FRONTAL_BAND = (1.0, 10.0)
ECG_BAND = (8.0, 16.0)
# a component is removed if its correlation with a reference is an outlier
# among the components (|z| > 3, z-scored once per reference, as in the first
# pass of MNE's find_bads_eog) and at least 0.3 in absolute value
ICA_THRESHOLD = 3.0
ICA_MIN_CORR = 0.3
ICA_MAX_EXCLUDED = 5
# components that look like brain rhythms are never removed: more than 40% of
# their power in 8-13 Hz, or larger weights over posterior than frontal electrodes
ICA_MAX_ALPHA_SHARE = 0.4
POSTERIOR_ICA_CHANNELS = ["O1", "O2", "P3", "P4", "Pz"]
# the ECG electrode is connected if less than half of its (unfiltered) power is
# mains
ECG_MAX_LINE_FRAC = 0.5

# blinks -------------------------------------------------------------------------
BLINK_HEIGHT = 75e-6


def load_raw(sub_id: str) -> mne.io.BaseRaw:
    """
    Load a BIDS recording in volts, keeping scalp and ECG channels.

    Args:
        sub_id (str): Subject identifier, e.g. T001.
    """
    bids_path = BIDSPath(
        root=BIDS_ROOT,
        subject=sub_id,
        datatype="eeg",
        task="rest",
        extension=".vhdr",
    )
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()

    # the export is in mV, stored as if in V
    raw.apply_function(lambda x: x * UNIT_SCALE, picks="all", channel_wise=False)

    keep = SCALP_CHANNELS + ([ECG_CHANNEL] if ECG_CHANNEL in raw.ch_names else [])
    raw.pick(keep)

    return raw


def mark_breaks(raw: mne.io.BaseRaw) -> np.ndarray:
    """
    Annotate recording breaks (NaN samples) as BAD_ACQ_SKIP and zero-fill them.

    Args:
        raw (mne.io.BaseRaw): The recording, modified in place.
    """
    data = raw._data
    breaks = np.where(np.isnan(data).any(axis=0))[0]
    data[:, breaks] = 0.0

    if breaks.size:
        onsets = breaks / raw.info["sfreq"]
        raw.set_annotations(
            raw.annotations
            + mne.Annotations(
                onset=onsets,
                duration=np.full(breaks.size, 1 / raw.info["sfreq"]),
                description=["BAD_ACQ_SKIP"] * breaks.size,
                orig_time=raw.annotations.orig_time,
            )
        )

    return breaks


def union_length(intervals: list[tuple[float, float]]) -> float:
    """
    Total length of the union of time intervals.

    Args:
        intervals (list[tuple[float, float]]): (start, end) pairs in seconds.
    """
    total = 0.0
    last_end = -np.inf
    for a, b in sorted(intervals):
        a = max(a, last_end)
        if b > a:
            total += b - a
        last_end = max(last_end, b)

    return total


def mark_exclusions(
    raw: mne.io.BaseRaw, breaks: np.ndarray, photic: pd.DataFrame, hv: pd.DataFrame
) -> tuple[float, float]:
    """
    Annotate time that may not provide epochs: recording edges, the
    surroundings of breaks, photic stimulation and hyperventilation. Returns
    the excluded photic and hyperventilation time in seconds.

    Args:
        raw (mne.io.BaseRaw): The recording, modified in place.
        breaks (np.ndarray): Sample indices of recording breaks.
        photic (pd.DataFrame): Photic stimulation intervals (onset_s, offset_s).
        hv (pd.DataFrame): Hyperventilation intervals (onset_s, offset_s).
    """
    sfreq = raw.info["sfreq"]
    total = raw.n_times / sfreq

    intervals = [(0.0, EDGE_EXCLUDE_S), (total - EDGE_EXCLUDE_S, total)]
    intervals += [
        (b / sfreq - BREAK_MARGIN_S, b / sfreq + BREAK_MARGIN_S) for b in breaks
    ]
    photic_intervals = [
        (max(0.0, on - PHOTIC_MARGIN_S), min(total, off + PHOTIC_MARGIN_S))
        for on, off in zip(photic["onset_s"], photic["offset_s"])
    ]
    # hyperventilation and the time after it
    hv_intervals = [
        (on, min(total, off + HV_MARGIN_AFTER_S))
        for on, off in zip(hv["onset_s"], hv["offset_s"])
    ]
    intervals += photic_intervals + hv_intervals

    intervals = [(max(0.0, a), min(total, b)) for a, b in intervals if b > a]
    raw.set_annotations(
        raw.annotations
        + mne.Annotations(
            onset=[a for a, _ in intervals],
            duration=[b - a for a, b in intervals],
            description=["BAD_exclude"] * len(intervals),
            orig_time=raw.annotations.orig_time,
        )
    )

    return union_length(photic_intervals), union_length(hv_intervals)


def nearest_midpoint(onsets: np.ndarray, n_times: int, sfreq: float, n: int) -> np.ndarray:
    """
    Indices (in time order) of the n epochs whose centres are nearest the
    recording midpoint.

    Args:
        onsets (np.ndarray): Sample indices of the epoch onsets.
        n_times (int): Length of the recording in samples.
        sfreq (float): Sampling frequency.
        n (int): Number of epochs to keep.
    """
    centres = onsets / sfreq + EPOCH_DURATION / 2
    midpoint = n_times / sfreq / 2

    return np.sort(np.argsort(np.abs(centres - midpoint), kind="stable")[:n])


def channel_flags(raw: mne.io.BaseRaw) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build the candidate pool (the 1-s epochs outside BAD annotations nearest the
    recording midpoint) and find, in the recording reference, the fraction of
    pool epochs in which each scalp channel is flat or poorly connected.
    Returns the pool onsets (samples) and the fractions.

    Args:
        raw (mne.io.BaseRaw): The annotated recording in volts.
    """
    sfreq = raw.info["sfreq"]
    scalp = raw.copy().pick(SCALP_CHANNELS).filter(l_freq=1.0, h_freq=None)
    epochs = mne.make_fixed_length_epochs(
        scalp, duration=EPOCH_DURATION, reject_by_annotation=True, preload=True
    )
    pool = nearest_midpoint(epochs.events[:, 0], raw.n_times, sfreq, CANDIDATE_POOL)
    x = epochs.get_data()[pool]

    f, pxx = welch(x, fs=sfreq, nperseg=x.shape[-1])
    df = f[1] - f[0]
    mains = np.sqrt(pxx[..., (f >= 48) & (f <= 52)].sum(axis=-1) * df)
    rms = np.sqrt(pxx[..., (f >= 1) & (f <= 40)].sum(axis=-1) * df)

    # poor contact relative to each epoch's lower-quartile channel
    flat = rms < BAD_FLAT_RMS
    contact = (mains >= BAD_CONTACT_RATIO * np.quantile(mains, 0.25, axis=1, keepdims=True)) & (
        rms >= BAD_CONTACT_RATIO * np.quantile(rms, 0.25, axis=1, keepdims=True)
    )

    fractions = pd.DataFrame(
        {
            "channel": SCALP_CHANNELS,
            "flat": flat.mean(axis=0),
            "contact": contact.mean(axis=0),
            "flagged": (flat | contact).mean(axis=0),
        }
    )
    return epochs.events[pool, 0], fractions


def find_bad_channels(fractions: pd.DataFrame) -> tuple[list[str], dict]:
    """
    Apply the fixed bad-channel criteria: channels flat or poorly connected in
    more than 20% of the pool epochs. Returns the bad channels and the reason
    for each (the more frequent of the two).

    Args:
        fractions (pd.DataFrame): Output of channel_flags.
    """
    bad = fractions[fractions["flagged"] > BAD_EPOCH_FRACTION]
    reasons = {
        row.channel: "flat" if row.flat >= row.contact else "poor contact"
        for row in bad.itertuples()
    }
    return bad["channel"].tolist(), reasons


def ecg_connected(raw: mne.io.BaseRaw) -> bool:
    """
    Decide whether the ECG electrode is connected, on the unfiltered recording
    outside excluded periods.

    Args:
        raw (mne.io.BaseRaw): The annotated recording in volts, before filtering.
    """
    if ECG_CHANNEL not in raw.ch_names:
        return False

    sfreq = raw.info["sfreq"]
    x = raw.get_data(picks=[ECG_CHANNEL], reject_by_annotation="omit")[0]
    f, pxx = welch(x, fs=sfreq, nperseg=int(2 * sfreq))
    line_frac = pxx[(f >= 48) & (f <= 52)].sum() / pxx[(f >= 1) & (f <= 52)].sum()

    return bool(line_frac < ECG_MAX_LINE_FRAC)


def usable_mask(raw: mne.io.BaseRaw) -> np.ndarray:
    """
    Samples outside BAD annotations.

    Args:
        raw (mne.io.BaseRaw): The annotated recording.
    """
    mask = np.ones(raw.n_times, dtype=bool)
    sfreq = raw.info["sfreq"]
    for a in raw.annotations:
        if a["description"].upper().startswith("BAD"):
            start = int(round(a["onset"] * sfreq))
            stop = int(round((a["onset"] + a["duration"]) * sfreq))
            mask[max(0, start) : max(0, stop) + 1] = False
    return mask


def select_components(
    ica: ICA, raw_ica: mne.io.BaseRaw, ecg_used: bool
) -> tuple[list[int], list[int]]:
    """
    Ocular and cardiac components. Every component is correlated with each
    reference on the continuous usable data (frontal electrodes band-passed
    1-10 Hz, ECG 8-16 Hz). For each reference, the correlations are z-scored
    once across components (no iterative re-scoring); a component is a
    candidate if |z| > 3 and |r| >= 0.3 on the same reference. Alpha- or
    posterior-dominant candidates are never removed; at most 5 are, ranked by
    |r|. Returns the removed and the protected components.

    Args:
        ica (ICA): The fitted ICA.
        raw_ica (mne.io.BaseRaw): The continuous 1 Hz high-passed data.
        ecg_used (bool): Whether the ECG channel is a usable reference.
    """
    references = [(ch, FRONTAL_BAND) for ch in FRONTAL_CHANNELS]
    if ecg_used:
        references.append((ECG_CHANNEL, ECG_BAND))

    strength = {}
    for ch, (l_freq, h_freq) in references:
        r = ica.score_sources(
            raw_ica, target=ch, score_func="pearsonr", l_freq=l_freq, h_freq=h_freq,
            reject_by_annotation=True,
        )
        z = (r - r.mean()) / r.std()
        for i in np.flatnonzero((np.abs(z) > ICA_THRESHOLD) & (np.abs(r) >= ICA_MIN_CORR)):
            strength[int(i)] = max(strength.get(int(i), 0.0), float(abs(r[i])))

    # brain rhythms are protected
    sfreq = raw_ica.info["sfreq"]
    sources = ica.get_sources(raw_ica).get_data(reject_by_annotation="omit")
    f, pxx = welch(sources, fs=sfreq, nperseg=int(2 * sfreq))
    band = (f >= 1) & (f <= 40)
    alpha_share = pxx[:, (f >= 8) & (f <= 13)].sum(axis=1) / pxx[:, band].sum(axis=1)
    patterns = np.abs(ica.get_components())
    names = ica.info["ch_names"]
    frontal = patterns[[names.index(c) for c in FRONTAL_CHANNELS]].mean(axis=0)
    posterior = patterns[[names.index(c) for c in POSTERIOR_ICA_CHANNELS]].mean(axis=0)

    candidates, protected = [], []
    for i in strength:
        if alpha_share[i] > ICA_MAX_ALPHA_SHARE or posterior[i] > frontal[i]:
            protected.append(i)
        else:
            candidates.append(i)

    removed = sorted(candidates, key=lambda i: -strength[i])[:ICA_MAX_EXCLUDED]
    return sorted(removed), sorted(protected)


def blink_rate(raw: mne.io.BaseRaw) -> float:
    """
    Blinks per minute over all usable data, from a vertical EOG proxy
    (Fp1/Fp2 minus C3/C4, 1-8 Hz) before ICA, with a fixed 75 uV threshold.
    The exports differ in polarity, so deflections are counted in both
    directions and the larger count (the recording's blink polarity) is kept.

    Args:
        raw (mne.io.BaseRaw): Filtered, average-referenced, annotated data before ICA.
    """
    sfreq = raw.info["sfreq"]
    x = raw.get_data(picks=["Fp1", "Fp2", "C3", "C4"])
    v = x[:2].mean(axis=0) - x[2:].mean(axis=0)
    v = mne.filter.filter_data(v, sfreq, l_freq=1.0, h_freq=8.0, verbose=False)

    usable = usable_mask(raw)
    n_blinks = 0
    for polarity in [1, -1]:
        peaks, _ = find_peaks(polarity * v, height=BLINK_HEIGHT, distance=int(0.3 * sfreq))
        n_blinks = max(n_blinks, usable[peaks].sum())

    return float(n_blinks / (usable.sum() / sfreq / 60))


def spindle_rate(raw: mne.io.BaseRaw, onsets: np.ndarray, min_run_s: float = 5.0) -> float:
    """
    Sigma bursts (12-15 Hz, YASA spindle detector) per minute, mean over central
    channels, detected separately on each run of at least 5 s of consecutive kept
    epochs (so that excluded periods never enter the detection).

    Args:
        raw (mne.io.BaseRaw): Cleaned continuous data.
        onsets (np.ndarray): Sample indices of the kept epochs.
        min_run_s (float): Shortest run analysed, in seconds.
    """
    sfreq = raw.info["sfreq"]
    width = int(EPOCH_DURATION * sfreq)
    runs = np.split(onsets, np.where(np.diff(onsets) != width)[0] + 1)

    n_spindles, n_epochs = 0, 0
    for run in runs:
        if len(run) * EPOCH_DURATION < min_run_s:
            continue
        x = raw.get_data(picks=SPINDLE_CHANNELS, start=run[0], stop=run[-1] + width) * 1e6
        sp = yasa.spindles_detect(
            x, sf=sfreq, ch_names=SPINDLE_CHANNELS, freq_sp=SPINDLE_FREQ,
            remove_outliers=False, verbose=False,
        )
        n_epochs += len(run)
        if sp is not None:
            n_spindles += len(sp.summary())

    if n_epochs == 0:
        return 0.0
    minutes = n_epochs * EPOCH_DURATION / 60
    return float(n_spindles / len(SPINDLE_CHANNELS) / minutes)


def process_subject(sub_id: str, events: pd.DataFrame) -> dict:
    """
    Preprocess one recording and save its kept epochs.

    Args:
        sub_id (str): Subject identifier.
        events (pd.DataFrame): Recording events of this subject.
    """
    group = "test" if sub_id.startswith("T") else "control"
    qc = {"id": sub_id, "group": group, "excluded": False, "exclusion_reason": ""}
    out_dir = os.path.join(DERIV_ROOT, f"sub-{sub_id}", "eeg")
    os.makedirs(out_dir, exist_ok=True)

    try:
        raw = load_raw(sub_id)
        sfreq = raw.info["sfreq"]
        qc["duration_s"] = raw.n_times / sfreq

        n_expected = events.loc[events["event"] == "recording", "offset_sample"].iloc[0]
        if raw.n_times != n_expected:
            raise ValueError(f"length {raw.n_times} != export {n_expected}")

        # breaks and exclusions
        breaks = mark_breaks(raw)
        qc["n_breaks"] = int(breaks.size)
        photic = events[events["event"] == "photic"]
        hv = events[events["event"] == "hv"]
        qc["photic_excluded_s"], qc["hv_excluded_s"] = mark_exclusions(raw, breaks, photic, hv)

        # bad channels, judged on the candidate pool
        pool, fractions = channel_flags(raw)
        bads, reasons = find_bad_channels(fractions)
        qc["bad_channels"] = bads
        qc["n_bad_channels"] = len(bads)
        qc["bad_channel_reasons"] = reasons
        qc["channel_flagged_fraction"] = {
            row.channel: round(row.flagged, 3) for row in fractions.itertuples() if row.flagged > 0
        }
        if len(bads) > MAX_BAD_CHANNELS:
            qc["excluded"] = True
            qc["exclusion_reason"] = f"{len(bads)} unusable channels ({', '.join(bads)})"
            return qc
        if bads:
            raw.info["bads"] = bads
            raw.interpolate_bads(reset_bads=True)

        # the ECG reference is judged before filtering
        ecg_used = ecg_connected(raw)

        # band-pass and average reference
        raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, picks="all")
        raw.set_eeg_reference("average", projection=False)
        raw_pre_ica = raw.copy()

        # ICA on 1 Hz high-passed candidate epochs
        raw_ica = raw.copy().filter(l_freq=ICA_L_FREQ, h_freq=None, picks="all")
        epochs_ica = mne.make_fixed_length_epochs(
            raw_ica, duration=EPOCH_DURATION, reject_by_annotation=True, preload=True
        )
        qc["candidate_epochs"] = len(epochs_ica)
        rank = len(SCALP_CHANNELS) - 1 - len(bads)
        ica = ICA(
            n_components=rank, method="fastica", random_state=SEED, max_iter="auto"
        )
        ica.fit(epochs_ica, picks="eeg")

        removed, protected = select_components(ica, raw_ica, ecg_used)
        ica.exclude = removed
        qc["ecg_used"] = ecg_used
        qc["ica_n_components"] = rank
        qc["ica_excluded"] = ica.exclude
        qc["n_ica_excluded"] = len(ica.exclude)
        qc["ica_protected"] = protected

        # how much signal ICA removed (usable data, scalp channels)
        before = raw.get_data(picks=SCALP_CHANNELS, reject_by_annotation="omit")
        raw = ica.apply(raw)
        after = raw.get_data(picks=SCALP_CHANNELS, reject_by_annotation="omit")
        qc["ica_removed_variance"] = float(1 - after.var(axis=1).sum() / before.var(axis=1).sum())

        # Autoreject (local) on the candidate pool
        epochs = mne.make_fixed_length_epochs(
            raw, duration=EPOCH_DURATION, reject_by_annotation=True, preload=True
        )
        epochs.pick(SCALP_CHANNELS)
        epochs = epochs[np.isin(epochs.events[:, 0], pool)]
        qc["pool_epochs"] = len(epochs)
        ar = AutoReject(
            n_interpolate=np.array([1, 2, 3]),
            consensus=np.linspace(0, 1.0, 11),
            random_state=SEED,
            n_jobs=1,
            verbose=False,
        )
        epochs_ar, reject_log = ar.fit_transform(epochs, return_log=True)
        epochs_ar.set_eeg_reference("average", projection=False)

        # learned thresholds relative to each channel's median peak-to-peak
        # amplitude (values below 1 mean most epochs exceed the threshold)
        x = epochs.get_data()
        median_p2p = np.median(x.max(axis=2) - x.min(axis=2), axis=0)
        ratios = [ar.threshes_[ch] / median_p2p[i] for i, ch in enumerate(epochs.ch_names)]
        qc["ar_threshold_ratio"] = float(np.median(ratios))
        qc["median_p2p_uv"] = float(np.median(median_p2p) * 1e6)

        qc["ar_rejected_epochs"] = int(reject_log.bad_epochs.sum())
        good = ~reject_log.bad_epochs
        qc["ar_interpolated_channel_epochs"] = int((reject_log.labels[good] == 2).sum())
        qc["good_epochs"] = len(epochs_ar)

        if len(epochs_ar) < MIN_EPOCHS:
            qc["excluded"] = True
            qc["exclusion_reason"] = f"{len(epochs_ar)} artefact-free epochs"
            return qc

        # the artefact-free epochs nearest to the recording midpoint
        nearest = nearest_midpoint(epochs_ar.events[:, 0], raw.n_times, sfreq, N_EPOCHS)
        kept = epochs_ar[nearest]
        kept_onsets = kept.events[:, 0]

        kept_s = kept_onsets / sfreq
        qc["n_epochs"] = len(kept)
        qc["kept_first_s"] = float(kept_s.min())
        qc["kept_last_s"] = float(kept_s.max() + EPOCH_DURATION)
        qc["kept_median_s"] = float(np.median(kept_s))
        qc["kept_median_frac"] = float(np.median(kept_s) / qc["duration_s"])

        # what ICA removed from the analysed epochs (the same epochs before and
        # after ICA, before Autoreject)
        width = int(EPOCH_DURATION * sfreq)
        before = np.stack(
            [raw_pre_ica.get_data(picks=SCALP_CHANNELS, start=o, stop=o + width) for o in kept_onsets]
        )
        after = np.stack(
            [raw.get_data(picks=SCALP_CHANNELS, start=o, stop=o + width) for o in kept_onsets]
        )
        f, p_before = welch(before, fs=sfreq, nperseg=width)
        _, p_after = welch(after, fs=sfreq, nperseg=width)
        alpha = (f >= 8) & (f <= 13)
        post = [SCALP_CHANNELS.index(c) for c in POSTERIOR_ICA_CHANNELS]
        front = [SCALP_CHANNELS.index(c) for c in FRONTAL_CHANNELS]
        qc["ica_posterior_alpha_retained"] = float(
            p_after[:, post][..., alpha].sum() / p_before[:, post][..., alpha].sum()
        )
        qc["ica_frontal_rms_retained"] = float(
            np.sqrt(after[:, front].var(axis=2).mean() / before[:, front].var(axis=2).mean())
        )

        # data quality and vigilance
        qc["blink_rate"] = blink_rate(raw_pre_ica)
        qc["spindle_rate"] = spindle_rate(raw, kept_onsets)
        qc["spindle_flag"] = qc["spindle_rate"] >= SPINDLE_RATE_FLAG

        x = kept.get_data()
        f, pxx = welch(x, fs=sfreq, nperseg=int(EPOCH_DURATION * sfreq))
        hf = pxx[:, :, (f >= 20) & (f <= 40)].mean(axis=(0, 2))
        qc["hf_power_db"] = float(10 * np.log10(hf.mean() * 1e12))
        qc["median_rms_uv"] = float(np.median(x.std(axis=2)) * 1e6)
        qc["avg_ref_residual"] = float(
            np.abs(x.sum(axis=1)).max() / np.abs(x).max()
        )

        kept.save(
            os.path.join(out_dir, f"sub-{sub_id}_task-rest_desc-clean_epo.fif"),
            overwrite=True,
        )
        print(f"---> {sub_id} cleaned ({len(kept)} epochs)")

    except Exception as e:
        qc["excluded"] = True
        qc["exclusion_reason"] = f"error: {e}"
        print(f"    ... {sub_id} failed: {e}")

    finally:
        with open(os.path.join(out_dir, f"sub-{sub_id}_qc.json"), "w") as f:
            json.dump(qc, f, indent=2, default=str)

    return qc


if __name__ == "__main__":
    events = pd.read_csv(os.path.join(DATA_DIR, "recording_events.csv"))
    subjects = sorted(events["id"].unique())

    results = Parallel(n_jobs=min(16, multiprocessing.cpu_count()), backend="loky", verbose=5)(
        delayed(process_subject)(s, events[events["id"] == s]) for s in subjects
    )

    print()
    print(f"---> Cleaned {sum(not r['excluded'] for r in results)} of {len(results)}")
    for r in results:
        if r["excluded"]:
            print(f"    - {r['id']} excluded: {r['exclusion_reason']}")

    errors = [r["id"] for r in results if r["exclusion_reason"].startswith("error")]
    if errors:
        raise RuntimeError(f"Processing failed for {', '.join(errors)}")
