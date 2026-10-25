"""
Post hoc check of eye state, not part of the analysis plan (25 October 2026).

Eye opening and closing were not annotated in the exports. Blinks show that the
eyes were open, so blinks were located around the analysed epochs with the
detector of the blink rate (autoclean.py: vertical EOG proxy, Fp1/Fp2 minus
C3/C4, 1-8 Hz, 75 uV, before ICA, the recording's blink polarity). An analysed
epoch counts as near a blink if a blink lies within BLINK_WINDOW_S of its
middle; long blink-free stretches may mean closed eyes.
"""

import os
import multiprocessing

import numpy as np
import pandas as pd
import mne
from joblib import Parallel, delayed
from scipy.signal import find_peaks

from config import DATA_DIR, DERIV_ROOT, EPOCH_DURATION, L_FREQ, H_FREQ
from autoclean import BLINK_HEIGHT, load_raw, mark_breaks, mark_exclusions, usable_mask

BLINK_WINDOW_S = 15.0
# a stricter window, as a check
BLINK_WINDOW_STRICT_S = 5.0


def blink_times(raw: mne.io.BaseRaw) -> np.ndarray:
    """
    Times (s) of the blinks in the usable data, in the recording's blink polarity.

    Args:
        raw (mne.io.BaseRaw): Filtered, average-referenced, annotated data before ICA.
    """
    sfreq = raw.info["sfreq"]
    x = raw.get_data(picks=["Fp1", "Fp2", "C3", "C4"])
    v = x[:2].mean(axis=0) - x[2:].mean(axis=0)
    v = mne.filter.filter_data(v, sfreq, l_freq=1.0, h_freq=8.0, verbose=False)

    usable = usable_mask(raw)
    best = np.array([], dtype=int)
    for polarity in [1, -1]:
        peaks, _ = find_peaks(polarity * v, height=BLINK_HEIGHT, distance=int(0.3 * sfreq))
        peaks = peaks[usable[peaks]]
        if len(peaks) > len(best):
            best = peaks

    return best / sfreq


def process_subject(sub_id: str, events: pd.DataFrame, bads: list[str]) -> dict:
    """
    Blinks around the analysed epochs of one recording.

    Args:
        sub_id (str): Subject identifier.
        events (pd.DataFrame): Recording events of the subject.
        bads (list[str]): Electrodes interpolated in the pipeline.
    """
    # the pipeline's data before ICA (autoclean.py)
    raw = load_raw(sub_id)
    breaks = mark_breaks(raw)
    mark_exclusions(raw, breaks, events[events["event"] == "photic"], events[events["event"] == "hv"])
    if bads:
        raw.info["bads"] = bads
        raw.interpolate_bads(reset_bads=True)
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, picks="all", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)

    blinks = blink_times(raw)
    epochs = mne.read_epochs(
        os.path.join(DERIV_ROOT, f"sub-{sub_id}", "eeg", f"sub-{sub_id}_task-rest_desc-clean_epo.fif"),
        verbose=False,
    )
    middles = (epochs.events[:, 0] - raw.first_samp) / raw.info["sfreq"] + EPOCH_DURATION / 2
    distance = np.array([np.abs(blinks - m).min() if blinks.size else np.inf for m in middles])

    first, last = middles.min() - EPOCH_DURATION / 2, middles.max() + EPOCH_DURATION / 2
    in_span = blinks[(blinks >= first) & (blinks <= last)]

    return {
        "id": sub_id,
        "blinks_analysed_span": int(in_span.size),
        "analysed_span_s": float(last - first),
        "epochs_near_blink": float((distance <= BLINK_WINDOW_S).mean()),
        "epochs_near_blink_5s": float((distance <= BLINK_WINDOW_STRICT_S).mean()),
    }


if __name__ == "__main__":
    qc = pd.read_csv(os.path.join(DATA_DIR, "qc_summary.csv"))
    events = pd.read_csv(os.path.join(DATA_DIR, "recording_events.csv"))
    analysable = qc[~qc["excluded"]]

    rows = Parallel(n_jobs=min(16, multiprocessing.cpu_count()), backend="loky")(
        delayed(process_subject)(
            row.id,
            events[events["id"] == row.id],
            [c for c in str(row.bad_channels).split(";") if c and c != "nan"],
        )
        for row in analysable.itertuples()
    )

    df = pd.DataFrame(rows).merge(qc[["id", "group"]], on="id")
    df["blinks_per_min"] = df["blinks_analysed_span"] / (df["analysed_span_s"] / 60)
    df.to_csv(os.path.join(DATA_DIR, "blinks_analysed.csv"), index=False)

    print(df.groupby("group")[["blinks_per_min", "epochs_near_blink", "analysed_span_s"]].describe().T.round(2))
