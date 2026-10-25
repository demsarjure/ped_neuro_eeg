"""
Extracts recording events (length, recording breaks, photic stimulation and
hyperventilation) from the raw Natus text exports.

All sample indices are on the time axis of the BIDS data: csv_to_bids.py keeps
every non-header line of the export as one sample, including the
"--- BREAK IN DATA ---" rows, which become NaN samples.

Hyperventilation is detected from the respiration trace (export column C032,
'DIH, respiratory'), which csv_to_bids.py does not convert. The trace is
assessed only if it shows a breathing rhythm (its spectrum peaks at 0.15-1 Hz).
Hyperventilation is a run of at least 60 s in which the breathing amplitude
(RMS of the 0.1-1.5 Hz band-passed trace in 10-s windows, 1-s steps) is at
least twice the recording's median.
"""

import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

from config import DATA_ROOT, DATA_DIR, RAW_GROUPS, SFREQ, study_ids

# the photic channel is exported in 4-s blocks; in every block with flashes,
# the first k samples are ON (k = number of flashes in the block)
PHOTIC_BLOCK_S = 4.0
# photic blocks closer than this are merged into one interval
PHOTIC_MERGE_S = 30.0

# respiration trace: field of C032 in a data row (Date.Time, EB, Stamp,
# C001-C035, PHOTIC)
RESP_FIELD = 34
RESP_BAND = (0.1, 1.5)
# a breathing rhythm: the spectrum (median over 30-s windows) has its maximum
# within RESP_BAND at 9-60 breaths per minute
RESP_SPECTRUM_S = 30.0
BREATHING_BAND = (0.15, 1.0)
# hyperventilation: breathing amplitude (RMS in 10-s windows, 1-s steps) at
# least twice the recording's median for at least 60 s
HV_WINDOW_S = 10.0
HV_RATIO = 2.0
HV_MIN_S = 60.0


def detect_hv(resp: np.ndarray) -> tuple[bool, list[tuple[int, int]]]:
    """
    Decide whether the respiration trace shows a breathing rhythm and, if it
    does, return its hyperventilation intervals (onset and offset samples).

    Args:
        resp (np.ndarray): Respiration trace, NaN at breaks.
    """
    good = ~np.isnan(resp)
    if not good.any():
        return False, []
    x = np.interp(np.arange(resp.size), np.flatnonzero(good), resp[good])

    # breathing rhythm
    f, pxx = welch(x, fs=SFREQ, nperseg=int(RESP_SPECTRUM_S * SFREQ), average="median")
    band = (f >= RESP_BAND[0]) & (f <= RESP_BAND[1])
    peak = f[band][np.argmax(pxx[band])]
    if not BREATHING_BAND[0] <= peak <= BREATHING_BAND[1]:
        return False, []

    # breathing amplitude at each second: RMS of the band-passed trace in the
    # surrounding 10 s
    sos = butter(2, RESP_BAND, btype="band", fs=SFREQ, output="sos")
    energy = np.concatenate([[0.0], np.cumsum(sosfiltfilt(sos, x) ** 2)])
    width, step = int(HV_WINDOW_S * SFREQ), int(SFREQ)
    starts = np.arange(0, x.size - width + 1, step)
    rms = np.sqrt((energy[starts + width] - energy[starts]) / width)

    # runs of high amplitude; each window stands for the second at its centre
    high = np.concatenate([[0], rms >= HV_RATIO * np.median(rms), [0]]).astype(int)
    run_starts = np.flatnonzero(np.diff(high) == 1)
    run_ends = np.flatnonzero(np.diff(high) == -1)
    hv = [
        (int(starts[a] + width // 2), int(starts[b - 1] + width // 2 + step))
        for a, b in zip(run_starts, run_ends)
        if (b - a) * step >= HV_MIN_S * SFREQ
    ]

    return True, hv


def parse_export(file_path: str) -> dict:
    """
    Parse one raw export and return its events.

    Args:
        file_path (str): Path to the UTF-16 text export.
    """
    sub_id = os.path.splitext(os.path.basename(file_path))[0]
    group = "test" if sub_id.startswith("T") else "control"

    n_samples = 0
    breaks = []
    photic_runs = []
    in_photic = False
    start_time = None
    resp = []

    with open(file_path, "r", encoding="UTF-16") as f:
        for line in f:
            if line.startswith("%"):
                continue

            if line.startswith("---"):
                breaks.append(n_samples)
                in_photic = False
                resp.append(np.nan)
            else:
                fields = line.rstrip().split("\t")
                if start_time is None:
                    start_time = fields[0]
                is_on = fields[-1].strip() == "ON"
                if is_on and not in_photic:
                    photic_runs.append(n_samples)
                in_photic = is_on
                try:
                    resp.append(float(fields[RESP_FIELD]))
                except ValueError:
                    resp.append(np.nan)

            n_samples += 1

    # photic blocks start where an ON run starts and last one export block
    block = int(PHOTIC_BLOCK_S * SFREQ)
    merge_gap = int(PHOTIC_MERGE_S * SFREQ)
    photic = []
    for onset in photic_runs:
        offset = onset + block
        if photic and onset - photic[-1][1] <= merge_gap:
            photic[-1][1] = max(photic[-1][1], offset)
        else:
            photic.append([onset, offset])

    resp_usable, hv = detect_hv(np.array(resp))

    events = [("recording", 0, n_samples)]
    events += [("break", b, b + 1) for b in breaks]
    events += [("photic", min(on, n_samples), min(off, n_samples)) for on, off in photic]
    events += [("hv", on, off) for on, off in hv]

    rows = [
        {
            "id": sub_id,
            "group": group,
            "event": event,
            "onset_sample": onset,
            "offset_sample": offset,
            "onset_s": onset / SFREQ,
            "offset_s": offset / SFREQ,
        }
        for event, onset, offset in events
    ]
    respiration = {
        "id": sub_id,
        "group": group,
        "resp_usable": resp_usable,
        "n_hv": len(hv),
        "hv_s": sum(off - on for on, off in hv) / SFREQ,
    }

    return {"rows": rows, "respiration": respiration, "start_time": start_time, "group": group}


if __name__ == "__main__":
    files = []
    for g in RAW_GROUPS:
        group_dir = os.path.join(DATA_ROOT, g)
        files += [
            os.path.join(group_dir, f)
            for f in sorted(os.listdir(group_dir))
            if f.endswith(".txt")
        ]

    print(f"---> Parsing {len(files)} raw exports")
    with ProcessPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(parse_export, files))

    # only participants in the study age range
    keep = study_ids()
    results = [r for r in results if r["respiration"]["id"] in keep]

    df = pd.DataFrame([row for r in results for row in r["rows"]])
    df = df.sort_values(["id", "onset_sample", "event"]).reset_index(drop=True)
    output_path = os.path.join(DATA_DIR, "recording_events.csv")
    df.to_csv(output_path, index=False)

    resp = pd.DataFrame([r["respiration"] for r in results]).sort_values("id")
    resp_path = os.path.join(DATA_DIR, "respiration.csv")
    resp.to_csv(resp_path, index=False)

    # recording dates are reported as a range only, never stored
    for g in RAW_GROUPS:
        dates = [
            datetime.strptime(r["start_time"], "%m/%d/%Y %H:%M:%S")
            for r in results
            if r["group"] == g and r["start_time"]
        ]
        print(f"    - {g}: recordings from {min(dates):%Y-%m} to {max(dates):%Y-%m}")

        d = resp[resp["group"] == g]
        print(
            f"    - {g}: respiration usable in {d['resp_usable'].sum()} of {len(d)}, "
            f"hyperventilation in {(d['n_hv'] > 0).sum()}"
        )

    print(f"---> Events saved to {output_path}")
    print(f"---> Respiration summary saved to {resp_path}")
