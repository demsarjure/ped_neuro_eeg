"""
Quality control: collects the per-recording QC records of autoclean.py into a
summary table (failing if a record is missing or a recording failed), compares
data quality between groups, and (with --plots) draws one internal QC figure
per recording.
"""

import os
import sys
import json
from glob import glob

import numpy as np
import pandas as pd
import mne
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

from config import DERIV_ROOT, DATA_DIR, QC_DIR, SFREQ, EPOCH_DURATION, POSTERIOR_CHANNELS, AGE_RANGE

mne.set_log_level("ERROR")

GROUP_COLUMNS = [
    "duration_s",
    "n_breaks",
    "photic_excluded_s",
    "hv_excluded_s",
    "n_bad_channels",
    "n_ica_excluded",
    "candidate_epochs",
    "ar_rejected_epochs",
    "ar_interpolated_channel_epochs",
    "good_epochs",
    "kept_median_s",
    "kept_median_frac",
    "blink_rate",
    "spindle_rate",
    "hf_power_db",
    "median_rms_uv",
]


def load_summary() -> pd.DataFrame:
    """
    Collect the per-recording QC records. Fails if a recording has no record or
    its processing failed.
    """
    records = []
    for path in sorted(glob(os.path.join(DERIV_ROOT, "sub-*", "eeg", "sub-*_qc.json"))):
        with open(path) as f:
            record = json.load(f)
        for key, value in record.items():
            if isinstance(value, list):
                record[key] = ";".join(str(v) for v in value)
            elif isinstance(value, dict):
                record[key] = ";".join(f"{k}:{v}" for k, v in value.items())
        records.append(record)

    df = pd.DataFrame(records)

    events = pd.read_csv(os.path.join(DATA_DIR, "recording_events.csv"))
    missing = sorted(set(events["id"]) - set(df.get("id", [])))
    if missing:
        raise RuntimeError(f"No QC record for {', '.join(missing)}")
    failed = df.loc[df["exclusion_reason"].str.startswith("error"), "id"].tolist()
    if failed:
        raise RuntimeError(f"Processing failed for {', '.join(failed)}")

    dem = pd.concat(
        [
            pd.read_csv(os.path.join(DATA_DIR, "demographics_test.csv")),
            pd.read_csv(os.path.join(DATA_DIR, "demographics_control.csv")),
        ],
        ignore_index=True,
    )
    df = df.merge(dem, on="id", how="left")
    # only participants in the study age range
    df = df[df["age"].between(*AGE_RANGE)].reset_index(drop=True)
    resp = pd.read_csv(os.path.join(DATA_DIR, "respiration.csv"))
    df = df.merge(resp[["id", "resp_usable"]], on="id", how="left")
    df["excluded"] = df["excluded"].astype(bool)
    df["spindle_flag"] = df["spindle_flag"].fillna(False).astype(bool)

    return df


def group_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Median and interquartile range of QC measures per group (analysable
    recordings only).

    Args:
        df (pd.DataFrame): QC summary.
    """
    rows = []
    analysable = df[~df["excluded"]]
    for column in GROUP_COLUMNS:
        row = {"measure": column}
        for group in ["test", "control"]:
            x = analysable.loc[analysable["group"] == group, column].astype(float)
            q1, med, q3 = np.nanpercentile(x, [25, 50, 75])
            row[f"{group}_median"] = med
            row[f"{group}_q1"] = q1
            row[f"{group}_q3"] = q3
            row[f"{group}_n"] = int(x.notna().sum())
        rows.append(row)

    return pd.DataFrame(rows)


def plot_recording(sub_id: str, events: pd.DataFrame, record: pd.Series, pdr: pd.DataFrame):
    """
    One internal QC figure: recording timeline, spectrum and a trace sample.

    Args:
        sub_id (str): Subject identifier.
        events (pd.DataFrame): Recording events.
        record (pd.Series): QC record of the subject.
        pdr (pd.DataFrame): PDR table (may be empty).
    """
    epo_path = os.path.join(
        DERIV_ROOT, f"sub-{sub_id}", "eeg", f"sub-{sub_id}_task-rest_desc-clean_epo.fif"
    )
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    ev = events[events["id"] == sub_id]
    duration = ev.loc[ev["event"] == "recording", "offset_s"].iloc[0]

    # timeline
    ax = axes[0]
    for _, p in ev[ev["event"] == "photic"].iterrows():
        ax.axvspan(p["onset_s"], p["offset_s"], color="orange", alpha=0.4, lw=0)
    for _, h in ev[ev["event"] == "hv"].iterrows():
        ax.axvspan(h["onset_s"], h["offset_s"], color="purple", alpha=0.3, lw=0)
    for _, b in ev[ev["event"] == "break"].iterrows():
        ax.axvline(b["onset_s"], color="red", lw=1)
    if os.path.exists(epo_path):
        epochs = mne.read_epochs(epo_path)
        onsets = epochs.events[:, 0] / SFREQ
        ax.vlines(onsets, 0, 1, color="tab:blue", lw=0.5)
    ax.axvline(duration / 2, color="k", ls="--", lw=1)
    ax.set_xlim(0, duration)
    ax.set_yticks([])
    ax.set_xlabel("Time (s)")
    title = f"{sub_id} ({record['group']}, {record.get('age', '?')} y)"
    if record["excluded"]:
        title += f" EXCLUDED: {record['exclusion_reason']}"
    ax.set_title(title)

    if os.path.exists(epo_path):
        # spectrum
        data = epochs.get_data()
        f, pxx = welch(data, fs=SFREQ, nperseg=int(EPOCH_DURATION * SFREQ), nfft=500)
        pxx = pxx.mean(axis=0)
        post = pxx[[epochs.ch_names.index(c) for c in POSTERIOR_CHANNELS]].mean(axis=0)
        cent = pxx[[epochs.ch_names.index(c) for c in ["C3", "C4"]]].mean(axis=0)
        ax = axes[1]
        sel = (f >= 1) & (f <= 40)
        ax.semilogy(f[sel], post[sel] * 1e12, label="posterior")
        ax.semilogy(f[sel], cent[sel] * 1e12, label="central")
        row = pdr[pdr["id"] == sub_id]
        if len(row) and bool(row["pdr_clear"].iloc[0]):
            ax.axvline(row["pdr_hz"].iloc[0], color="k", ls=":", label="PDR")
        ax.axvspan(8, 13, color="grey", alpha=0.15)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (µV²/Hz)")
        ax.legend()

        # trace sample (first 10 epochs)
        ax = axes[2]
        x = np.concatenate(data[:10], axis=1) * 1e6
        offset = 100
        for i, ch in enumerate(epochs.ch_names):
            ax.plot(np.arange(x.shape[1]) / SFREQ, x[i] - i * offset, lw=0.5, color="k")
        ax.set_yticks([-i * offset for i in range(len(epochs.ch_names))])
        ax.set_yticklabels(epochs.ch_names, fontsize=7)
        ax.set_xlabel("Time (s, first 10 kept epochs)")

    fig.tight_layout()
    fig.savefig(os.path.join(QC_DIR, f"{sub_id}_qc.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    summary = load_summary()
    summary.to_csv(os.path.join(DATA_DIR, "qc_summary.csv"), index=False)
    comparison = group_comparison(summary)
    comparison.to_csv(os.path.join(DATA_DIR, "qc_group_comparison.csv"), index=False)

    print(f"---> QC summary of {len(summary)} recordings, {(~summary['excluded']).sum()} analysable")
    print(summary.loc[summary["excluded"], ["id", "exclusion_reason"]].to_string(index=False))
    print()
    print(comparison.round(2).to_string(index=False))

    if "--plots" in sys.argv:
        os.makedirs(QC_DIR, exist_ok=True)
        events = pd.read_csv(os.path.join(DATA_DIR, "recording_events.csv"))
        pdr_path = os.path.join(DATA_DIR, "pdr.csv")
        pdr = pd.read_csv(pdr_path) if os.path.exists(pdr_path) else pd.DataFrame(columns=["id"])
        for _, record in summary.iterrows():
            plot_recording(record["id"], events, record, pdr)
        print(f"---> QC figures saved to {QC_DIR}")
