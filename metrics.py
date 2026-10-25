"""
Connectome metrics calculation script.
"""

import os
from glob import glob

import bct
import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    CONNECTOME_DIR,
    SCALP_CHANNELS,
    BANDS,
    LEFT_CHANNELS,
    RIGHT_CHANNELS,
    LAQ_CHANNELS,
    LPQ_CHANNELS,
    RAQ_CHANNELS,
    RPQ_CHANNELS,
    FRONTAL_CHANNELS,
)


def calculate_average_strength(
    connectome: np.ndarray, channels_1: list[str], channels_2: list[str]
) -> float:
    """
    Average connection strength (ACS) between two sets of channels. Negative
    dbWPLI values are kept: the debiased estimator is unbiased, so averaging it
    gives an unbiased estimate of mean connectivity.

    Args:
        connectome (np.ndarray): The connectome matrix.
        channels_1 (list[str]): The first set of channels.
        channels_2 (list[str]): The second set of channels.
    """
    idx_1 = [SCALP_CHANNELS.index(c) for c in channels_1]
    idx_2 = [SCALP_CHANNELS.index(c) for c in channels_2]

    return float(connectome[np.ix_(idx_1, idx_2)].mean())


def calculate_within_strength(connectome: np.ndarray, channel_sets: list[list[str]]) -> float:
    """
    Average connection strength over the pairs within each set of channels
    (each pair counted once), pooled over the sets. Used post hoc for the
    connectivity within the hemispheres.

    Args:
        connectome (np.ndarray): The connectome matrix.
        channel_sets (list[list[str]]): The sets of channels.
    """
    values = []
    for channels in channel_sets:
        idx = [SCALP_CHANNELS.index(c) for c in channels]
        values.append(connectome[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)])

    return float(np.concatenate(values).mean())


def calculate_ge(connectome: np.ndarray) -> float:
    """
    Global efficiency of the weighted connectome. Negative dbWPLI values (no
    consistent phase lag) are set to zero, since edge lengths are the inverse
    of the weights and must be positive.

    Args:
        connectome (np.ndarray): The connectome matrix.
    """
    w = connectome.copy()
    w[w < 0] = 0
    np.fill_diagonal(w, 0)

    return float(bct.efficiency_wei(w))


def calculate_strength_without(
    connectome: np.ndarray, channels_1: list[str], channels_2: list[str], excluded: set[str]
) -> float:
    """
    Average connection strength leaving out pairs that involve an excluded
    channel (interpolated, S10; or frontal, S11).

    Args:
        connectome (np.ndarray): The connectome matrix.
        channels_1 (list[str]): The first set of channels.
        channels_2 (list[str]): The second set of channels.
        excluded (set[str]): Channels to leave out.
    """
    return calculate_average_strength(
        connectome,
        [c for c in channels_1 if c not in excluded],
        [c for c in channels_2 if c not in excluded],
    )


def process_connectomes(group: str, band: str, interpolated: dict) -> pd.DataFrame:
    """
    Calculate metrics for all connectomes of a group in a band.

    Args:
        group (str): The group ("test" or "control").
        band (str): The frequency band.
        interpolated (dict): Channels interpolated for the whole recording, by id.
    """
    results = []
    files = sorted(glob(os.path.join(CONNECTOME_DIR, group, band, "*.csv")))
    for file in files:
        sub_id = os.path.basename(file).split("_")[0]
        connectome = np.loadtxt(file, delimiter=",")
        upper = connectome[np.triu_indices(len(SCALP_CHANNELS), k=1)]

        results.append(
            {
                "id": sub_id,
                "group": group,
                "lh_rh": calculate_average_strength(connectome, LEFT_CHANNELS, RIGHT_CHANNELS),
                "lh_rh_noint": calculate_strength_without(
                    connectome, LEFT_CHANNELS, RIGHT_CHANNELS, interpolated.get(sub_id, set())
                ),
                # without the frontal-pole and anterior temporal electrodes (36 pairs)
                "lh_rh_nofront": calculate_strength_without(
                    connectome, LEFT_CHANNELS, RIGHT_CHANNELS, set(FRONTAL_CHANNELS)
                ),
                "laq_raq": calculate_average_strength(connectome, LAQ_CHANNELS, RAQ_CHANNELS),
                "lpq_rpq": calculate_average_strength(connectome, LPQ_CHANNELS, RPQ_CHANNELS),
                "laq_rpq": calculate_average_strength(connectome, LAQ_CHANNELS, RPQ_CHANNELS),
                "raq_lpq": calculate_average_strength(connectome, RAQ_CHANNELS, LPQ_CHANNELS),
                # post hoc: within the hemispheres, the same 16 electrodes (56 pairs)
                "intra": calculate_within_strength(connectome, [LEFT_CHANNELS, RIGHT_CHANNELS]),
                "ge": calculate_ge(connectome),
                "mean_conn": float(upper.mean()),
                "neg_frac": float((upper < 0).mean()),
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    dem = pd.concat(
        [
            pd.read_csv(os.path.join(DATA_DIR, "demographics_test.csv")),
            pd.read_csv(os.path.join(DATA_DIR, "demographics_control.csv")),
        ],
        ignore_index=True,
    )

    # channels interpolated for the whole recording (not independent measurements)
    qc = pd.read_csv(os.path.join(DATA_DIR, "qc_summary.csv"))
    interpolated = {
        row["id"]: set(str(row["bad_channels"]).split(";"))
        for _, row in qc.iterrows()
        if isinstance(row["bad_channels"], str) and row["bad_channels"]
    }

    for band in list(BANDS) + ["iaf"]:
        df = pd.concat([process_connectomes(g, band, interpolated) for g in ["test", "control"]])
        df = df.merge(dem, on="id", how="left")
        df.to_csv(os.path.join(DATA_DIR, f"connectome_metrics_{band}.csv"), index=False)

        print(f"---> {band}: metrics for {len(df)} recordings")
