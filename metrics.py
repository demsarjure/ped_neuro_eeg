"""
Connectome metrics calculation script.
"""

import os
from glob import glob

import bct
import pandas as pd
import numpy as np

# config
CONNECTOME_DIR = os.path.join(".", "data", "connectomes")

# electrode groups
ELECTRODES = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
]
LEFT_ELECTRODES = ["Fp1", "F7", "F3", "T3", "C3", "T5", "P3", "O1"]
RIGHT_ELECTRODES = ["Fp2", "F8", "F4", "T4", "C4", "T6", "P4", "O2"]

# left anterior quadrant
LAQ_ELECTRODES = ["Fp1", "F3", "F7"]
# left posterior quadrant
LPQ_ELECTRODES = ["P3", "O1", "T5"]
# right anterior quadrant
RAQ_ELECTRODES = ["Fp2", "F4", "F8"]
# right posterior quadrant
RPQ_ELECTRODES = ["P4", "O2", "T6"]


def load_connectome(file_path: str) -> np.ndarray:
    """
    Load a connectome from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
    """
    try:
        connectome = pd.read_csv(file_path, header=None)
        connectome = connectome.values
        return connectome
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.ndarray([])


def calculate_average_strength(
    connectome: np.ndarray, electrode_array_1: list[str], electrode_array_2: list[str]
) -> float:
    """
    Calculate the average strength between two sets of electrodes.

    Args:
        connectome (np.ndarray): The connectome matrix.
        electrode_array_1 (list[str]): The first array of electrodes.
        electrode_array_2 (list[str]): The second array of electrodes.
    """
    strengths = []
    for e_1 in electrode_array_1:
        idx_1 = ELECTRODES.index(e_1)
        for e_2 in electrode_array_2:
            idx_2 = ELECTRODES.index(e_2)
            strengths.append(connectome[idx_1, idx_2])

    return float(np.mean(strengths))


def calculate_ge(connectome: np.ndarray) -> float:
    """
    Calculate the global efficiency of a connectome.

    Args:
        connectome (np.ndarray): The connectome matrix.
    """
    try:
        ge = bct.efficiency_wei(connectome)
        return ge
    except Exception as e:
        print(f"Error in calculating global efficiency: {e}")
        return np.nan


def process_connectomes(group: str, connectome_dir: str) -> pd.DataFrame:
    """
    Process all connectomes and calculate metrics.

    Args:
        group (str): The group to process ("test" or "control").
    """
    # get all connectomes
    connectome_files = glob(os.path.join(connectome_dir, "*.csv"))

    results = []

    for file in connectome_files:
        # get subject id from filename
        filename = os.path.splitext(os.path.basename(file))[0]
        subject_id = filename.split("_")[0]
        print(f"    - processing {subject_id}")

        # connectome
        connectome = load_connectome(file)

        # if nan in connectome or only contains 0 and 1, skip
        if not np.isnan(connectome).any() and np.unique(connectome).size > 2:
            # metrics
            lh_rh = calculate_average_strength(
                connectome, LEFT_ELECTRODES, RIGHT_ELECTRODES
            )
            laq_raq = calculate_average_strength(
                connectome, LAQ_ELECTRODES, RAQ_ELECTRODES
            )
            lpq_rpq = calculate_average_strength(
                connectome, LPQ_ELECTRODES, RPQ_ELECTRODES
            )
            laq_rpq = calculate_average_strength(
                connectome, LAQ_ELECTRODES, RPQ_ELECTRODES
            )
            raq_lpq = calculate_average_strength(
                connectome, RAQ_ELECTRODES, LPQ_ELECTRODES
            )
            ge = calculate_ge(connectome)

            # Append results
            results.append(
                {
                    "id": subject_id,
                    "group": group,
                    "lh_rh": lh_rh,
                    "laq_raq": laq_raq,
                    "lpq_rpq": lpq_rpq,
                    "laq_rpq": laq_rpq,
                    "raq_lpq": raq_lpq,
                    "ge": ge,
                }
            )

    # to df
    return pd.DataFrame(results)


for band in ["theta", "alpha"]:
    all_results = []
    for g in ["test", "control"]:
        print(f"---> Processing {band} connectomes for group: {g}")

        # calculate metrics
        cd = os.path.join(CONNECTOME_DIR, g, band)
        results_df = process_connectomes(g, cd)

        # append
        all_results.append(results_df)

    # append demographics
    dem_test = pd.read_csv(os.path.join(".", "data", "demographics_test.csv"))
    dem_control = pd.read_csv(os.path.join(".", "data", "demographics_control.csv"))
    demographics = pd.concat([dem_test, dem_control], ignore_index=True)

    # combine and save results
    all_results = pd.concat(all_results)
    all_results = all_results.merge(demographics, on="id", how="left")
    all_results.to_csv(
        os.path.join(".", "data", f"connectome_metrics_{band}.csv"), index=False
    )

    print()
    print("---> All results saved to connectome_metrics.csv")
