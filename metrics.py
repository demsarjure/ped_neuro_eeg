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
        return None


def calculate_ihs(connectome: np.ndarray) -> float:
    """
    Calculate the interhemispheric strength of a connectome.

    Args:
        connectome (np.ndarray): The connectome matrix.
    """
    # sum of weights between left and right hemisphere electrodes
    ihs = 0

    for le in LEFT_ELECTRODES:
        l_idx = LEFT_ELECTRODES.index(le)
        for re in RIGHT_ELECTRODES:
            r_idx = RIGHT_ELECTRODES.index(re)
            ihs += connectome[l_idx, r_idx]

    return ihs


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

        # ihs
        ihs = calculate_ihs(connectome)
        ge = calculate_ge(connectome)

        # Append results
        results.append(
            {
                "id": subject_id,
                "group": group,
                "ihs": ihs,
                "ge": ge,
            }
        )

    # to df
    return pd.DataFrame(results)


all_results = []
for g in ["test", "control"]:
    print(f"---> Processing connectomes for group: {g}")

    # calculate metrics
    cd = os.path.join(CONNECTOME_DIR, g)
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
all_results.to_csv(os.path.join(".", "data", "connectome_metrics.csv"), index=False)

print()
print("---> All results saved to connectome_metrics.csv")
