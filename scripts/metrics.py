"""
Connectome metrics calculation script.
"""

import os
from glob import glob

import bct
import pandas as pd
import numpy as np

# config
DATA_ROOT = "../../ped_neuro_eeg_data"

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


def calculate_modularity(connectome):
    """
    Calculate the modularity of a connectome.

    Args:
        connectome (numpy.ndarray): The connectome matrix.

    Returns:
        float: The modularity value or np.nan if calculation fails.
    """
    try:
        _, q = bct.community_louvain(connectome, B="negative_sym")

        return q
    except Exception as e:
        print(f"Error calculating modularity: {e}")
        return np.nan


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
        # get subject ID from filename
        filename = os.path.splitext(os.path.basename(file))[0]
        subject_id = filename.split("_")[0]
        print(f"    - processing {subject_id}")

        # connectome
        connectome = load_connectome(file)

        # metrics
        modularity = calculate_modularity(connectome)
        ih_strength = calculate_ihs(connectome)

        # Append results
        results.append(
            {
                "subject_id": subject_id,
                "group": group,
                "modularity": modularity,
                "ihs": ih_strength,
            }
        )

    # to df
    return pd.DataFrame(results)


all_results = []
for g in ["test", "control"]:
    print(f"---> Processing connectomes for group: {g}")

    # calculate metrics
    cd = os.path.join(DATA_ROOT, "connectomes", g)
    results_df = process_connectomes(g, cd)

    # append
    all_results.append(results_df)


# combine and save results
all_results = pd.concat(all_results)
all_results.to_csv(
    os.path.join(DATA_ROOT, "metrics", "connectome_metrics.csv"), index=False
)
print()
print("---> All results saved to connectome_metrics.csv")

# t test
from scipy.stats import ttest_ind

print()
print("---> T-test results:")
# modularity
mod_test = all_results[all_results["group"] == "test"]["modularity"]
mod_control = all_results[all_results["group"] == "control"]["modularity"]
modularity_test = ttest_ind(mod_test, mod_control)
print("    - Modularity:")
print(f"        - mean test: {np.mean(mod_test)}")
print(f"        - mean control: {np.mean(mod_control)}")
print(f"        - t-statistic: {modularity_test.statistic}")
print(f"        - p-value: {modularity_test.pvalue}")

# interhemispheric strength
ihs_test = all_results[all_results["group"] == "test"]["ihs"]
ihs_control = all_results[all_results["group"] == "control"]["ihs"]
ih_strength_test = ttest_ind(ihs_test, ihs_control)
print("    - IHS:")
print(f"        - mean test: {np.mean(ihs_test)}")
print(f"        - mean control: {np.mean(ihs_control)}")
print(f"        - t-statistic: {ih_strength_test.statistic}")
print(f"        - p-value: {ih_strength_test.pvalue}")
print()
print("---> Done!")
