"""
A helper script that loads the eeg data, keeps only eeg electrodes, and saves
the data in BIDS format.
"""

import os

import mne
from mne_bids import write_raw_bids, BIDSPath
import pandas as pd
import numpy as np


# a helper function to convert a single CSV file to BIDS
# ------------------------------------------------------------------------------
def convert_csv_to_bids(file_path, bids_root, subject_id, eeg_labels):
    """
    Load CSV data, create a RawArray, and write it in BIDS format.

    Parameters:
        file_path (str): Path to the CSV file.
        bids_root (str): The root folder where the BIDS dataset will be stored.
        subject_id (str): Subject identifier (e.g., 'C-001' or 'T-001').
        eeg_labels (pd.DataFrame): a dataframe with eeg labels and indexes.
    """
    print(f"---> Converting {subject_id} to BIDS")

    # load the data file
    print(f"    ... loading {file_path}")
    data_lines = []
    with open(file_path, "r", encoding="UTF-16") as f:
        for line in f:
            if not line.startswith("%"):
                data_lines.append(line.strip())

    # parse the data into a DataFrame
    print("    ... converting the data into appropriate format")
    columns = (
        ["Date.Time", "EB", "Stamp"]
        + [f"C{str(i).zfill(3)}" for i in range(1, 36)]
        + ["PHOTIC"]
    )
    data = pd.DataFrame([line.split("\t") for line in data_lines], columns=columns)

    # keep only eeg, eog, emg and ecg channels and rename
    data = data[eeg_labels["channel"].tolist()]
    data.columns = eeg_labels["label"].tolist()

    # to np array
    data_array = data.values.astype(np.float64)

    print(eeg_labels["label"].tolist())

    # mne info
    info = mne.create_info(
        ch_names=eeg_labels["label"].tolist(),
        ch_types=eeg_labels["description"].tolist(),
        sfreq=250.0,
    )

    # to RawArray
    raw = mne.io.RawArray(data_array.T, info)

    # channel locations
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore")

    # create a BIDSPath
    bids_path = BIDSPath(
        subject=subject_id,
        task="rest",
        datatype="eeg",
        root=bids_root,
    )

    # write to BIDS
    print("    ... saving")
    write_raw_bids(
        raw,
        bids_path=bids_path,
        overwrite=True,
        allow_preload=True,
        format="BrainVision",
    )

    print(f"   ... converted subject {subject_id} to BIDS")
    print()


# define common parameters
# ------------------------------------------------------------------------------
# eeg channel names
# setup labels
labels = pd.read_csv("./support_files/labels.csv")
labels = labels[labels["description"].isin(["eeg", "eog", "emg", "ecg"])]
labels = labels[["label", "channel", "description"]]

# loop through your files and convert them
# ------------------------------------------------------------------------------
# consts
BIDS_ROOT = "/mnt/d/work/ped_neuro_eeg_data/bids"
DATA_ROOT = "/mnt/d/work/ped_neuro_eeg_data"
GROUPS = ["control", "test"]

for group in GROUPS:
    group_data_dir = f"{DATA_ROOT}/{group}"

    for fname in os.listdir(group_data_dir):
        if fname.endswith(".txt"):
            convert_csv_to_bids(
                f"{group_data_dir}/{fname}",
                BIDS_ROOT,
                os.path.splitext(fname)[0],
                labels,
            )
