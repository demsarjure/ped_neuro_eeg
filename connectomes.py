"""
Connectomes calculation script.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs
from functools import partial


# paths
BIDS_ROOT = "/mnt/d/work/ped_neuro_eeg_data/bids"
DATA_ROOT = "/mnt/d/work/ped_neuro_eeg_data"
CONNECTOME_DIR = os.path.join(".", "data", "connectomes")
GROUPS = ["test", "control"]


# frequencies - extended alpha band
BANDS = {
    "theta": [3, 7],
    "alpha": [7, 14],
}

# create output directory
os.makedirs(os.path.join(CONNECTOME_DIR, "control", "theta"), exist_ok=True)
os.makedirs(os.path.join(CONNECTOME_DIR, "control", "alpha"), exist_ok=True)
os.makedirs(os.path.join(CONNECTOME_DIR, "test", "theta"), exist_ok=True)
os.makedirs(os.path.join(CONNECTOME_DIR, "test", "alpha"), exist_ok=True)

# get subjects
subjects = [subject for subject in os.listdir(BIDS_ROOT) if subject.startswith("sub-")]


def process_subject(subject, band):
    sub_id = subject.split("-")[1]
    group = "control"
    if sub_id.startswith("T"):
        group = "test"

    # get frequencies
    low_freq, high_freq = BANDS[band]

    # get the cleaned eeg file
    eeg_file = os.path.join(
        BIDS_ROOT,
        subject,
        "eeg",
        f"{subject}_task-rest_cleaned_eeg.fif",
    )
    basename = os.path.basename(eeg_file)

    # load
    raw = mne.io.read_raw_fif(eeg_file, preload=True)

    # 20 minutes = 100 epochs
    total_duration = raw.times[-1]
    middle_point = total_duration / 2
    t_min = max(0, middle_point - (50 * 12))
    t_max = min(total_duration, middle_point + (50 * 12))
    raw.crop(tmin=t_min, tmax=t_max)

    # epoch
    epochs = mne.make_fixed_length_epochs(
        raw, duration=12, overlap=12 / 2, preload=True
    )

    # dPLI
    result = spectral_connectivity_epochs(
        epochs,
        method="wpli2_debiased",
        mode="multitaper",
        fmin=low_freq,
        fmax=high_freq,
        faverage=True,
        n_jobs=1,
    )

    # convert to a 2D connectome matrix
    n_channels = len(result.names)
    con_vals = result.get_data().reshape(n_channels, n_channels)
    conn_matrix = con_vals + con_vals.T

    # store as csv
    output_filename = f"{sub_id}_{band}_task-rest_connectome_eeg.csv"
    output_path = os.path.join(CONNECTOME_DIR, group, band, output_filename)
    np.savetxt(output_path, conn_matrix, delimiter=",")

    print(f"---> Connectome saved as {output_filename}")


with ProcessPoolExecutor() as executor:
    for band_name in BANDS.keys():
        partial_subject_processor = partial(process_subject, band=band_name)
        list(executor.map(partial_subject_processor, subjects))

print()
print(f"---> All connectomes have been generated!")
