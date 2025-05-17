"""
Connectomes calculation script.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs


# paths
BIDS_ROOT = "/mnt/d/work/ped_neuro_eeg_data/bids"
DATA_ROOT = "/mnt/d/work/ped_neuro_eeg_data"
CONNECTOME_DIR = os.path.join(".", "data", "connectomes")
GROUPS = ["test", "control"]

# epoch
EPOCH = 12

# frequencies - extended alpha band
LOW_FREQ = 7
HIGH_FREQ = 14

# create output directory
os.makedirs(os.path.join(CONNECTOME_DIR, "control"), exist_ok=True)
os.makedirs(os.path.join(CONNECTOME_DIR, "test"), exist_ok=True)

# get subjects
subjects = [subject for subject in os.listdir(BIDS_ROOT) if subject.startswith("sub-")]

def process_subject(subject):
    sub_id = subject.split("-")[1]
    group = "control"
    if sub_id.startswith("T"):
        group = "test"

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

    # preprocessing
    raw.filter(LOW_FREQ, HIGH_FREQ)

    # 20 minutes = 100 epochs
    total_duration = raw.times[-1]
    middle_point = total_duration / 2
    t_min = max(0, middle_point - (50 * EPOCH))
    t_max = min(total_duration, middle_point + (50 * EPOCH))
    raw.crop(tmin=t_min, tmax=t_max)

    # epoch
    epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH, preload=True)

    # dPLI
    result = spectral_connectivity_epochs(
        epochs,
        method="wpli2_debiased",
        mode="multitaper",
        fmin=LOW_FREQ,
        fmax=HIGH_FREQ,
        n_jobs=1,
    )

    # get the con_matrix
    con_matrix = result.get_data(output="dense")[:, :, 0]
    symmetric_matrix = con_matrix + con_matrix.T
    np.fill_diagonal(symmetric_matrix, 0)

    # store as csv
    output_filename = f"{sub_id}_task-rest_connectome_eeg.csv"
    output_path = os.path.join(CONNECTOME_DIR, group, output_filename)
    np.savetxt(output_path, symmetric_matrix, delimiter=",")

    print(f"---> Connectome saved as {output_filename}")

with ProcessPoolExecutor() as executor:
    list(executor.map(process_subject, subjects))

print()
print(f"---> All connectomes have been generated!")
