"""
Connectomes calculation script.
"""

import os
import numpy as np
import mne
from mne_connectivity import spectral_connectivity_epochs

# paths
GROUP = "test"  # or control
DATA_ROOT = "../ped_neuro_eeg_data"
CLEANED_DIR = os.path.join(DATA_ROOT, "cleaned", GROUP)
CONNECTOME_DIR = os.path.join(DATA_ROOT, "connectomes", GROUP)

# frequencies - extended alpha band
LOW_FREQ = 7
HIGH_FREQ = 14

# create output directory
os.makedirs(CONNECTOME_DIR, exist_ok=True)

# Find all cleaned EEG files
eeg_files = []
for filename in os.listdir(CLEANED_DIR):
    if filename.endswith("_eeg.fif"):
        eeg_files.append(os.path.join(CLEANED_DIR, filename))

print(f"Found {len(eeg_files)} EEG files to process")

# Process each file
for eeg_file in eeg_files:
    print(f"---> Processing {eeg_file}")

    # get subject id
    basename = os.path.basename(eeg_file)
    sub_id = basename.split("_")[0]

    # load
    raw = mne.io.read_raw_fif(eeg_file, preload=True)

    # preprocessing
    raw.filter(LOW_FREQ, HIGH_FREQ)
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)

    # dPLI
    result = spectral_connectivity_epochs(
        epochs,
        method="wpli2_debiased",
        mode="multitaper",
        fmin=LOW_FREQ,
        fmax=HIGH_FREQ,
        faverage=True,
        n_jobs=1,
    )

    # get the con_matrix
    con_matrix = result.get_data(output="dense")[:, :, 0]
    ch_names = epochs.ch_names
    symmetric_matrix = con_matrix + con_matrix.T
    np.fill_diagonal(symmetric_matrix, 0)

    # store as csv
    output_filename = f"{sub_id}_task-rest_connectome_eeg.csv"
    output_path = os.path.join(CONNECTOME_DIR, output_filename)
    np.savetxt(output_path, symmetric_matrix, delimiter=",")

    print(f"    - saved connectome for {basename} to {output_filename}")

print()
print("---> All connectomes have been generated!")
