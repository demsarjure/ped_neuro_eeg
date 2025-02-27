"""
The auto cleanup script for EEG data.
"""

import os

import numpy as np
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import ICA


# consts
BIDS_ROOT = "../../ped_neuro_eeg_data/bids"
OUT_DIR = "../../ped_neuro_eeg_data/clean"

# get subjects
subjects = [sub for sub in os.listdir(BIDS_ROOT) if sub.startswith("sub-")]

for sub in subjects:
    subj_id = sub.split("-")[1]

    bids_path = BIDSPath(
        root=BIDS_ROOT,
        subject=subj_id,
        datatype="eeg",
        task="rest",
        extension=".vhdr",
    )

    # load the data
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()

    # band-pass
    raw.filter(l_freq=1.0, h_freq=40.0)
    raw._data = np.nan_to_num(raw.get_data())

    # ICA
    ica = ICA(n_components=20, random_state=42, max_iter="auto")
    ica.fit(raw)
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=["Fp1", "Fp2"])
    print(f"Components identified as EOG artifacts: {eog_indices}")
    # eog_indices, eog_scores = ica.find_bads_eog(raw)
    ica.exclude = eog_indices
    raw_clean = ica.apply(raw.copy())

    # Save the cleaned data (using the FIF format in this example)
    out_fname = f"{sub}_cleaned_raw.fif"
    raw_clean.save(out_fname, overwrite=True)

    os._exit(1)
