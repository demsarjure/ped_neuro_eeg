"""
The auto cleanup script for EEG data.
"""

import os

import mne
import numpy as np
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import ICA
from autoreject import get_rejection_threshold

# consts
BIDS_ROOT = "../ped_neuro_eeg_data/bids"

# get subjects
subjects = [sub for sub in os.listdir(BIDS_ROOT) if sub.startswith("sub-")]

skip_until = "sub-C046"

for sub in subjects:
    if skip_until and sub != skip_until:
        continue
    skip_until = None

    print(f"---> Cleaning {sub} [{subjects.index(sub) + 1}/{len(subjects)}]")

    subj_id = sub.split("-")[1]

    group = "control"
    if subj_id.startswith("T"):
        group = "test"

    bids_path = BIDSPath(
        root=BIDS_ROOT,
        subject=subj_id,
        datatype="eeg",
        task="rest",
        extension=".vhdr",
    )

    # load the data
    print("    ... loading data")
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()

    # band-pass
    print("    ... filtering")
    raw.filter(l_freq=1.0, h_freq=40.0)
    raw._data = np.nan_to_num(raw.get_data())

    # re-reference to average reference
    print("    ... applying average reference")
    raw.set_eeg_reference("average", projection=False)

    # find bad channels
    print("    ... detecting bad channels")
    epochs = mne.make_fixed_length_epochs(raw, duration=1.0, overlap=0.5)
    reject = get_rejection_threshold(epochs)
    epochs.drop_bad(reject=reject)
    bads = epochs.info["bads"]

    if bads:
        print(f"    ... detected bad channels: {bads}")
        print("    ... interpolating bad channels")
        raw.info["bads"] = bads
        raw.interpolate_bads(reset_bads=True)

    # ICA
    print("    ... ICA")
    ica = ICA(n_components=20, max_iter="auto")
    ica.fit(raw)
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    try:
        ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
    except TypeError as e:
        print(f"    ... ECG artifact detection failed: {e}{str(e)}")
        ecg_indices = []
    print(f"    ... components identified as EOG artifacts: {eog_indices}")
    print(f"    ... components identified as ECG artifacts: {ecg_indices}")
    ica.exclude = list(set(eog_indices) | set(ecg_indices))
    clean = ica.apply(raw.copy())

    # keep only middle 600 s of data
    print("    ... keeping only middle 600 s of data")
    total_duration = clean.times[-1]
    middle_point = total_duration / 2
    clean.crop(tmin=middle_point - 300, tmax=middle_point + 300)

    # keep only EEG channels, drop also A1 and A2
    print("    ... keeping only EEG channels")
    clean.pick(picks="eeg", exclude=["A1", "A2"])

    # save
    print("    ... saving cleaned data")
    save_path = os.path.join(
        "..",
        "ped_neuro_eeg_data",
        "cleaned",
        group,
        f"{sub}_task-rest_cleaned_eeg.fif",
    )
    clean.save(save_path, overwrite=True)
