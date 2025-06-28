"""
The auto cleanup script for EEG data.
"""

import os
import multiprocessing
from joblib import Parallel, delayed

import mne
import numpy as np
from mne_bids import BIDSPath, read_raw_bids
from mne.preprocessing import ICA
from autoreject import get_rejection_threshold


# consts
BIDS_ROOT = "/mnt/d/work/ped_neuro_eeg_data/bids"

# get subjects
subjects = [subject for subject in os.listdir(BIDS_ROOT) if subject.startswith("sub-")]


def process_subject(subject):
    sub_id = subject.split("-")[1]

    bids_path = BIDSPath(
        root=BIDS_ROOT,
        subject=sub_id,
        datatype="eeg",
        task="rest",
        extension=".vhdr",
    )

    # load the data
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()

    # keep 100 middle 1s epochs
    total_duration = raw.times[-1]
    middle_point = total_duration / 2
    t_min = max(0, middle_point - 50)
    t_max = min(total_duration, middle_point + 50)
    raw.crop(tmin=t_min, tmax=t_max)

    # band-pass
    raw.filter(l_freq=0.5, h_freq=40)
    raw._data[:] = np.nan_to_num(raw.get_data())

    # re-reference to average reference
    raw.set_eeg_reference("average", projection=False)

    # ICA
    try:
        ica = ICA(n_components=None, max_iter="auto")
        ica.fit(raw)
        eog_indices, eog_scores = ica.find_bads_eog(raw)
        try:
            ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
        except TypeError as e:
            print(f"    ... ECG artifact detection failed: {e}{str(e)}")
            ecg_indices = []
        ica.exclude = list(set(eog_indices) | set(ecg_indices))
        clean = ica.apply(raw.copy())

        # keep only EEG channels, drop also A1 and A2
        clean.pick(picks="eeg", exclude=["A1", "A2"])

        # find bad channels
        epochs = mne.make_fixed_length_epochs(clean, duration=1)
        reject = get_rejection_threshold(epochs)
        epochs.drop_bad(reject=reject)
        bads = epochs.info["bads"]

        if bads:
            clean.info["bads"] = bads
            clean.interpolate_bads(reset_bads=True)

        # save
        save_path = os.path.join(
            BIDS_ROOT,
            subject,
            "eeg",
            f"{subject}_task-rest_cleaned_eeg.fif",
        )
        clean.save(save_path, overwrite=True)

        print()
        print(
            "--------------------------------------------------------------------------------"
        )
        print(f"---> {subject} cleaned")
        print(
            "--------------------------------------------------------------------------------"
        )
        print()
    except Exception as e:
        print(f"    ... {subject} failed: {e}")
        print()


if __name__ == "__main__":
    Parallel(n_jobs=multiprocessing.cpu_count(), backend="loky", verbose=5)(
        delayed(process_subject)(subject) for subject in subjects
    )
