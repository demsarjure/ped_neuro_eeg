# ped_neuro_eeg

Code and derived data for the study of resting-state EEG connectivity in children with CTNNB1 syndrome.

The analysis plan is in [analysis_spec.md](analysis_spec.md), frozen before any group comparison
(file hashes in `freeze_manifest.sha256`).

## Pipeline

Run in this order from the repository root (paths and parameters in `config.py`):

1. `recording_events.py`: recording breaks, photic stimulation and hyperventilation from the raw exports
2. `autoclean.py`: preprocessing and selection of the 100 analysed 1-s epochs per recording
3. `qc.py`: quality-control summary
4. `pdr.py`: dominant posterior rhythm, individual frequency band and background-slowing flag
5. `connectomes.py`: debiased weighted phase lag index connectomes
6. `metrics.py`: interhemispheric and quadrant strength, global efficiency
7. `analysis_group.R`, `analysis_age.R`, `analysis_sensitivity.R`: Bayesian models (cmdstanr)
8. `collect_numbers.py`, `figures.R`: reported numbers and figures

The raw EEG recordings are not public; they are available from the authors on reasonable request.
