# ped_neuro_eeg

Code and derived data for the study of resting-state EEG connectivity in children with CTNNB1 syndrome.
The analysed sample is children aged 2–9 years (`AGE_RANGE` in `config.py`).

The analysis plan is in [analysis_spec.md](analysis_spec.md), frozen before any group comparison
(file hashes in `freeze_manifest.sha256`) and amended once afterwards (see its last section).

## Pipeline

Run in this order from the repository root (paths and parameters in `config.py`), after converting the raw
text exports to BIDS with `csv_to_bids.py` (paths set in the script):

1. `recording_events.py`: recording breaks, photic stimulation and hyperventilation from the raw exports
2. `autoclean.py`: preprocessing and selection of the 100 analysed 1-s epochs per recording
3. `qc.py`: quality-control summary
4. `pdr.py`: dominant posterior rhythm, individual frequency band and background-slowing flag
5. `connectomes.py`: debiased weighted phase lag index connectomes
6. `metrics.py`: interhemispheric and quadrant strength, global efficiency
7. `analysis_group.R`, `analysis_age.R`, `analysis_sensitivity.R`: Bayesian models (cmdstanr); `analysis_posthoc.R`: post hoc analyses, not part of the plan (after `blinks_posthoc.py`, `noise_posthoc.py`, `spectra_posthoc.py`, `epochs_posthoc.py` and `reliability_posthoc.py`, which need the raw data or the cleaned epochs)
8. `collect_numbers.py`, `figures.R`: reported numbers and figures (Figures 4–6, Supplementary Figures 1–4)

Figures 1–3 are drawn in `figures/visualizations.af` (Affinity), with the numbers of Figures 2 and 3 from
`results/fig2_numbers.csv`, and exported as `figures/electrodes.pdf`, `figures/ihs.pdf` and `figures/all_quadrants.pdf`.

Only derived data are public; the raw EEG recordings are kept private because of their sensitivity.
