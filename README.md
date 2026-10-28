# ped_neuro_eeg

Code and derived data for the paper "Electroencephalographic functional brain connectivity patterns in children
living with the CTNNB1 neurodevelopmental syndrome". The analysed sample is children aged 2–9 years (`AGE_RANGE`
in `config.py`).

## Pipeline

Run in this order from the repository root (paths and parameters in `config.py`), after converting the raw
text exports to BIDS with `csv_to_bids.py` (paths set in the script):

1. `recording_events.py`: recording breaks, photic stimulation and hyperventilation from the raw exports
2. `autoclean.py`: preprocessing and selection of the 100 analysed 1-s epochs per recording
3. `qc.py`: quality-control summary
4. `pdr.py`: posterior alpha peak, individual alpha band and the flag for a weak or slow posterior alpha peak
5. `connectomes.py`: debiased weighted phase lag index connectomes
6. `metrics.py`: interhemispheric and quadrant strength, global efficiency
7. `analysis_group.R`, `analysis_age.R`, `analysis_sensitivity.R`: Bayesian models (Stan, cmdstanr)
8. `figures.R`: Figs 5–8

Steps 1–5 need the raw EEG recordings; steps 6–8 run on the derived data in `data/`. Python dependencies
are in `requirements.txt`; the R scripts use tidyverse, cmdstanr, posterior, cowplot, ggdist and ragg.
`figures.R` uses Arial when it is installed (under WSL, from the Windows fonts) and otherwise Liberation Sans
(`fonts/`, SIL Open Font License), which has the same letter widths.

## Results in the paper

| Paper | File | Analysis |
|---|---|---|
| Interhemispheric strength (IHS) | `results/model1.csv` | primary |
| Quadrants and global efficiency | `results/model1.csv` | exploratory |
| (i) Individual alpha band | `results/model1.csv` | S1 individual band |
| Theta band | `results/model1.csv` | S6 theta |
| (ii) Children aged 2–4 and 5–9 years | `results/sensitivity_model1.csv` | S2 age 2-4, S2 age 5+ |
| (iii) Without a weak or slow posterior alpha peak | `results/sensitivity_model1.csv` | S3 without slowing |
| (iv) Data quality as covariates | `results/sensitivity_model1.csv` | S8a data quality |
| (v) Without Fp1, Fp2, F7 and F8 | `results/sensitivity_model1.csv` | S11 without frontal electrodes |
| Change with age (IHS and global efficiency) | `results/model2.csv` | key secondary, exploratory |

Figs 1–4 are drawings (`figures/visualizations.af`, Affinity); Fig 4 is exported as
`figures/all_quadrants.pdf`. The probabilities in Figs 3 and 4 are from `results/model1.csv`.

## Data

`data/` holds the derived data of the analysed children: the connectomes (8–13 Hz, individual alpha band and
theta), demographics, connectome metrics, posterior alpha peak measures and quality control.
The raw EEG recordings are not public because of their sensitivity.
