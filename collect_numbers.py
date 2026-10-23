"""
Collects every number quoted in the manuscript into results/numbers.json, so
that the text is generated from (and checked against) the analysis outputs.
The statistical results are added once the analysis scripts have written
results/; before that, only the flow and QC numbers are collected.
"""

import os
import json
import subprocess

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR, N_EPOCHS, MIN_EPOCHS, SPINDLE_RATE_FLAG, CANDIDATE_POOL, BANDS

# every key of numbers.json: <g> is test or control, <m1> a Model 1 prefix
# (MODEL1_PREFIXES, or expl_<metric> for the exploratory metrics), <m2> a
# Model 2 prefix (MODEL2_PREFIXES) and <par> a Model 2 parameter
# (MODEL2_PARAMETERS); "_lower"/"_upper" are the ends of the 95% credible
# interval and "_p" the posterior probability that the value is below 0
KEYS = {
    # flow, recordings and data quality
    "n_rec_<g>": "recordings obtained",
    "n_excl_<g>": "recordings excluded",
    "n_an_<g>": "analysable recordings",
    "dur_<g>_min": "median recording duration (min)",
    "n_photic_<g>": "recordings with photic stimulation",
    "ica_median_<g>": "median number of removed ICA components (analysable)",
    "rejected_median_<g>": "median % of candidate epochs rejected by Autoreject (analysable)",
    "n_interp_children_<g>": "analysable children with at least one interpolated electrode",
    "n_spindle_<g>": "analysable children above the spindle-rate threshold",
    "n_sleep_<g>": "analysable children with signs of sleep (S4)",
    "n_slowing_<g>": "analysable children with background slowing (S3)",
    "n_resp_unusable_<g>": "recordings whose respiration trace was unusable",
    "n_hv_<g>": "recordings with hyperventilation detected",
    "n_iaf_theta_<g>": "analysable children whose individual band is the theta band (4-7 Hz)",
    "n_epochs": "epochs analysed per recording",
    "min_epochs": "minimum number of epochs",
    "min_available_epochs": "fewest epochs free of large artefacts in an analysable recording",
    "candidate_pool": "candidate epochs for Autoreject",
    "spindle_threshold": "spindle-rate threshold (per min)",
    "control_period": "period of the control recordings",
    # software and supplement
    "r_version": "R version",
    "cmdstan_version": "CmdStan version",
    "python_version": "Python version",
    "mne_version": "MNE version",
    "repo_url": "code repository",
    "table_*, fig_*": "supplementary item numbers (SUPPLEMENT)",
    # Model 1 (results/model1.csv, sensitivity_model1.csv, check_matched_set.csv)
    "<m1>_mu": "posterior mean of mu (control SD units)",
    "<m1>_mu_lower, <m1>_mu_upper": "95% credible interval of mu",
    "<m1>_mu_p": "P(mu < 0)",
    "<m1>_n_test, <m1>_n_control": "matched CTNNB1 children and controls (not for matched_set)",
    "primary_n_strata": "sex-by-age strata of the primary analysis",
    "n_unmatched_test": "analysable CTNNB1 children without a control of the same sex and age (not in Model 1)",
    "primary_raw, primary_raw_lower, primary_raw_upper": "primary mu on the dbWPLI scale",
    "robust_label": "robust, not robust or not assessed (results/robustness.csv)",
    # Model 2 (results/model2.csv, sensitivity_s5.csv, model2_influence.csv)
    "<m2>_<par>": "posterior mean (control SD units; slopes and bga per year)",
    "<m2>_<par>_lower, <m2>_<par>_upper": "95% credible interval",
    "<m2>_<par>_p": "P(< 0)",
    "<m2>_n_test, <m2>_n_control": "children in the model",
    "influence_n": "leave-one-out refits of the key secondary (one per CTNNB1 child)",
    "influence_ids": "CTNNB1 children whose removal changes the sign of beta_ga or whether its 95% CrI excludes 0 ('none')",
    "influence_bga_min, influence_bga_max": "range of the posterior mean of beta_ga over the refits",
}

# Model 1 analyses and their key prefixes
MODEL1_PREFIXES = {
    "primary": "primary",
    "S1 individual band": "s1",
    "S2 age 0-4": "s2_young",
    "S2 age 5+": "s2_older",
    "S3 without slowing": "s3",
    "S4 without sleep": "s4",
    "S6 theta": "s6",
    "S8a data quality": "s8a",
    "S8b data quality and alpha power": "s8b",
    "S9 without Autoreject mis-calibration": "s9",
    "S10 without interpolated electrodes": "s10",
    "S11 without frontal electrodes": "s11",
    "prior mu ~ N(0, 0.5)": "prior_narrow",
    "prior mu ~ N(0, 2)": "prior_wide",
    "check: matched-set model": "matched_set",
}

# Model 2 analyses and parameters and their key prefixes
MODEL2_PREFIXES = {
    "key secondary": "m2",
    "check: matched only": "m2_matched",
    "exploratory": "m2_ge",
    "S5 all children": "s5",
}
MODEL2_PARAMETERS = {
    "group (age 5)": "group",
    "slope control (per year)": "slope_control",
    "slope test (per year)": "slope_test",
    "group x age (per year)": "bga",
    "group (spline age)": "group",
}

# supplementary items, numbered in order of first citation in the manuscript
SUPPLEMENT = {
    "table_qc_group": "1",
    "table_qc_subject": "2",
    "table_sensitivity": "3",
    "table_miljevic": "4",
    "fig_pdr": "1",
    "fig_ge": "2",
    "fig_theta": "3",
}


def fmt(x: float, digits: int = 2) -> str:
    """
    Format a number with a true minus sign.

    Args:
        x (float): The number.
        digits (int): Decimal places.
    """
    return f"{x:.{digits}f}".replace("-", "−")


def pct(p: float, n_other: int) -> str:
    """
    Posterior probability as a percentage; '>99.9%' when fewer than 10 draws
    fall on the other side.

    Args:
        p (float): Probability.
        n_other (int): Number of draws on the other side of zero.
    """
    if n_other < 10:
        return ">99.9%" if p > 0.5 else "<0.1%"
    return f"{100 * p:.1f}%"


def flow_numbers(qc: pd.DataFrame, events: pd.DataFrame) -> dict:
    """
    Sample flow, recording characteristics and data quality.

    Args:
        qc (pd.DataFrame): QC summary.
        events (pd.DataFrame): Recording events.
    """
    n = {}
    for g, key in [("test", "test"), ("control", "control")]:
        d = qc[qc["group"] == g]
        n[f"n_rec_{key}"] = int(len(d))
        n[f"n_excl_{key}"] = int(d["excluded"].sum())
        n[f"n_an_{key}"] = int((~d["excluded"]).sum())
        n[f"dur_{key}_min"] = f"{np.median(d['duration_s']) / 60:.0f}"
        photic_ids = events.loc[events["event"] == "photic", "id"].unique()
        n[f"n_photic_{key}"] = int(d["id"].isin(photic_ids).sum())

        a = d[~d["excluded"]]
        n[f"ica_median_{key}"] = f"{a['n_ica_excluded'].median():.0f}"
        n[f"rejected_median_{key}"] = f"{100 * (a['ar_rejected_epochs'] / a['pool_epochs']).median():.0f}"
        n[f"n_interp_children_{key}"] = int((a["n_bad_channels"] > 0).sum())
        n[f"n_spindle_{key}"] = int(a["spindle_flag"].sum())

    # analysable CTNNB1 children without a control of the same sex and age
    # (utils/matching.R), who are left out of Model 1
    a = qc[~qc["excluded"]]
    control_cells = set(zip(a.loc[a["group"] == "control", "sex"], a.loc[a["group"] == "control", "age"]))
    tests = a[a["group"] == "test"]
    n["n_unmatched_test"] = int(sum((s, g) not in control_cells for s, g in zip(tests["sex"], tests["age"])))

    n["n_epochs"] = N_EPOCHS
    n["min_epochs"] = MIN_EPOCHS
    n["candidate_pool"] = CANDIDATE_POOL
    n["spindle_threshold"] = SPINDLE_RATE_FLAG

    return n


def flag_numbers(pdr: pd.DataFrame, respiration: pd.DataFrame) -> dict:
    """
    Children flagged for background slowing or signs of sleep (analysable),
    children whose individual band is the theta band, and recordings with an
    unusable respiration trace or hyperventilation.

    Args:
        pdr (pd.DataFrame): PDR table of the analysable children.
        respiration (pd.DataFrame): Respiration summary of all recordings.
    """
    n = {}
    theta_low, theta_high = BANDS["theta"]
    for g in ["test", "control"]:
        p = pdr[pdr["group"] == g]
        r = respiration[respiration["group"] == g]
        n[f"n_slowing_{g}"] = int(p["slowing_flag"].sum())
        n[f"n_sleep_{g}"] = int(p["sleep_flag"].sum())
        n[f"n_resp_unusable_{g}"] = int((~r["resp_usable"].astype(bool)).sum())
        n[f"n_hv_{g}"] = int((r["n_hv"] > 0).sum())
        n[f"n_iaf_theta_{g}"] = int(((p["iaf_low"] == theta_low) & (p["iaf_high"] == theta_high)).sum())

    return n


def posterior(stem: str, row: pd.Series, mean: str, lower: str, upper: str) -> dict:
    """
    Posterior mean, 95% credible interval and P(< 0) of one result row.

    Args:
        stem (str): Key of the posterior mean; the other keys extend it.
        row (pd.Series): Result row with p_below, n_above and n_below.
        mean (str): Column of the posterior mean.
        lower (str): Column of the lower end of the interval.
        upper (str): Column of the upper end of the interval.
    """
    return {
        stem: fmt(row[mean]),
        f"{stem}_lower": fmt(row[lower]),
        f"{stem}_upper": fmt(row[upper]),
        f"{stem}_p": pct(row["p_below"], int(min(row["n_above"], row["n_below"]))),
    }


def result_numbers(results_dir: str) -> dict:
    """
    Statistical results: Model 1 (primary, exploratory and sensitivity
    analyses, matched-set check), the robustness label and Model 2 (key
    secondary, checks, S5 and the influence check).

    Args:
        results_dir (str): Directory with the analysis outputs.
    """
    n = {}

    # Model 1
    model1 = pd.concat(
        [
            pd.read_csv(os.path.join(results_dir, "model1.csv")),
            pd.read_csv(os.path.join(results_dir, "sensitivity_model1.csv")),
        ],
        ignore_index=True,
    )
    for _, row in model1.iterrows():
        if row["analysis"] == "exploratory":
            prefix = f"expl_{row['metric']}"
        else:
            prefix = MODEL1_PREFIXES[row["analysis"]]
        n.update(posterior(f"{prefix}_mu", row, "mu", "mu_lower", "mu_upper"))
        n[f"{prefix}_n_test"] = int(row["n_test"])
        n[f"{prefix}_n_control"] = int(row["n_control"])

    primary = model1[model1["analysis"] == "primary"].iloc[0]
    n["primary_n_strata"] = int(primary["n_strata"])
    n["primary_raw"] = fmt(primary["raw"], 3)
    n["primary_raw_lower"] = fmt(primary["raw_lower"], 3)
    n["primary_raw_upper"] = fmt(primary["raw_upper"], 3)

    matched_set = pd.read_csv(os.path.join(results_dir, "check_matched_set.csv")).iloc[0]
    n.update(posterior(f"{MODEL1_PREFIXES[matched_set['analysis']]}_mu", matched_set, "mean", "lower", "upper"))

    robustness = pd.read_csv(os.path.join(results_dir, "robustness.csv"))
    n["robust_label"] = robustness["label"].iloc[0]

    # Model 2
    model2 = pd.concat(
        [
            pd.read_csv(os.path.join(results_dir, "model2.csv")),
            pd.read_csv(os.path.join(results_dir, "sensitivity_s5.csv")),
        ],
        ignore_index=True,
    )
    for _, row in model2.iterrows():
        prefix = MODEL2_PREFIXES[row["analysis"]]
        n.update(posterior(f"{prefix}_{MODEL2_PARAMETERS[row['parameter']]}", row, "mean", "lower", "upper"))
        n[f"{prefix}_n_test"] = int(row["n_test"])
        n[f"{prefix}_n_control"] = int(row["n_control"])

    influence = pd.read_csv(os.path.join(results_dir, "model2_influence.csv"))
    influential = influence.loc[influence["influential"].astype(bool), "left_out"].tolist()
    n["influence_n"] = int(len(influence))
    n["influence_ids"] = ", ".join(influential) if influential else "none"
    n["influence_bga_min"] = fmt(influence["mean"].min())
    n["influence_bga_max"] = fmt(influence["mean"].max())

    return n


def versions() -> dict:
    """
    Software versions for the Methods.
    """
    out = {}
    try:
        r = subprocess.run(
            ["Rscript", "-e", "cat(R.version$major, R.version$minor, sep='.'); cat(' '); cat(as.character(cmdstanr::cmdstan_version()))"],
            capture_output=True, text=True, check=True,
        )
        r_version, cmdstan_version = r.stdout.split()
        out["r_version"] = r_version
        out["cmdstan_version"] = cmdstan_version
    except Exception:
        out["r_version"] = "unknown"
        out["cmdstan_version"] = "unknown"

    import sys
    import mne

    out["python_version"] = sys.version.split()[0]
    out["mne_version"] = mne.__version__.split("+")[0]
    return out


if __name__ == "__main__":
    qc = pd.read_csv(os.path.join(DATA_DIR, "qc_summary.csv"))
    events = pd.read_csv(os.path.join(DATA_DIR, "recording_events.csv"))
    pdr = pd.read_csv(os.path.join(DATA_DIR, "pdr.csv"))
    respiration = pd.read_csv(os.path.join(DATA_DIR, "respiration.csv"))

    numbers = {}
    numbers.update(flow_numbers(qc, events))
    numbers.update(flag_numbers(pdr, respiration))
    numbers.update(versions())
    numbers.update(SUPPLEMENT)
    numbers["min_available_epochs"] = int(qc.loc[~qc["excluded"], "good_epochs"].min())
    numbers["repo_url"] = "https://github.com/demsarjure/ped_neuro_eeg"
    # recording dates are not stored; range printed by recording_events.py
    numbers["control_period"] = "October 2023 and February 2025"

    # statistical results, once the analysis scripts have run
    if os.path.exists(os.path.join(RESULTS_DIR, "model1.csv")):
        numbers.update(result_numbers(RESULTS_DIR))
    else:
        print("---> no statistical results yet: flow and QC numbers only")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "numbers.json"), "w") as f:
        json.dump(numbers, f, indent=2, ensure_ascii=False)

    print(json.dumps(numbers, indent=2, ensure_ascii=False))
