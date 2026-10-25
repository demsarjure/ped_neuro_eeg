"""
Collects every number quoted in the manuscript into results/numbers.json, so
that the text is generated from (and checked against) the analysis outputs.
The statistical results are added once the analysis scripts have written
results/; before that, only the flow and QC numbers are collected.
With the statistical results, it also writes results/fig2_numbers.csv, the
numbers shown in Fig. 2.
"""

import os
import json
import subprocess

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR, N_EPOCHS, MIN_EPOCHS, SPINDLE_RATE_FLAG, CANDIDATE_POOL, BANDS
from config import CONNECTOME_DIR, SCALP_CHANNELS, LEFT_CHANNELS, RIGHT_CHANNELS, IAF_HALF_WIDTH

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
    # Model 2 (results/model2.csv, model2_influence.csv)
    "<m2>_<par>": "posterior mean (control SD units; slopes and bga per year)",
    "<m2>_<par>_lower, <m2>_<par>_upper": "95% credible interval",
    "<m2>_<par>_p": "P(< 0)",
    "<m2>_n_test, <m2>_n_control": "children in the model",
    "influence_n": "leave-one-out refits of the key secondary (one per CTNNB1 child)",
    "influence_ids": "CTNNB1 children whose removal changes the sign of beta_ga or whether its 95% CrI excludes 0 ('none')",
    "influence_bga_min, influence_bga_max": "range of the posterior mean of beta_ga over the refits",
    # Results section and Table 1 (analysable children unless stated)
    "age_median_<g>, age_low_<g>, age_high_<g>": "age in years: median and range",
    "n_girls_<g>, n_boys_<g>": "girls and boys",
    "dur_an_median_<g>, dur_an_low_<g>, dur_an_high_<g>": "recording length (min): median and range",
    "blink_median_<g>": "median blink rate (per min)",
    "muscle_median_<g>": "median muscle activity (20-40 Hz power, dB)",
    "alpha_power_median_<g>": "median aperiodic-adjusted posterior alpha power (dB)",
    "n_pdr_clear_<g>, n_pdr_absent_<g>": "children with and without a clear PDR",
    "iaf_centre_median_<g>": "median centre of the individual band (Hz)",
    "primary_age_median_<g>, primary_age_low_<g>, primary_age_high_<g>": "age in the main comparison (Model 1 sample)",
    "primary_n_girls_<g>, primary_n_boys_<g>": "girls and boys in the main comparison",
    "<m1>_raw, <m1>_raw_lower, <m1>_raw_upper": "mu on the original scale (dbWPLI; global efficiency in its own units)",
    "<m1>_n_lower": "CTNNB1 children below the mean of their matched controls (primary, exploratory, S1, S6)",
    "quadrant_mu_min, quadrant_mu_max, quadrant_p_min": "range of mu and the lowest P(mu < 0) over the four quadrant pairs",
    "robust_p_min": "lowest posterior probability of the primary's sign among the robustness checks",
    "influence_n_sign_changed, influence_n_cri_changed": "refits that changed the sign of beta_ga, or whether its 95% CrI excludes 0",
    "influence_p_min, influence_p_max": "range of P(beta_ga < 0) over the refits",
    "matrix_n_pairs, matrix_n_lower": "interhemispheric electrode pairs, and those with a lower mean in CTNNB1 (Fig. 3C)",
    "ppc_p_min, ppc_p_max": "range of P(replicated > observed) over the mean, SD, minimum and maximum of d in the posterior predictive checks of the primary model (results/model1_ppc.csv)",
    "<key>_ptext": "every P as text: 'P = 99.7%', 'P > 99.9%' or 'P < 0.1%'",
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
}
MODEL2_PARAMETERS = {
    "group (age 5)": "group",
    "slope control (per year)": "slope_control",
    "slope test (per year)": "slope_test",
    "group x age (per year)": "bga",
    "group (spline age)": "group",
}

# supplementary items, numbered in order of first citation in the manuscript
# (main text first, then the supplement; manuscript/v3/render.py checks it)
SUPPLEMENT = {
    "table_qc_group": "1",
    "table_deviations": "2",
    "table_miljevic": "3",
    "table_strata": "4",
    "table_sensitivity": "5",
    "table_model2": "6",
    "fig_flow": "1",
    "fig_pdr": "2",
    "fig_alpha": "3",
    "fig_freq": "4",
    "table_posthoc": "7",
}

# post hoc analyses (results/posthoc.csv, not part of the analysis plan) and their
# key prefixes
POSTHOC_PREFIXES = {
    "post hoc: within the hemispheres": "ph_intra",
    "post hoc: between minus within the hemispheres": "ph_contrast",
    "post hoc: overlapping alpha power, adjusted": "ph_overlap",
    "post hoc: blinks near the analysed epochs": "ph_blinks",
    "post hoc: control alpha power degraded with noise": "ph_noise",
    "post hoc: blinks check (epochs_near_blink >= 0.5)": "ph_blinks50",
    "post hoc: blinks check (epochs_near_blink >= 0.75)": "ph_blinks75",
    "post hoc: blinks check (epochs_near_blink_5s >= 0.5)": "ph_blinks5s",
    "post hoc: posterior alpha power": "ph_alpha",
    "post hoc: without the two highest control children": "ph_top2",
    "post hoc: without spindles": "ph_nospindle",
    "post hoc: epochs matched for alpha power": "ph_epochs",
}

# Model 1 outcomes of Fig. 2 (interhemispheric strength and the quadrant pairs)
FIG2_METRICS = {
    "lh_rh": "Interhemispheric strength",
    "laq_raq": "Left and right anterior quadrants",
    "lpq_rpq": "Left and right posterior quadrants",
    "laq_rpq": "Left anterior and right posterior quadrants",
    "raq_lpq": "Right anterior and left posterior quadrants",
}
QUADRANT_METRICS = ["laq_raq", "lpq_rpq", "laq_rpq", "raq_lpq"]


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
    secondary, checks and the influence check).

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
    model2 = pd.read_csv(os.path.join(results_dir, "model2.csv"))
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


def participant_numbers(qc: pd.DataFrame, pdr: pd.DataFrame) -> dict:
    """
    Characteristics of the analysable children by group (Table 1): age, sex,
    recording length, dominant posterior rhythm and individual band, and the
    data-quality measures quoted in the Results.

    Args:
        qc (pd.DataFrame): QC summary.
        pdr (pd.DataFrame): PDR table of the analysable children.
    """
    n = {}
    for g in ["test", "control"]:
        a = qc[~qc["excluded"] & (qc["group"] == g)]
        p = pdr[pdr["group"] == g]
        n[f"age_median_{g}"] = f"{a['age'].median():g}"
        n[f"age_low_{g}"] = int(a["age"].min())
        n[f"age_high_{g}"] = int(a["age"].max())
        n[f"n_girls_{g}"] = int((a["sex"] == "F").sum())
        n[f"n_boys_{g}"] = int((a["sex"] == "M").sum())
        minutes = a["duration_s"] / 60
        n[f"dur_an_median_{g}"] = f"{minutes.median():.0f}"
        n[f"dur_an_low_{g}"] = f"{minutes.min():.0f}"
        n[f"dur_an_high_{g}"] = f"{minutes.max():.0f}"
        n[f"blink_median_{g}"] = f"{a['blink_rate'].median():.1f}"
        n[f"muscle_median_{g}"] = fmt(a["hf_power_db"].median(), 1)
        n[f"alpha_power_median_{g}"] = fmt(p["alpha_adj_db"].median(), 1)
        n[f"n_pdr_clear_{g}"] = int(p["pdr_clear"].sum())
        n[f"n_pdr_absent_{g}"] = int((~p["pdr_clear"].astype(bool)).sum())
        # the individual band is centred on the rounded PDR (or the PDR expected
        # for age); its upper edge is never clipped
        n[f"iaf_centre_median_{g}"] = f"{(p['iaf_high'] - IAF_HALF_WIDTH).median():g}"

    return n


def extra_result_numbers(results_dir: str, qc: pd.DataFrame) -> dict:
    """
    Further results for the Results section: Model 1 effects on the original
    scale, the CTNNB1 children below their matched controls, the range of the
    quadrant-pair effects, the weakest robustness check, the influence-check
    counts, the main-comparison sample and the electrode pairs of Fig. 3C.

    Args:
        results_dir (str): Directory with the analysis outputs.
        qc (pd.DataFrame): QC summary.
    """
    n = {}

    model1 = pd.concat(
        [
            pd.read_csv(os.path.join(results_dir, "model1.csv")),
            pd.read_csv(os.path.join(results_dir, "sensitivity_model1.csv")),
        ],
        ignore_index=True,
    )
    model1["prefix"] = [
        f"expl_{m}" if a == "exploratory" else MODEL1_PREFIXES[a] for a, m in zip(model1["analysis"], model1["metric"])
    ]
    primary = model1[model1["analysis"] == "primary"].iloc[0]

    # mu on the original scale (the primary is in result_numbers)
    for _, row in model1[model1["prefix"] != "primary"].iterrows():
        n[f"{row['prefix']}_raw"] = fmt(row["raw"], 3)
        n[f"{row['prefix']}_raw_lower"] = fmt(row["raw_lower"], 3)
        n[f"{row['prefix']}_raw_upper"] = fmt(row["raw_upper"], 3)

    # CTNNB1 children whose value is below the mean of their matched controls
    differences = pd.read_csv(os.path.join(results_dir, "model1_differences.csv"))
    for (analysis, metric), d in differences.groupby(["analysis", "metric"]):
        prefix = f"expl_{metric}" if analysis == "exploratory" else MODEL1_PREFIXES[analysis]
        n[f"{prefix}_n_lower"] = int((d["raw"] < 0).sum())

    # range of the four quadrant-pair effects
    quadrants = model1[(model1["analysis"] == "exploratory") & model1["metric"].isin(QUADRANT_METRICS)]
    weakest = quadrants.loc[quadrants["p_below"].idxmin()]
    n["quadrant_mu_min"] = fmt(quadrants["mu"].min())
    n["quadrant_mu_max"] = fmt(quadrants["mu"].max())
    n["quadrant_p_min"] = pct(weakest["p_below"], int(min(weakest["n_above"], weakest["n_below"])))

    # lowest posterior probability of the primary's sign among the robustness checks
    robustness = pd.read_csv(os.path.join(results_dir, "robustness.csv"))
    check = robustness.loc[robustness["p_same_sign"].idxmin()]
    row = model1[model1["analysis"] == check["analysis"]].iloc[0]
    n_other = row["n_above"] if primary["mu"] < 0 else row["n_below"]
    n["robust_p_min"] = pct(check["p_same_sign"], int(n_other))

    # leave-one-out refits of the key secondary
    influence = pd.read_csv(os.path.join(results_dir, "model2_influence.csv"))
    n["influence_n_sign_changed"] = int(influence["sign_changed"].astype(bool).sum())
    n["influence_n_cri_changed"] = int(influence["cri_changed"].astype(bool).sum())
    lowest = influence.loc[influence["p_below"].idxmin()]
    highest = influence.loc[influence["p_below"].idxmax()]
    n["influence_p_min"] = pct(lowest["p_below"], int(min(lowest["n_above"], lowest["n_below"])))
    n["influence_p_max"] = pct(highest["p_below"], int(min(highest["n_above"], highest["n_below"])))

    # posterior predictive checks of the primary model: the range, over the
    # mean, SD, minimum and maximum of d, of the probability that replicated
    # data exceed the observed value
    ppc = pd.read_csv(os.path.join(results_dir, "model1_ppc.csv"))
    n["ppc_p_min"] = f"{100 * ppc['p_rep_greater'].min():.1f}%"
    n["ppc_p_max"] = f"{100 * ppc['p_rep_greater'].max():.1f}%"

    # the main-comparison sample: the matched CTNNB1 children and all analysable
    # controls of their sex-by-age strata
    tests = differences[differences["analysis"] == "primary"]
    analysable = qc[~qc["excluded"]]
    controls = analysable[analysable["group"] == "control"]
    strata = set(zip(tests["sex"], tests["age"]))
    matched = controls[[(s, a) in strata for s, a in zip(controls["sex"], controls["age"])]]
    assert len(tests) == primary["n_test"] and len(matched) == primary["n_control"]
    for g, d in [("test", tests), ("control", matched)]:
        n[f"primary_age_median_{g}"] = f"{d['age'].median():g}"
        n[f"primary_age_low_{g}"] = int(d["age"].min())
        n[f"primary_age_high_{g}"] = int(d["age"].max())
        n[f"primary_n_girls_{g}"] = int((d["sex"] == "F").sum())
        n[f"primary_n_boys_{g}"] = int((d["sex"] == "M").sum())

    # Fig. 3C: the mean interhemispheric matrix of the matched CTNNB1 children
    # minus the mean over these children of their matched controls' mean matrix
    left = [SCALP_CHANNELS.index(c) for c in LEFT_CHANNELS]
    right = [SCALP_CHANNELS.index(c) for c in RIGHT_CHANNELS]

    def block(sub_id: str, group: str) -> np.ndarray:
        path = os.path.join(CONNECTOME_DIR, group, "alpha", f"{sub_id}_alpha_task-rest_connectome_eeg.csv")
        return np.loadtxt(path, delimiter=",")[np.ix_(left, right)]

    def stratum_mean(sex: str, age: int) -> np.ndarray:
        ids = matched.loc[(matched["sex"] == sex) & (matched["age"] == age), "id"]
        return np.mean([block(c, "control") for c in ids], axis=0)

    test_matrix = np.mean([block(i, "test") for i in tests["id"]], axis=0)
    control_matrix = np.mean([stratum_mean(s, a) for s, a in zip(tests["sex"], tests["age"])], axis=0)
    n["matrix_n_pairs"] = int(test_matrix.size)
    n["matrix_n_lower"] = int((test_matrix < control_matrix).sum())

    return n


def fig2_numbers(results_dir: str) -> pd.DataFrame:
    """
    Numbers of Fig. 2, for interhemispheric strength and the quadrant pairs in
    the Model 1 sample: the mean of the matched CTNNB1 children, the mean over
    these children of their matched controls' means (as in Fig. 3), mu on the
    dbWPLI scale and in control SD units with its 95% credible interval, and
    P(mu < 0).

    Args:
        results_dir (str): Directory with the analysis outputs.
    """
    metrics = pd.read_csv(os.path.join(DATA_DIR, "connectome_metrics_alpha.csv")).set_index("id")
    model1 = pd.read_csv(os.path.join(results_dir, "model1.csv"))
    differences = pd.read_csv(os.path.join(results_dir, "model1_differences.csv"))

    rows = []
    for metric, label in FIG2_METRICS.items():
        r = model1[(model1["band"] == "alpha") & (model1["metric"] == metric)].iloc[0]
        d = differences[(differences["band"] == "alpha") & (differences["metric"] == metric)]
        values = metrics.loc[d["id"], metric].to_numpy()
        rows.append(
            {
                "metric": metric,
                "outcome": label,
                "n_ctnnb1": int(r["n_test"]),
                "n_controls": int(r["n_control"]),
                "mean_ctnnb1": fmt(values.mean(), 3),
                # each child's value minus its difference is its matched controls' mean
                "mean_matched_controls": fmt((values - d["raw"].to_numpy()).mean(), 3),
                "difference": fmt(r["raw"], 3),
                "difference_lower": fmt(r["raw_lower"], 3),
                "difference_upper": fmt(r["raw_upper"], 3),
                "mu": fmt(r["mu"]),
                "mu_lower": fmt(r["mu_lower"]),
                "mu_upper": fmt(r["mu_upper"]),
                "P": pct(r["p_below"], int(min(r["n_above"], r["n_below"]))),
            }
        )

    return pd.DataFrame(rows)


def posthoc_numbers(results_dir: str, pdr: pd.DataFrame) -> dict:
    """
    Post hoc analyses, not part of the analysis plan (analysis_posthoc.R):
    connectivity within the hemispheres, the contrast between and within the
    hemispheres, and the comparison at overlapping alpha power; also the alpha
    power of the children without an atypical posterior rhythm.

    Args:
        results_dir (str): Directory with the analysis outputs.
        pdr (pd.DataFrame): PDR table of the analysable children.
    """
    n = {}
    path = os.path.join(results_dir, "posthoc.csv")
    if not os.path.exists(path):
        return n

    posthoc = pd.read_csv(path)
    for _, row in posthoc.iterrows():
        prefix = POSTHOC_PREFIXES[row["analysis"]]
        n.update(posterior(f"{prefix}_mu", row, "mu", "mu_lower", "mu_upper"))
        # dbWPLI to 3 decimals; alpha power in dB to 1
        digits = 1 if prefix == "ph_alpha" else 3
        n[f"{prefix}_raw"] = fmt(row["raw"], digits)
        n[f"{prefix}_raw_lower"] = fmt(row["raw_lower"], digits)
        n[f"{prefix}_raw_upper"] = fmt(row["raw_upper"], digits)
        n[f"{prefix}_n_test"] = int(row["n_test"])
        n[f"{prefix}_n_control"] = int(row["n_control"])

    overlap = posthoc[posthoc["analysis"] == "post hoc: overlapping alpha power, adjusted"].iloc[0]
    n["alpha_overlap_low"] = fmt(overlap["alpha_low"], 1)
    n["alpha_overlap_high"] = fmt(overlap["alpha_high"], 1)
    for g in ["test", "control"]:
        p = pdr[(pdr["group"] == g) & ~pdr["slowing_flag"].astype(bool)]
        n[f"alpha_power_median_typical_{g}"] = fmt(p["alpha_adj_db"].median(), 1)

    # blinks around the analysed epochs (blinks_posthoc.py)
    blinks = pd.read_csv(os.path.join(DATA_DIR, "blinks_analysed.csv"))
    for g in ["test", "control"]:
        b = blinks[blinks["group"] == g]
        n[f"blinks_analysed_median_{g}"] = fmt(b["blinks_per_min"].median(), 1)
        n[f"epochs_near_blink_median_{g}"] = f"{100 * b['epochs_near_blink'].median():.0f}%"

    # noise added to the matched control children (noise_posthoc.py)
    noise = pd.read_csv(os.path.join(DATA_DIR, "noise_injection.csv"))
    n["noise_n_degraded"] = int((noise["noise_scale"] > 0).sum())
    n["noise_alpha_after_mean"] = fmt(noise["alpha_after_db"].mean(), 1)
    n["noise_alpha_target_mean"] = fmt(noise["alpha_target_db"].mean(), 1)

    # the open-eyes check with other thresholds or a stricter window
    checks = posthoc[posthoc["analysis"].str.startswith("post hoc: blinks")]
    n["ph_blinks_checks_min"] = fmt(checks["mu"].max())
    n["ph_blinks_checks_max"] = fmt(checks["mu"].min())

    # influence of single control children and strata; placement within the matched set
    influence = pd.read_csv(os.path.join(results_dir, "posthoc_influence.csv"))
    for name, key in [("leave one control out", "loo_control"), ("leave one stratum out", "loo_stratum")]:
        r = influence[influence["analysis"] == name]
        n[f"{key}_n"] = int(len(r))
        n[f"{key}_mu_min"] = fmt(r["mu"].max())
        n[f"{key}_mu_max"] = fmt(r["mu"].min())
        n[f"{key}_p_min"] = pct(r["p_below"].min(), 10)
    placement = pd.read_csv(os.path.join(results_dir, "posthoc_placement.csv"))
    n["placement_median_pct"] = f"{100 * placement['percentile'].median():.0f}"
    n["placement_n_below_median"] = int((placement["percentile"] < 0.5).sum())

    # the matched-set check on the dbWPLI scale (the same SD as the main analysis)
    model1 = pd.read_csv(os.path.join(results_dir, "model1.csv"))
    primary = model1[model1["analysis"] == "primary"].iloc[0]
    matched_set = pd.read_csv(os.path.join(results_dir, "check_matched_set.csv")).iloc[0]
    for col, key in [("mean", ""), ("lower", "_lower"), ("upper", "_upper")]:
        n[f"matched_set_raw{key}"] = fmt(matched_set[col] * primary["pooled_sd"], 3)

    # S9 and the main analysis to three decimals (S9 agrees to two)
    sensitivity = pd.read_csv(os.path.join(results_dir, "sensitivity_model1.csv"))
    n["primary_mu_3dp"] = fmt(primary["mu"], 3)
    n["s9_mu_3dp"] = fmt(sensitivity.loc[sensitivity["analysis"].str.startswith("S9"), "mu"].iloc[0], 3)

    # aperiodic-adjusted posterior theta power (theta-alpha difference plus alpha power)
    for g in ["test", "control"]:
        p = pdr[pdr["group"] == g]
        n[f"theta_power_median_{g}"] = fmt((p["theta_alpha_db"] + p["alpha_adj_db"]).median(), 1)

    # epochs selected to narrow the difference in alpha power (epochs_posthoc.py), alpha
    # power in the open-eyes subset, the quality of the aperiodic fits, and the
    # unweighted summaries of the per-child differences
    epochs = pd.read_csv(os.path.join(DATA_DIR, "epochs_alpha_matched.csv"))
    open_eyes = blinks.loc[blinks["epochs_near_blink"] >= 0.9, "id"]
    for g in ["test", "control"]:
        e = epochs[epochs["group"] == g]
        n[f"epochs_alpha_all_median_{g}"] = fmt(e["alpha_all_db"].median(), 1)
        n[f"epochs_alpha_matched_median_{g}"] = fmt(e["alpha_matched_db"].median(), 1)
        p = pdr[pdr["group"] == g]
        n[f"alpha_open_eyes_median_{g}"] = fmt(p.loc[p["id"].isin(open_eyes), "alpha_adj_db"].median(), 1)
        n[f"fit_r2_median_{g}"] = fmt(p["fit_r2"].median(), 3)
    differences = pd.read_csv(os.path.join(results_dir, "model1_differences.csv"))
    d = differences[(differences["analysis"] == "primary") & (differences["metric"] == "lh_rh")]["d"]
    n["primary_d_mean"] = fmt(d.mean())

    # split-half reliability (Spearman-Brown corrected Spearman correlation of the odd and
    # even epochs) and the chance level of IHS (reliability_posthoc.py)
    reliability = pd.read_csv(os.path.join(DATA_DIR, "ihs_reliability.csv"))
    for g in ["test", "control"]:
        r = reliability[reliability["group"] == g]
        for col, key in [("ihs", "rel_ihs"), ("alpha", "rel_alpha")]:
            rho = r[f"{col}_odd"].corr(r[f"{col}_even"], method="spearman")
            n[f"{key}_{g}"] = fmt(2 * rho / (1 + rho))
        n[f"chance_n_above_{g}"] = int(r["above_chance"].sum())
    n["primary_d_median"] = fmt(d.median())

    # the key secondary age model in the individual band (all children; clear rhythm only)
    age = pd.read_csv(os.path.join(results_dir, "posthoc_age.csv"))
    bga = age[age["parameter"] == "group x age (per year)"]
    n.update(posterior("ph_age_iaf_bga", bga.iloc[0], "mean", "lower", "upper"))
    n.update(posterior("ph_age_iaf_clear_bga", bga.iloc[1], "mean", "lower", "upper"))
    n["ph_age_iaf_clear_n_test"] = int(bga.iloc[1]["n_test"])
    n["ph_age_iaf_clear_n_control"] = int(bga.iloc[1]["n_control"])

    # the key secondary without each control child in turn
    age_loo = pd.read_csv(os.path.join(results_dir, "posthoc_age_influence.csv"))
    n["age_loo_n"] = int(len(age_loo))
    n["age_loo_bga_min"] = fmt(age_loo["mean"].max())
    n["age_loo_bga_max"] = fmt(age_loo["mean"].min())
    n["age_loo_p_min"] = pct(age_loo["p_below"].min(), 10)
    n["age_loo_p_max"] = pct(age_loo["p_below"].max(), 10)

    # leave-one-out ranges over control children and strata together
    n["loo_mu_min"] = fmt(influence["mu"].max())
    n["loo_mu_max"] = fmt(influence["mu"].min())
    n["loo_raw_min"] = fmt(influence["raw"].max(), 3)
    n["loo_raw_max"] = fmt(influence["raw"].min(), 3)

    # leave-one-out ranges on the dbWPLI scale
    for name, key in [("leave one control out", "loo_control"), ("leave one stratum out", "loo_stratum")]:
        r = influence[influence["analysis"] == name]
        n[f"{key}_raw_min"] = fmt(r["raw"].max(), 3)
        n[f"{key}_raw_max"] = fmt(r["raw"].min(), 3)

    # probabilistic index (Bayesian bootstrap)
    index = pd.read_csv(os.path.join(results_dir, "posthoc_probability_index.csv")).iloc[0]
    n["prob_index"] = f"{100 * index['estimate']:.0f}%"
    n["prob_index_lower"] = f"{100 * index['lower']:.0f}%"
    n["prob_index_upper"] = f"{100 * index['upper']:.0f}%"
    n["prob_index_n_pairs"] = int(index["n_pairs"])

    # the alpha-power measure: a shorter aperiodic fit, absolute power, the aperiodic
    # exponent and what ICA retained (spectra_posthoc.py, pdr.py, qc.py)
    spectral = pd.read_csv(os.path.join(DATA_DIR, "spectral_checks.csv"))
    qc = pd.read_csv(os.path.join(DATA_DIR, "qc_summary.csv"))
    qc = qc[~qc["excluded"]]
    for g in ["test", "control"]:
        s = spectral[spectral["group"] == g]
        n[f"alpha_20hz_median_{g}"] = fmt(s["alpha_adj_db_20hz"].median(), 1)
        n[f"alpha_abs_median_{g}"] = fmt(s["alpha_abs_db"].median(), 1)
        n[f"aperiodic_exponent_median_{g}"] = fmt(pdr.loc[pdr["group"] == g, "aperiodic_exponent"].median(), 2)
        q = qc[qc["group"] == g]
        n[f"ica_alpha_retained_median_{g}"] = f"{100 * q['ica_posterior_alpha_retained'].median():.1f}%"
        n[f"ica_frontal_retained_median_{g}"] = f"{100 * q['ica_frontal_rms_retained'].median():.0f}%"

    return n


def p_texts(numbers: dict) -> dict:
    """
    Every posterior probability as text for the manuscript: 'P = 99.7%', or
    'P > 99.9%' and 'P < 0.1%' at the limits.

    Args:
        numbers (dict): Collected numbers; probabilities are the keys ending in _p.
    """
    out = {}
    for key, value in numbers.items():
        if key.endswith("_p") and isinstance(value, str) and value.endswith("%"):
            if value[0] in "<>":
                out[f"{key}text"] = f"P {value[0]} {value[1:]}"
            else:
                out[f"{key}text"] = f"P = {value}"

    return out


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
    numbers.update(participant_numbers(qc, pdr))
    numbers.update(versions())
    numbers.update(SUPPLEMENT)
    numbers["min_available_epochs"] = int(qc.loc[~qc["excluded"], "good_epochs"].min())
    numbers["repo_url"] = "https://github.com/demsarjure/ped_neuro_eeg"
    # recording dates are not stored; range printed by recording_events.py
    numbers["control_period"] = "October 2023 and February 2025"

    # statistical results, once the analysis scripts have run
    if os.path.exists(os.path.join(RESULTS_DIR, "model1.csv")):
        numbers.update(result_numbers(RESULTS_DIR))
        numbers.update(extra_result_numbers(RESULTS_DIR, qc))
        numbers.update(posthoc_numbers(RESULTS_DIR, pdr))
        fig2_numbers(RESULTS_DIR).to_csv(os.path.join(RESULTS_DIR, "fig2_numbers.csv"), index=False)
    else:
        print("---> no statistical results yet: flow and QC numbers only")
    numbers.update(p_texts(numbers))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "numbers.json"), "w") as f:
        json.dump(numbers, f, indent=2, ensure_ascii=False)

    print(json.dumps(numbers, indent=2, ensure_ascii=False))
