# Analysis specification

Status: **FROZEN** on 23 October 2026, after the QC review and before any group comparison with the revised pipeline. The freeze is the git commit that adds this line; the SHA-256 hashes of the code, this plan and the derived data at that point are in freeze_manifest.sha256 (and those of the cleaned epochs, which are not public, in freeze_manifest_epochs.sha256).

Amended once, on 24 October 2026, after the group comparison: the sample was restricted to children aged 2–9 years (see the last section).

The plan was written during the re-analysis, after the results of the original analysis were known, and frozen before any group comparison with the revised pipeline.

Every rule below is applied in the same way to every recording; nothing is set by hand for a single recording.

## Data

- **Recordings:** resting EEG during wakefulness, 19 scalp channels, 250 Hz. Children with CTNNB1 syndrome were recorded during the study visit, controls during routine clinical EEGs. Eye opening and closing are not annotated in the exports, so eye state is not known.
- **Excluded periods:** recording breaks (±3.5 s), the first and last 30 s, photic stimulation (±60 s; from the photic channel) and hyperventilation together with the 120 s after it.
- **Hyperventilation:** detected from the respiration trace (export column C032, 'DIH, respiratory'):
  - the trace is assessed only if it shows a breathing rhythm: the maximum of its spectrum (median over 30-s windows) within 0.1–1.5 Hz lies at 0.15–1 Hz; otherwise hyperventilation cannot be detected in that recording (stated as a limitation);
  - breathing amplitude = RMS of the 0.1–1.5 Hz band-passed trace in 10-s windows (1-s steps);
  - hyperventilation = a run of at least 60 s in which the amplitude is at least twice the recording's median.
- **Candidate epochs:** the 400 1-s epochs nearest the recording midpoint outside the excluded periods.
- **Bad electrodes** (judged in each candidate epoch, in the recording reference, after a 1-Hz high-pass):
  - flat: 1–40 Hz RMS below 1 µV;
  - poor contact: 48–52 Hz mains amplitude and 1–40 Hz RMS both at least 5× those of that epoch's lower-quartile electrode.

  An electrode is bad if it is flat or has poor contact in more than 20% of the candidate epochs. Bad electrodes are interpolated (spherical splines).
- **Epochs:** after the 1–40 Hz band-pass, the common average reference and ICA, Autoreject (local; ρ ∈ {1, 2, 3} and consensus κ ∈ {0, 0.1, …, 1}, both cross-validated per recording; seed 42) is fitted on the candidate epochs. An epoch is rejected if at least the proportion κ of electrodes exceed their thresholds; otherwise up to ρ of them (largest peak-to-peak first) are interpolated and any others are left unchanged. The data are then re-referenced to the common average. The 100 epochs nearest the midpoint among those not rejected are analysed (all of them if fewer). They are described as free of large artefacts, not as artefact-free.
- **Excluded at Checkpoint 1:** T029 and T030. In both, the same 10 electrodes (F3, F4, F7, F8, Fz, P3, P4, Pz, T5, T6) were not connected (reduced montage); no reduced-montage analysis (decided at Checkpoint 1).
- **Exclusions (defined during re-analysis; deliberately lax for this clinical population):**
  - more than 3 bad electrodes;
  - fewer than 30 epochs not rejected by Autoreject.
- **Connectivity:** debiased estimator of the squared weighted phase lag index (dbWPLI; Vinck et al. 2011), from cross-spectra with a single DPSS taper per epoch (MNE multitaper mode, 2-Hz bandwidth, ±1 Hz smoothing), averaged over the 1-Hz bins of each band.
- **Bands:**
  - 8–13 Hz (primary);
  - individual band PDR ± 2 Hz, lower edge floored at 4 Hz (S1);
  - 4–7 Hz theta (S6, supplement).
- **Metrics** (negative dbWPLI values are kept in the averages):
  - interhemispheric strength (IHS): mean over the 64 pairs between Fp1, F7, F3, T3, C3, T5, P3, O1 and Fp2, F8, F4, T4, C4, T6, P4, O2;
  - quadrant pairs: LAQ–RAQ, LPQ–RPQ, LAQ–RPQ and RAQ–LPQ (LAQ Fp1/F3/F7, RAQ Fp2/F4/F8, LPQ T5/P3/O1, RPQ T6/P4/O2);
  - global efficiency (GE): negative weights set to 0, edge lengths = 1/weight;
  - mean connectivity: mean over all 171 pairs.

## Estimands

**Primary:** μ in Model 1 for interhemispheric strength (IHS) at 8–13 Hz, on the standardised scale (control SD units). Reported as the posterior mean, the 95% credible interval and P(μ < 0).

**Key secondary:** β_ga (group × age) in Model 2 for IHS at 8–13 Hz (the difference between the group-specific age slopes, per year).

**Exploratory** (no multiplicity adjustment; stated as such): Model 1 at 8–13 Hz for the four quadrant pairs, global efficiency (= S7) and mean connectivity; Model 2 for global efficiency; the sensitivity analyses S1–S11; the prior-sensitivity fits. The matched-set model, the matched-only Model 2 fit and the leave-one-out refits are checks, not estimands.

**Robust:** assessed only if the primary 95% credible interval excludes 0. The primary result is robust if the posterior probability of the primary's sign is at least 0.90 in each of S1, S2 (each age stratum), S3, S4 and S11, and not robust otherwise. If the primary 95% credible interval includes 0, the label is "not assessed". The credible intervals of these analyses are reported but need not exclude 0.

## ICA (amended after the code audit, before any group comparison)

- **Fitting:** FastICA (seed 42; 18 components, one fewer per interpolated electrode) on all 1-s epochs outside the excluded periods (1 Hz high-passed).
- **References:** Fp1, Fp2, F7 and F8, and the ECG electrode if less than half of its unfiltered 1–52 Hz power is 48–52 Hz mains. The EOG inputs are not used: they carried no ocular signal.
- **Scoring:** each component is correlated (Pearson r) with each reference on the continuous data outside the excluded periods (MNE's score_sources; band-pass 1–10 Hz for the frontal references, 8–16 Hz for ECG). For each reference, the correlations are z-scored once across components, without the iterative re-scoring of MNE's find_bads_eog.
- **Removal:** a component is a candidate if |z| > 3 **and** |r| ≥ 0.3 on the same reference. Candidates with more than 40% of their 1–40 Hz power in 8–13 Hz, or with larger mean absolute weights over posterior (O1, O2, P3, P4, Pz) than frontal (Fp1, Fp2, F7, F8) electrodes, are never removed. At most 5 components are removed, ranked by |r|.
- **QC:** variance removed (usable data); retention of posterior alpha power and of frontal amplitude on the 100 analysed epochs (the same epochs before and after ICA).

## Dominant posterior rhythm (PDR)

- **Spectra:** 2-s Hann windows (50% overlap) over runs of consecutive analysed epochs, averaged over the posterior electrodes (O1, O2, P3, P4, Pz) and, separately, over the central electrodes (C3, C4). Each average spectrum is parameterised with FOOOF (1.5–35 Hz; aperiodic mode 'fixed'; peak widths 1–8 Hz; at most 6 peaks).
- **Acceptance:** a posterior peak counts as the PDR only if:
  - it is at most 5 Hz wide (FOOOF bandwidth);
  - it is at least 1 dB above the aperiodic component;
  - its frequency (CF) is within ±3 Hz of the age-expected PDR (Freschl et al. 2022, linearly interpolated), with the window clipped to 5–13 Hz;
  - the aperiodic-adjusted posterior spectrum has a local maximum within CF ± 1 Hz (a bin higher than both of its neighbours in the full spectrum);
  - the rhythm itself (the power above the aperiodic component, averaged over CF ± 1 Hz) is larger posteriorly than centrally.

  The strongest qualifying peak is taken. Otherwise the child has no clear PDR.
- **Amendments to the approved plan** (decided before any group comparison):
  - the age window, which removes implausible 11–13 Hz sigma/beta picks in young children; a rhythm outside the window counts as "no clear PDR" and is therefore flagged as slowing;
  - the local-maximum and width tests, which reject fitting artefacts and shoulders;
  - after the code audit: the local-maximum test is applied as written (bug fix), the aperiodic component is fitted without a knee, posterior dominance is judged on the power of the rhythm above the aperiodic component (CF ± 1 Hz) instead of the total power at one frequency bin, and the width limit is 5 Hz instead of 4 Hz because peaks fitted without a knee are wider;
  - norms from all controls with a clear PDR, rather than from non-spindle controls, because a clear posterior rhythm implies wakefulness and the spindle detector alone flags alpha/mu bursts in awake children.
- **Age norms:** OLS on log(age + 1) in controls with a clear PDR (nominal 95% prediction interval). The same is done for the aperiodic-adjusted theta/alpha ratio (mean adjusted power at 4–7 Hz minus that at 8–13 Hz, in dB).
- **Background-slowing flag (S3):** no clear PDR, a PDR below the prediction interval, or a theta/alpha ratio above it.
- **Sleep flag (S4):** at least 0.5 sigma bursts/min (12–15 Hz, YASA spindle detector on the ICA-cleaned data, in runs of at least 5 s of consecutive analysed epochs, mean over C3, C4 and Cz) **and** no clear PDR. The spindle criterion alone flagged mostly awake children with fast alpha or mu rhythm (decided at Checkpoint 1).
- **Individual band:** the PDR rounded to the nearest 1 Hz, ± 2 Hz, with the lower edge at least 4 Hz. Without a clear PDR, the band is centred on the norm-expected PDR.

## Matching

- Each test child is matched to all controls of the same sex and the same age in completed years (column age of data/demographics_*.csv), among the analysable recordings. There is no caliper widening and no fallback match. Test children of the same sex-by-age stratum share its control mean.
- A test child without a match is left out of Model 1 and its sensitivity analyses. It stays in Model 2 if aged 2–9 years, and in S5.
- Subset analyses (S2, S3, S4, S9) remove children from both groups first and then match again, so m_s, the control means and s are recomputed. A test child whose stratum has no control left drops out.

## Model 1 — per-child difference

- **Difference:** d_i = (y_i − mean of the matched controls' y) / s, where s is the pooled within-stratum SD of the matched controls (strata with at least two controls). s is computed for each metric, band and analysis sample, so each analysis is on its own standardised scale. μ is also reported on the dbWPLI scale (μ × s).
- **Likelihood:** d_i ~ student_t(ν, μ + u[s_i] + x_i·β, σ), with u_s ~ N(0, 1/√m_s), where m_s is the number of controls in stratum s. The stratum term absorbs the sampling error of the control mean shared by several test children in the same stratum. x_i holds the covariate differences (S8a and S8b only).
- **Priors:** μ ~ N(0, 1), σ ~ Exp(1), ν ~ Gamma(2, 0.1) (ν ≥ 1), β ~ N(0, 1).
- **Posterior predictive check** of the primary fit: mean, SD, minimum and maximum of d.
- **Check (matched-set model):** the outcome is y / s, centred on the mean of the matched controls. Controls ~ t(ν, θ_s, σ_C) and tests ~ t(ν, θ_s + μ, σ_T), with θ_s ~ N(0, 2.5), μ ~ N(0, 1), σ_C, σ_T ~ Exp(1) and ν ~ Gamma(2, 0.1). Model 1 remains primary.

## Model 2 — group × age

- **Sample:** all analysable children aged 2–9 years, the age range covered continuously by both groups. Check: the children of the matched strata only, at the same ages.
- **Outcome:** y, z-scored with the mean and SD of the controls in the sample.
- **Linear predictor:** α + β_g·group + β_a·age + β_ga·group·age + β_s·sex.
- **Coding:** group 0/1 (control = 0); sex −0.5 (female) / 0.5 (male); age in completed years, centred at 5 years and divided by 2 SD of age in the sample.
- **Likelihood:** Student-t, with σ by group.
- **Priors:** Cauchy(0, 2.5) on the coefficients and Cauchy(0, 10) on the intercept (Gelman et al. 2008); σ ~ Exp(1); ν ~ Gamma(2, 0.1) (ν ≥ 1).
- **Reporting:** the group difference at age 5, the slope per year in each group, and β_ga per year with P(β_ga < 0).
- **Influence:** the key secondary is refitted once without each CTNNB1 child in the sample. A child is named as influential if leaving it out changes the sign of the posterior mean of β_ga, or whether its 95% credible interval excludes 0.

## Sensitivity analyses (IHS at 8–13 Hz with Model 1, unless noted)

| | Analysis |
|---|---|
| S1 | individual band (PDR ± 2 Hz; centred on the norm-expected PDR without a clear PDR) |
| S2 | age strata ≤4 and ≥5 years, fitted separately (no contrast between the strata) |
| S3 | without children flagged by the background-slowing proxy, in both groups |
| S4 | without recordings with signs of sleep (sleep flag), in both groups |
| S5 | all analysable children aged 19 years or less: the Model 2 likelihood and priors with a natural cubic spline of age (R splines::ns, df = 3, on the S5 sample) in place of the linear age and group × age terms; estimand β_g |
| S6 | theta 4–7 Hz |
| S7 | global efficiency instead of IHS (the exploratory global-efficiency fit) |
| S8a | covariates for data quality: the child-minus-matched-control differences in 20–40 Hz power of the analysed epochs (muscle activity) and in the number of ICA components removed, each standardised like d |
| S8b | S8a plus the difference in aperiodic-adjusted posterior alpha power. Alpha power may lie on the causal path, so S8b may remove part of a genuine effect (mediation). |
| S9 | without recordings in which Autoreject's learned thresholds fell below the channels' median peak-to-peak amplitude (median threshold ratio < 1; mis-calibration with many rejected epochs). Every recording has at least 100 epochs free of large artefacts, so an epoch-count criterion would exclude no one. |
| S10 | IHS without electrode pairs involving an electrode interpolated for the whole recording (interpolated electrodes are not independent measurements) |
| S11 | IHS without Fp1, Fp2, F7 and F8 (36 pairs) |

Prior sensitivity: Model 1 with μ ~ N(0, 0.5) and N(0, 2).

## Computation

- cmdstanr, 4 chains × (1,000 warm-up + 2,500 post-warm-up) draws, seed 42, adapt_delta 0.8, max_treedepth 10.
- Checked on every sampled parameter (not transformed parameters or generated quantities): R-hat < 1.01; bulk and tail ESS > 400; no divergent transitions; E-BFMI ≥ 0.3 in every chain; no transition at the maximum tree depth.
- A fit that fails a check is refitted once with adapt_delta 0.99 and max_treedepth 12. If it still fails, it is kept and reported with the failed checks stated.
- P is reported as ">99.9%" (or "<0.1%") when fewer than 10 of the 10,000 draws fall on the other side of 0.

## Amendment of 24 October 2026 (after the group comparison)

The sample was restricted to children aged 2–9 years in completed years (`AGE_RANGE` in config.py), the age range of the primary and key secondary analyses, so that all analyses use the same children. The recordings outside this range were removed from the derived data: 4 of children with CTNNB1 syndrome (including the excluded T029) and 11 of controls (3 under 2 years, 6 aged 10–19 years and 2 adults). As a result:

- the PDR age norms are estimated from the 54 controls aged 2–9 years with a clear PDR. This changed no individual band and the background-slowing flag of one control (C065), so S3 has 38 instead of 37 controls (μ −0.62 instead of −0.61; the primary result is still robust);
- S5 (all children up to 19 years) was dropped;
- the primary analysis, Model 2, the matched-set check and all other sensitivity analyses use the same children as before, and their results are unchanged.

The freeze manifests describe the state at the freeze, before this amendment.

## Post hoc analyses (25 October 2026, not part of the plan)

Added after the group comparison, in response to reviewer-style questions; reported as post hoc (analysis_posthoc.R):

- **Within the hemispheres:** Model 1 for intrahemispheric strength (the mean dbWPLI over the 56 pairs within each hemisphere of the 16 IHS electrodes; column `intra` of the metrics) and for the within-child contrast IHS − intrahemispheric strength.
- **Comparable alpha power:** both groups restricted to the range of aperiodic-adjusted posterior alpha power observed in both groups, then Model 1 with the child-minus-control alpha-power difference as a covariate (as in S8b, but without extrapolation).
- **Open eyes:** blinks located around the analysed epochs with the blink-rate detector before ICA (blinks_posthoc.py; data/blinks_analysed.csv); Model 1 in the children with a blink within 15 s of at least 90% of their analysed epochs.
- **Added noise:** independent 1/f noise added to the analysed epochs of each matched control child until its alpha power matched the mean of the CTNNB1 children of its stratum (bisection; noise_posthoc.py; data/noise_injection.csv); alpha connectome recomputed; Model 1.
- **Age in the individual band:** Model 2 refitted to IHS in the individual band.
- **Open eyes, checks:** thresholds of 50% and 75%, and a ±5-s window with a 50% threshold.
- **Alpha power as an outcome:** Model 1 for the aperiodic-adjusted posterior alpha power.
- **Influence and placement:** the primary refitted without each matched control child and without each stratum in turn, and without the two controls furthest above their stratum mean together; each CTNNB1 child's percentile among its matched controls (ties count half); a probabilistic index with a Bayesian-bootstrap interval; a table of the strata (results/strata.csv).
- **Sleep spindles:** the primary without the control children with a spindle rate of at least 0.5/min.
- **Age model checks:** Model 2 without each control child in turn, and in the individual band in children with a clear rhythm only.
- **Alpha-power measure:** alpha power with the aperiodic fit over 1.5–20 Hz and as absolute power (spectra_posthoc.py; data/spectral_checks.csv).
- **Epochs matched for alpha power:** IHS from the 50 analysed epochs with the least (controls) or most (CTNNB1 children) posterior alpha power, Model 1 (epochs_posthoc.py; data/epochs_alpha_matched.csv).
- **Reliability and chance level:** split-half (odd/even epochs) reliability of IHS and posterior alpha power, Spearman–Brown corrected; chance level of IHS from 200 shuffles of the right-hemisphere epochs (reliability_posthoc.py; data/ihs_reliability.csv).
- **Frequency by frequency:** dbWPLI in 1-Hz bins (2–30 Hz) and posterior power, group means (data/ihs_by_frequency.csv; Supp Fig 4).

