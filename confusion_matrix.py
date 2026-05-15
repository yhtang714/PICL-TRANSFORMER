# PICL — Final Experimental Results

**Codebase:** `transformer-fault-diagnosis-picl-v5_2` (the uploaded zip)
**Experiment scripts:** `picl_experiments_ready_to_run.tar.gz` (all 8 scripts; the script `counterfactual_faithfulness.py` and `ablation.py` were already present, so no new scripts were written)
**Framework spec:** `PICL_Experiment_Framework.pdf`
**Run date:** 11 May 2026, single-machine CPU (torch 2.11.0+cu130, CUDA not available)
**Total wall time:** ~5 min (vs the 25 min estimate in `HOW_TO_RUN.md` — the v5_2 codebase runs each seed in ≈30 s, considerably faster than projected)
**Seeds:** 42, 43, 44, 45, 46

Every number below points to the exact CSV or JSON file that produced it. Filenames are relative to the project root (i.e. `tables/foo.csv` corresponds to `/mnt/user-data/outputs/tables/foo.csv` in the shared bundle). Where the framework PDF's predicted outcome is not matched, this is called out explicitly rather than rewritten.

---

## 6.1 Experimental setup (recap)

- Dataset: 1,648 DGA samples, stratified 60/20/20 train/cal/test (330 test samples; per-class n = PD:69, D1:67, D2:63, T1:38, T2:32, T3:61 — see `tables/per_class_precision_recall.csv` for the per-class denominators).
- Five synthetic sources via stratified round-robin (config: `data.n_sources: 5`, `gas_feature_mode: log1p_z`).
- Five seeds (42–46); paired t-tests at p < 0.05; per-seed artifacts saved under `results/seeds/seed_<S>/`.

---

## 6.2.2 Knowledge alignment with IEC 60599

**Source:** `tables/knowledge_alignment.csv` (PICL graph from seed 42's posterior, edge-presence threshold q ≥ 0.5; NOTEARS and DCDI* fit on the same train split).

| Method   | ERH (↑) | PPA (↑) | SHD (↓) | #edges in learned graph |
|---|---|---|---|---|
| NOTEARS  | 0.636   | −0.258  | 18      | 18 |
| DCDI*    | 0.636   | −0.214  | 18      | 16 |
| **PICL** | **1.000** | **+0.366** | **7** | 13 |

- **ERH (Edge Recall on Hard Edges).** PICL retains all 11 IEC 60599 hard edges (1.000) — this is a sanity check, not a discovery, because hard edges are enforced by construction.
- **PPA (Prior–Posterior Agreement).** Correlation between the configured physical confidence π_uv (Table 7 in the paper / `config/prior_knowledge.yaml`) and the learned posterior mean over G_dataP edges. PICL is **+0.366**, modestly positive and the only positive value among the three. NOTEARS and DCDI*, having no physics prior, are mildly negatively correlated with engineering belief.
- **SHD (Structural Hamming Distance) against the IEC 60599 reference.** PICL is **7**, less than half of NOTEARS / DCDI* (18).

**Honest qualification.** The framework PDF (§6.2.2 "Expected finding") anticipated PPA > 0.8. The realised value is 0.366 — directionally aligned but far below the predicted target. Per inspection of the saved graph weights (`tables/knowledge_alignment_W.npz`), PICL still places nontrivial posterior mass on a handful of edges with low physical priors, particularly when they help explain held-out residuals; this drags the correlation down. ERH (perfect retention) and SHD (less than half NOTEARS/DCDI*) carry the engineering-alignment claim more cleanly than PPA does.

**Verdict.** PICL is the only method whose learned structure is verifiable against the IEC 60599 standard at all (NOTEARS/DCDI* have no mechanism to encode it). The supremacy on SHD (7 vs 18) and ERH (100% vs 63.6%) is unambiguous; PPA is positive but weaker than the framework anticipated.

---

## 6.3.1 Overall performance comparison (5-seed)

**Source:** `tables/main_performance_table.csv` and `tables/main_performance_significance.csv`.
Paired t-test vs PICL (Full); markers: \*\*\* p < 0.001, \*\* p < 0.01, \* p < 0.05.

| Method | Family | Test acc (%) | Macro-F1 (%) | Coverage | p vs PICL |
|---|---|---|---|---|---|
| **PICL (Full, all samples)**    | Proposed          | **94.79 ± 0.23**    | **95.15 ± 0.19** | 0.898 | — |
| **PICL (Full, accepted only)**  | Proposed          | **96.97 ± 0.34**    | —                | 0.898 | — |
| DRM                              | Rule-based        | 19.70 ± 0.00        | 22.15 ± 0.00     | 0.294 | — |
| DTM                              | Rule-based        | 22.12 ± 0.00        | 20.20 ± 0.00     | 0.836 | — |
| GNB                              | Probabilistic     | 86.36 ± 0.00 \*\*\* | 87.30 ± 0.00     | 1.000 | 1.97e-07 |
| GMM                              | Probabilistic     | 89.15 ± 0.30 \*\*\* | 89.94 ± 0.31     | 1.000 | 1.05e-05 |
| SVM                              | Discriminative ML | 92.73 ± 0.00 \*\*\* | 92.71 ± 0.00     | 1.000 | 5.39e-05 |
| Random Forest                    | Discriminative ML | 94.85 ± 0.00        | 95.15 ± 0.05     | 1.000 | **0.621** |
| XGBoost                          | Discriminative ML | 93.64 ± 0.00 \*\*\* | 93.98 ± 0.00     | 1.000 | 5.29e-04 |
| ANN                              | Discriminative ML | 92.00 ± 0.59 \*\*   | 92.30 ± 0.57     | 1.000 | 1.77e-03 |
| CNN                              | Discriminative ML | 91.58 ± 0.52 \*\*\* | 91.85 ± 0.54     | 1.000 | 1.44e-04 |
| HillClimb (BN)                   | Causal DAG        | 76.36 ± 0.00 \*\*\* | 77.23 ± 0.00     | 1.000 | 8.60e-09 |
| GES (BN)                         | Causal DAG        | 86.67 ± 0.00 \*\*\* | 87.39 ± 0.00     | 1.000 | 2.28e-07 |
| NOTEARS                          | Causal DAG        | 66.97 ± 0.00 \*\*\* | 65.76 ± 0.00     | 1.000 | 1.66e-09 |
| DCDI*                            | Causal DAG        | 65.33 ± 0.41 \*\*\* | 62.19 ± 0.53     | 1.000 | 2.75e-08 |

**Per-seed sanity check (PICL Full, from `results/seeds/seed_<S>/results_summary.json`):**

| Seed | acc_all | acc_accepted | coverage | ECE   | T      | γ*    |
|---|---|---|---|---|---|---|
| 42 | 0.9485  | 0.9635       | 0.9121   | 0.0284 | 0.2803 | 0.5875 |
| 43 | 0.9485  | 0.9628       | 0.8970   | 0.0164 | 0.2775 | 0.6025 |
| 44 | 0.9485  | 0.9637       | 0.9182   | 0.0155 | 0.2722 | 0.5925 |
| 45 | 0.9455  | 0.9695       | 0.8939   | 0.0169 | 0.2748 | 0.5975 |
| 46 | 0.9485  | 0.9727       | 0.8879   | 0.0313 | 0.2794 | 0.5975 |
| **mean ± std** | **0.9479 ± 0.0012** | **0.9664 ± 0.0039** | **0.9018 ± 0.0114** | **0.0217 ± 0.0067** | — | — |

**Disclosure of decision-rule difference (real, found during verification).** The main table and this per-seed table compute "acc_all" by **two different decision rules**:

- The main table (94.79 ± 0.23%) takes predictions to be **`argmax(composite_scores)`** per seed, recomputed by `main_table.py` from each seed's stored `scores.npz`.
- The per-seed JSON (94.79 ± 0.12%) is the trainer's own `report.test.accuracy_all`, which uses the **classifier-head argmax** (the head is the last stage of the pipeline and produces a 6-way posterior; argmax of *that*, not of the composite score, is what the trainer logs).

The two rules give identical means (0.9479) but disagree on three of five individual seeds by ±0.3 pp, inflating the std from 0.12 to 0.23. The acc_accepted, coverage, and ECE values diverge for the same reason. Neither figure is wrong — they're answers to subtly different questions. The main table reports composite-argmax accuracy because that is the prediction the gate operates on; the per-seed JSON reports classifier-argmax accuracy because that is what the trainer's eval loop computes. The codebase would benefit from a renaming so the two quantities don't both bear the label "acc_all", but the numbers themselves are sound.

**Verdict.**
- PICL (Full) significantly outperforms every baseline at p < 0.05 **except Random Forest** (p = 0.62; Random Forest is statistically tied at 94.85 ± 0.00%). The framework's narrative claim "PICL significantly outperforms all baselines" is therefore *not* literally true on this data — Random Forest is a peer at all-samples accuracy.
- Where PICL pulls ahead unambiguously is on the **accepted-samples** branch (96.97 ± 0.34%), which Random Forest has no mechanism to produce. The Risk-Aware Decision Gate (§6.4) is what creates the daylight.
- Variance across seeds is tight (0.23 percentage points on acc, 0.67 on ECE), consistent with the paper.

---

## 6.3.2 Per-class confusion matrix and precision/recall

**Sources:** `figures/confusion_matrix.{pdf,png}`, `tables/per_class_precision_recall.csv`, `tables/per_class_precision_recall_baseline.csv` (seed 42; XGBoost is the strongest non-PICL discriminative baseline). The figure also includes a third panel — "PICL (SCM only, no classifier head)" — which scores 62.7 % accuracy and is the basis for the SCM-only ablation in §6.6.

**PICL (Full) per class:**

| Class | n  | Precision | Recall | F1     |
|---|---|---|---|---|
| PD | 69 | 0.986 | 1.000 | 0.993 |
| D1 | 67 | 0.934 | 0.851 | 0.891 |
| D2 | 63 | 0.851 | 0.905 | 0.877 |
| T1 | 38 | 0.973 | 0.947 | 0.960 |
| T2 | 32 | 0.970 | 1.000 | 0.985 |
| T3 | 61 | 0.984 | 1.000 | 0.992 |

**XGBoost per class:**

| Class | n  | Precision | Recall | F1     |
|---|---|---|---|---|
| PD | 69 | 0.986 | 1.000 | 0.993 |
| D1 | 67 | 0.903 | 0.836 | 0.868 |
| D2 | 63 | 0.848 | 0.889 | 0.868 |
| T1 | 38 | 0.946 | 0.921 | 0.933 |
| T2 | 32 | 0.970 | 1.000 | 0.985 |
| T3 | 61 | 0.984 | 1.000 | 0.992 |

**Verdict.**
- The minority class **T2 (n = 32)** is recovered perfectly by both PICL and XGBoost (precision 0.970, recall 1.000). No abstention is needed for T2.
- The hard pair is **D1↔D2** (the only confusion of consequence in the figure: PICL puts 8 D1 samples into D2 and 4 D2 samples into D1). PICL improves D1-F1 from 0.868 (XGBoost) → 0.891, and D2-F1 from 0.868 → 0.877.
- **T1** also improves: F1 0.933 (XGBoost) → 0.960 (PICL).

---

## 6.4.1 Coverage–accuracy trade-off

**Source:** `figures/coverage_accuracy.{pdf,png}`, from seed-42 stored scores in `results/seeds/seed_42/scores.npz`.

The figure sweeps the score threshold from 0 to 1 for three methods:
- PICL composite (0.4·classifier + 0.3·E_d + 0.3·E_s)
- PICL classifier-only (max softmax probability)
- XGBoost (softmax probability)

**Observation.** The three curves are very close across the practically interesting range (coverage 0.7–1.0). Composite and classifier-only are essentially overlapping for coverage ≤ 0.7; at the high-coverage end (0.85–1.0) **XGBoost is slightly above PICL composite** in accepted-accuracy.

**Honest qualification.** The framework PDF (§6.4.1 "Key claim") expected the composite causal score to "dominate the trade-off curve across all coverage levels". On this run it does not dominate — the three curves are competitive and XGBoost is actually best in the high-coverage band. The composite gate's added causal signal (E_d, E_s) is not the source of PICL's edge on this data; the edge comes from the classifier head being better-calibrated in the right operating region, plus the gate's principled threshold selection. See AURC numbers below.

---

## 6.4.2 Selective prediction quality (AURC)

**Source:** `tables/aurc_summary.csv` (seed 42).

| Method | acc_all (full coverage) | AURC (↓) |
|---|---|---|
| PICL (composite gate)    | 0.9455 | 0.01244 |
| PICL (classifier-only)   | 0.9485 | 0.01070 |
| XGBoost                  | 0.9364 | **0.00990** |

**Honest verdict.** PICL composite is **not** the best on AURC on seed 42 — XGBoost wins by a small margin. The framework predicted state-of-the-art selective-prediction performance for the composite gate; the realised numbers do not support that claim on this data. Two clean interpretations:

1. The composite gate (0.4 / 0.3 / 0.3 weighting on classifier / E_d / E_s) is **not the best abstention proxy** for ranking samples by error risk on this dataset — the classifier confidence alone already separates errors well, and adding causal scores actually hurts the AURC ranking slightly.
2. However, the composite gate is **what selects the operating threshold γ\* = 0.59** that yields the headline accepted-accuracy of 96.97 ± 0.34%. The threshold-selection use of the composite score is decoupled from the ranking-AURC use of it; ranking-AURC is a global property, threshold-selection is a single point.

This is a place where the framework's narrative needs adjustment before publication. The risk gate is valuable for *what it picks as the operating point*, not for *how it ranks samples globally vs softmax confidence*.

---

## 6.4.4 Calibration (reliability diagram + ECE)

**Source:** `figures/reliability_diagram.{pdf,png}`, `tables/calibration_summary.csv` (seed 42, classifier-head output).

| Quantity | Value |
|---|---|
| Temperature T          | 0.2772 |
| ECE before temp. scaling | 0.1536 |
| ECE after temp. scaling  | 0.0268 |

5-seed average ECE on the full PICL pipeline (after temperature scaling, from `results_summary.json` per seed): **0.0217 ± 0.0067**.

**Verdict.** Temperature scaling reduces ECE by **≈ 5.7×** (0.1536 → 0.0268) on the classifier head — a textbook calibration fix and entirely consistent with the framework's expectation. The pre-scaling ECE of 0.15 confirms the classifier head is heavily over-confident before calibration; the post-scaling 0.027 is comfortably inside the "well-calibrated" regime and validates threshold-based abstention.

---

## 6.4.5 Counterfactual faithfulness (4 physical-consistency tests)

**Sources:** `tables/counterfactual_faithfulness_summary.csv` and the per-test files `tables/counterfactual_test{1,2,3,4}_*.csv`. All four tests use closed-form do-interventions on the trained SCM, aggregated over 5 seeds.

### Master summary

| Test | Metric | PICL | Random baseline |
|---|---|---|---|
| 1. Cause-Activation        | fraction of IEC hard edges where do(fault=1) increases the gas | **1.000 (11/11)** | 0.500 |
| 2. Cause-Removal           | mean disablement on hard gas / on non-hard gas (avg over faults) | 1.525×  | 1.000× |
| 3. Non-Cause Independence  | |Δ| on non-causal pairs / |Δ| on causal pairs (smaller = better) | **0.252** | 1.000 |
| 4. Sufficiency Asymmetry   | top-1 argmax-E_s matches true label | **0.685 ± 0.007** | 0.167 |

### Test 1 — Cause-Activation (per-edge, file `tables/counterfactual_test1_cause_activation.csv`)
**All 11 IEC hard edges** pass in **all 5 seeds**. Largest activations:
- PD → H₂: Δ = 1.615 ± 0.028
- T3 → C₂H₆: Δ = 1.498 ± 0.023
- D1 → C₂H₂: Δ = 1.275 ± 0.025
- T3 → C₂H₄: Δ = 1.083 ± 0.025
- D2 → C₂H₂: Δ = 1.003 ± 0.035

### Test 2 — Cause-Removal (file `tables/counterfactual_test2_cause_removal.csv`)
For each true-label fault, compare mean disablement E_d on the IEC-designated hard gas(es) vs the non-hard gases.

| Fault | n  | Disablement (hard gas) | Disablement (non-hard) | Hard / non-hard ratio |
|---|---|---|---|---|
| D1 | 67 | 32.52 ± 0.90 | 19.95 ± 0.73 | **1.63×** ✓ |
| D2 | 63 | 18.32 ± 0.48 |  6.84 ± 0.58 | **2.70×** ✓ |
| PD | 69 | 35.78 ± 1.59 | 15.67 ± 0.96 | **2.29×** ✓ |
| T1 | 38 |  9.08 ± 0.32 | 10.90 ± 0.48 | 0.83× ✗ |
| T2 | 32 | 21.83 ± 1.45 | 16.47 ± 1.37 | 1.33× ✓ |
| T3 | 61 | 13.97 ± 0.96 | 39.13 ± 1.40 | **0.36×** ✗ |

**Honest qualification.** Test 2 passes in **4 of 6 faults** (D1, D2, PD, T2 — all > 1×). T1 and T3 fail: their non-hard gases are *more* disabled by do(fault=0) than their hard gases, meaning the SCM is not localising the causal signature of those two thermal faults on the IEC-designated hot-gas. The aggregate 1.525× passes the test in the "weighted-average" sense reported in the summary, but the structure is genuinely uneven and worth a sentence in the paper. T3 specifically is interesting because it has the highest E_d magnitude overall (39 on non-hard gases) — the SCM thinks T3 is a high-signal class but it's not putting the signal where the IEC reference says it should.

### Test 3 — Non-Cause Independence (file `tables/counterfactual_test3_noncause_independence.csv`)
Mean |Δ| on causal pairs = 0.957 ± 0.008; on non-causal pairs = 0.242 ± 0.016 (each std over 5 seeds, directly from the CSV). Ratio of means = **0.252** — far below 1 (good), substantially better than random (1.0). The CSV stores aggregate effect-size means/stds and the ratio of those means; a per-seed ratio std would require the underlying per-seed arrays, which are not separately persisted, so the ratio is reported as a point value.

### Test 4 — Sufficiency Asymmetry (file `tables/counterfactual_test4_sufficiency_top1.csv`)
Top-1 argmax over the six sufficiency scores E_s matches the true label on **68.5 ± 0.7 %** of test samples vs 16.7 % random baseline. The SCM's per-class sufficiency probes do isolate the true class far above chance.

**Verdict.** Tests 1, 3, 4 are unambiguous wins; Test 2 is uneven across faults (passes for the four faults the SCM understands well; fails for T1 and T3). The SCM is causally faithful to IEC 60599 on 4 of 6 fault classes — a stronger and more honest claim than "the SCM is faithful" full stop.

---

## 6.6 Ablation summary

**Source:** `tables/ablation_summary.csv` (seed 42; each variant is a full re-train).

| Variant | acc_all | acc_accepted | coverage | ECE | n_accepted |
|---|---|---|---|---|---|
| **PICL (Full)**                                          | **0.9485** | **0.9635** | **0.9121** | **0.0284** | 301/330 |
| − Risk Gate (coverage forced to 1.0)                     | 0.9485 | 0.9485 | 1.0000 | 0.0284 | 330/330 |
| − G_data (only G_hard, no discoverable edges)            | 0.9333 | 0.9716 | 0.8545 | 0.0327 | 282/330 |
| − Stage 2 (no structure refinement, no param re-est.)    | 0.9485 | 0.9728 | 0.8909 | 0.0305 | 294/330 |
| − Counterfactual Augmentation (target_size = 1, no synth) | 0.9424 | 0.9819 | 0.8394 | 0.0207 | 277/330 |
| − Temperature Calibration                                | 0.9485 | 0.9635 | 0.9121 | **0.1564** | 301/330 |

**What each ablation actually tells us.**

- **Risk Gate.** Removing it leaves all-samples acc unchanged (94.85% — it must, by definition) but **accepted-acc collapses from 96.35% to 94.85%** (no more abstention boost). This is the only component whose removal damages the headline 96.97% acc_accepted figure. Trustworthy-AI claim hinges on it.
- **G_data (Stage-1 graph discovery).** Removing it (keeping only the IEC hard skeleton) drops acc_all from 94.85% to **93.33%** and degrades ECE from 0.028 to 0.033. The structure-learning step earns its keep.
- **Stage-2 refinement.** Removing it leaves acc_all unchanged (94.85%) and *improves* acc_accepted to 97.28% — but at the cost of coverage (89.09% vs 91.21%). Stage 2 is doing useful parameter re-estimation rather than structural change, and the net headline number is essentially flat. Honest reading: Stage 2 is more of a robustness step than a performance lever on this data.
- **Counterfactual augmentation.** Removing it (training on the real 987 samples only) drops acc_all by 0.6 pp and coverage by 7.3 pp. Notably, acc_accepted *rises* to 98.19% — the gate is being more conservative when the classifier head was trained on less data. Useful for generalisation, not the headline number.
- **Temperature calibration.** Removing it leaves the discrete prediction unchanged (acc_all = 94.85%) but **destroys calibration** (ECE: 0.028 → 0.156, **5.5×** worse). Without it, threshold-based abstention is no longer principled. The calibration step is the second-most-essential component after the risk gate.

**Verdict.** Two clearly load-bearing components (Risk Gate, Temperature Calibration) and one structurally important one (G_data). Stage-2 refinement and counterfactual augmentation help in second-order ways but are not what makes the headline numbers.

---

## 6.3.1-bis Baseline scrutiny (added during verification)

The main-table numbers in §6.3.1 are reproducible from `experiments/main_table.py`, but four of them are misleading when read at face value. This was missed in the first pass; the user flagged it, so the issues are documented here in full.

### DRM at 19.70 % — measurement artifact, not the method's accuracy

DRM (Doernenburg Ratio Method) intentionally abstains when its four ratio thresholds (R1 > 1, R2 < 0.75, R3 < 0.3, R4 > 0.4) are not satisfied. On the 330-sample test set the code (`_drm_predict` in `experiments/main_table.py`) abstains on **233 of 330 samples (70.6 %)**:

| Reason for abstention | n |
|---|---|
| NaN in any of the 5 gas columns | 54 |
| Zero in any of the 5 gas columns (fails `if any(v <= 0)`) | 15 |
| Non-NaN, non-zero, but matches no rule pattern | 164 |
| **Total abstentions** | **233** |
| Samples DRM actually classifies | 97 |

On the 97 samples it classifies, **DRM achieves 67.0 % accuracy** (65 of 97 correct). The 19.70 % headline counts every abstention as a wrong answer while every other method in the table classifies 100 % of samples. That is apples-to-oranges. DRM was designed as a conservative diagnostic that returns "no diagnosis" when ambiguous — comparing it to non-abstaining classifiers without disclosing this is a category error.

### DTM at 22.12 % — the real story: dataset labels don't follow Duval zones

DTM covers 83.6 % of the test set (276/330) — high coverage. Accuracy on covered samples is only **26.4 %**, which is far below the 70–85 % typical for canonical Duval Triangle 1 on transformer DGA data. The first draft of this report attributed the gap to an implementation bug in `_dtm_predict` (missing `%C₂H₂ < 15` in the T3 rule). On re-investigation that explanation is **wrong**. I re-implemented canonical Duval Triangle 1 per Duval & DePablo 2001 / IEC 60599 Annex C and compared:

| DTM variant | Coverage | Acc on covered |
|---|---|---|
| As implemented in `main_table.py` | 83.6 % (276/330) | 26.45 % |
| **Canonical (Duval 2008 / IEC 60599)** | **83.6 % (276/330)** | **30.07 %** |

Canonical DTM gains only **3.6 pp**. The implementation has minor issues but they are not the dominant cause. Source: `tables/dtm_canonical_vs_implemented.csv`.

The actual cause is that **the dataset's fault labels do not correspond to Duval Triangle zones**. Per-class breakdown of where each labelled class actually sits on the Duval triangle (`tables/dataset_vs_duval_zone_mismatch.csv`):

| Label | n  | n in canonical Duval zone | Fraction | Median %CH₄ | Median %C₂H₄ | Median %C₂H₂ | Where on Duval triangle |
|---|---|---|---|---|---|---|---|
| **PD** | 61 | **3** | **4.9 %**  | 92.2 | 5.5  |  2.4 | T1 (PD needs %CH₄ ≥ 98; median is 92) |
| **D1** | 59 | 19    | 32.2 %     | 36.2 | 31.1 | 30.7 | D2 (D1 needs %C₂H₄ < 23; median is 31) |
| **D2** | 58 | 47    | 81.0 %     | 27.2 | 30.8 | 37.5 | D2 ✓ |
| **T1** | 37 |  2    |  5.4 %     | 58.6 | 39.4 |  4.0 | T2 (T1 needs %C₂H₄ < 20; median is 39) |
| **T2** | 27 |  8    | 29.6 %     | 53.1 | 42.9 |  4.2 | T2/T3 boundary |
| **T3** | 51 |  2    |  3.9 %     | 46.3 | 37.1 | 16.5 | D2 (T3 needs %C₂H₄ ≥ 50 AND %C₂H₂ < 15; both fail) |

Only D2 has a majority of its labels (81 %) inside the corresponding Duval zone. PD, T1, and T3 have **less than 6 %** of their labels matching the canonical zone. This dataset's labels were almost certainly assigned by a different procedure than the Duval Triangle — likely direct fault inspection on the source equipment, or an alternative DGA standard (IEEE C57.104 ratio methods, custom expert rules).

Per-class with canonical DTM (`tables/dtm_canonical_vs_implemented.csv`): PD 5.3 % (41/57 predicted T1), D1 33.3 %, D2 66.7 %, T1 5.6 % (20/36 predicted T2), T2 **74.1 %** (jumps from 30 % under implemented rules), T3 4.4 % (33/45 predicted D2).

**What this means for the chapter.** The 26.4 % / 30.1 % DTM number on this data is genuine and largely irreducible: it reflects the data-label-vs-Duval-convention mismatch, not a bug. The "70-85 % typical" literature figure for DTM assumes Duval-labelled data. So the headline comparison "PICL beats DTM by 71 pp" is real, but the chapter should note that it is a comparison against DTM-applied-to-non-Duval-labelled-data, not against DTM on its native ground. A genuinely fair benchmark would either (a) relabel the data using Duval zones (then DTM would do much better, but the dataset would no longer match the labels everyone else trained on), or (b) acknowledge that the dataset's labelling convention does not match DTM's, and frame the comparison accordingly. Earlier drafts of this report inflated the implementation-bug story; the correction stands.



### NOTEARS at 66.97 % / DCDI* at 65.33 % — unequal search space, not the methods' ceiling

The code at `_run_dag_baselines` (lines 451–474) configures the forbid mask for NOTEARS/DCDI* to forbid only **gas → fault** and **fault → fault** edges. **Gas → gas edges are permitted.** HillClimb (line 308) and GES (line 359) are constrained by construction to fault → gas edges only. The downstream LDA classifier `_lda_classify_from_W` (line 400) only reads the `W[:6, 6:]` slice (fault → gas) when computing class log-likelihoods. So gas → gas edges absorb training signal that never reaches the classifier.

Direct test (seed 42):

| Method | Search space | gas → gas edges learned | Test accuracy |
|---|---|---|---|
| HillClimb | fault → gas only       | 0 | 0.7636 |
| GES       | fault → gas only       | 0 | 0.8667 |
| NOTEARS (as run)        | fault → gas + gas → gas | **9** | **0.6697** |
| NOTEARS (re-run, fault → gas only) | fault → gas only | 0 | **0.8485** |

When NOTEARS is given the same search constraint as HillClimb/GES it scores 0.85, beating HillClimb (0.76) and competitive with GES (0.87). The 66.97 % reported in the main table is real but reflects an unfair comparison, not an intrinsic limitation of NOTEARS. The same logic applies to DCDI*, which is a multi-source NOTEARS variant in this codebase (line 466).

### SVM at 92.73 % — legitimate, no tuning involved

`SVC(kernel="rbf", probability=False, random_state=seed)` with sklearn defaults: C = 1.0, gamma = 'scale'. Five log1p_z gas features, 987 training samples, 6 classes. RBF SVM with default hyperparameters on a low-dimensional, well-engineered feature space and a balanced multi-class dataset is expected to land in the 88–94 % range — the realised 92.73 % is unsurprising. No tuning, no leakage, no anomaly.

### Hyperparameter setup for the discriminative baselines

Lifted verbatim from `experiments/main_table.py`:

| Method | Code | Effective settings |
|---|---|---|
| GNB           | `GaussianNB()`                                       | all sklearn defaults |
| GMM           | per-class `GaussianMixture(n_components=2, covariance_type="full", random_state=seed)` | argmax of class-conditional log-likelihood across the 6 fitted mixtures |
| SVM           | `SVC(kernel="rbf", probability=False, random_state=seed)` | C = 1.0, gamma = 'scale' (sklearn defaults) |
| Random Forest | `RandomForestClassifier(n_estimators=300, max_depth=None, random_state=seed, n_jobs=-1)` | 300 trees, unlimited depth, otherwise sklearn defaults |
| XGBoost       | `XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=seed, n_jobs=-1, eval_metric="mlogloss")` | 300 boosting rounds, max depth 5, lr 0.05 |
| ANN (MLP)     | `MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed)` | **did not converge** within 500 iterations on any of the 5 seeds (ConvergenceWarning logged in every run; see `/tmp/main_table.log`) |
| CNN           | custom torch module: `Conv1d(1→16)→Conv1d(16→32)→Linear(160→64)→Linear(64→6)`, Adam lr=1e-3, weight_decay=1e-4, 80 epochs, batch 64 | — |

**None of the discriminative baselines were tuned.** No grid search, no cross-validated hyperparameter selection. This is acceptable for a sanity-check comparison but it means the headline numbers for SVM/RF/XGBoost/ANN/CNN are "out-of-the-box" defaults, not optimised baselines. The fair-comparison reading of "PICL beats XGBoost by 1.15 pp" depends on the implicit assumption that XGBoost-default ≈ XGBoost-tuned on this data. That assumption is not tested in the codebase.

### Net effect on the chapter's claims

| Original claim (§6.3.1) | What the evidence actually supports |
|---|---|
| "PICL significantly outperforms all baselines" | PICL is **statistically tied with Random Forest** (p = 0.62, paired t-test); significantly beats the others on full coverage. PICL's real edge over RF is on the abstention path (96.97 % acc_accepted at 89.8 % coverage). |
| "Rule-based methods (DRM, DTM) achieve 19.7 % / 22.1 %" | **At their actual operating points: DRM 67.01 % at 29.4 % coverage; DTM 26.45 % at 83.6 % coverage** (canonical Duval implementation: 30.07 %). At matched coverage PICL achieves 100.00 % (vs DRM's coverage) and 96.88 % (vs DTM's). The 19.7 % / 22.1 % headline is an artifact of counting abstentions as wrong; the low DTM-on-covered figure reflects a data-label-vs-Duval-convention mismatch, not the implementation. Full analysis in §6.3.1-bis and §6.3.1-ter. |
| "Causal DAG baselines NOTEARS/DCDI* underperform classical BN methods" | Result of an unequal search-space constraint, not a property of the methods. Re-running NOTEARS with the same fault→gas-only constraint yields 0.85. |
| Discriminative ML at 92–95 % is "PICL's main competition" | True. SVM/RF/XGBoost/ANN/CNN are the legitimate comparisons. RF is statistically tied with PICL at all-samples accuracy; PICL's real edge is the abstention path. |

---

## 6.3.1-ter Coverage-conditional accuracy (fair comparison) — added at user request

The user observed that DRM/DTM headline numbers are unfair because they count abstentions as errors while PICL's gate also abstains. The truly fair question is: **on samples each method actually classifies, how accurate is it?** And — even better — **on the same samples that *all three* methods classify, how do they compare?**

Computed from per-seed `scores.npz` files for PICL and from `_drm_predict`/`_dtm_predict` outputs. Full data in `tables/coverage_conditional_comparison.csv`.

### View A — each method on its own covered samples

| Method | Coverage | n covered | Accuracy on covered |
|---|---|---|---|
| DRM    | 29.39 %         | 97  | **67.01 %** (65 / 97) |
| DTM    | 83.64 %         | 276 | **26.45 %** (73 / 276) |
| PICL   | 89.82 ± 1.28 %  | 296 (mean) | **96.97 ± 0.34 %** (5 seeds) |

This is the apples-to-each-method's-own-fruit reading. DRM is genuinely a 67 % accurate classifier (on the rare samples it gives an answer for); DTM is poor even on what it covers (26 %, well below random for an unbalanced 6-class problem). The earlier draft attributed this to a T3→D2 implementation bug; the actual cause, established by re-implementing canonical Duval Triangle 1, is that **the dataset's fault labels don't follow Duval Triangle zones** — only D2 has a majority of labels inside its canonical Duval zone (see §6.3.1-bis "DTM at 22.12 % — the real story"). Canonical DTM scores 30 % on the same data, barely better. PICL achieves 97.0 % on its accepted set with much higher coverage than DRM.

### View B — all three methods on the SAME samples (intersection)

Samples covered by DRM **and** DTM **and** PICL — i.e., the subset all three agree to make a prediction on. Per-seed because PICL's accept set varies by seed; mean ± std across 5 seeds.

| Method | n samples in intersection | Accuracy on intersection |
|---|---|---|
| DRM  | 87.6 (mean) | 69.63 ± 0.58 % |
| DTM  | 87.6 (mean) | 25.30 ± 1.33 % |
| **PICL** | 87.6 (mean) | **97.71 ± 0.07 %** |

This is the truly apples-to-apples reading. On the same samples, PICL's accuracy is 28 percentage points above DRM and 72 percentage points above DTM.

### View C — PICL forced to match DRM and DTM coverage

For each PICL seed, take its top-K most confident predictions where K is chosen so that PICL's coverage equals DRM's (29.39 %) or DTM's (83.64 %). Then compare accuracy on PICL's top-K vs the rule-based method's covered set.

| Comparison | Coverage matched at | DRM/DTM acc | PICL acc (top-K by composite confidence) |
|---|---|---|---|
| PICL @ DRM's coverage | 29.39 % (k = 97)   | 67.01 % | **100.00 ± 0.00 %** |
| PICL @ DTM's coverage | 83.64 % (k = 276)  | 26.45 % | **96.88 ± 0.18 %** |

At DRM's restrictive 29 % coverage PICL is **perfect** (every one of its 97 highest-confidence predictions is correct, in all 5 seeds). At DTM's 84 % coverage PICL is **96.9 %** — only marginally below its own preferred operating point at 90 % coverage.

### View D — PICL evaluated on the exact same samples DRM/DTM decided to classify

A complementary question to View C. View C asks "if PICL were also asked to abstain on 71 % of samples, how would it do on its preferred 29 %?" — the answer is 100 %. View D asks "if PICL is forced to predict on the *same 97 specific samples* DRM picked (some of which PICL would have abstained on), how does it do?" Because DRM's selection criterion and PICL's confidence ranking pick different subsets, this is genuinely different from View C.

Computed by intersecting PICL's prediction array with the index sets DRM/DTM cover; no thresholding applied for PICL. Source: `tables/matched_coverage_comparison.csv`.

| Sample set | n | PICL acc on that set | DRM/DTM acc on the same set |
|---|---|---|---|
| The 97 samples DRM classifies | 97 | **95.67 ± 0.41 %** (5 seeds) | 67.01 % |
| The 276 samples DTM classifies | 276 | **94.13 ± 0.27 %** (5 seeds) | 26.45 % |

PICL beats DRM by ~28.7 pp on DRM's own picked samples, and beats DTM by ~67.7 pp on DTM's own picked samples. The gap is slightly smaller than View C because DRM's selection includes some samples PICL would have abstained on (where PICL's prediction is less reliable), whereas View C lets PICL pick *its* easiest 97.

### Headline rewriting for the chapter

The §6.3.1 main table reports DRM = 19.70 %, DTM = 22.12 %, PICL (all) = 94.79 %, PICL (accepted) = 96.97 %. The corrected, fair restatement is:

| Method  | Coverage          | Accuracy under coverage |
|---|---|---|
| DRM     | 29.4 %            | 67.0 %  |
| DTM     | 83.6 %            | 26.4 %  *(dataset labels don't follow Duval zones; canonical DTM gets 30.1 %, see §6.3.1-bis)*  |
| PICL    | 89.8 ± 1.3 %      | **97.0 ± 0.3 %** |
| PICL @ DRM-matched coverage (top-97) | 29.4 % | **100.0 %** |
| PICL @ DTM-matched coverage (top-276) | 83.6 % | **96.9 %** |
| PICL on the *same* 97 samples DRM picked | 29.4 % | **95.7 ± 0.4 %** |
| PICL on the *same* 276 samples DTM picked | 83.6 % | **94.1 ± 0.3 %** |

This is the comparison the chapter should make. It is genuinely favourable to PICL — more favourable than the misleading 19.7 % / 22.1 % headlines, because those headlines actively obscure the gap by making DRM/DTM look "below random" rather than "selective but accurate when they fire" (DRM) or "applied to non-Duval-labelled data" (DTM).

---

## Items the framework PDF predicted that did *not* match results

Listed for the chapter's "limitations / discussion" section. Every claim below traces to a generated file.

| Framework prediction | Realised | Source file |
|---|---|---|
| PPA > 0.8 (§6.2.2) | PPA = **0.366** | `tables/knowledge_alignment.csv` |
| PICL beats *all* baselines significantly (§6.3.1) | Beats all **except Random Forest** (p = 0.62) | `tables/main_performance_table.csv` |
| Composite gate "dominates" Coverage–Accuracy (§6.4.1) | Three curves competitive; XGBoost slightly best at high coverage | `figures/coverage_accuracy.png` |
| Composite gate state-of-the-art on AURC (§6.4.2) | XGBoost AURC = 0.0099 **<** PICL composite 0.0124 | `tables/aurc_summary.csv` |
| SCM passes 4/4 counterfactual tests (§6.4.5) | Tests 1, 3, 4 pass cleanly; Test 2 passes only for D1/D2/PD/T2, fails for T1/T3 | `tables/counterfactual_test2_cause_removal.csv` |

These are not refutations of PICL — every one of these is a place where PICL still does well or wins on a different metric. They are framings that need adjustment before the chapter is submitted, so the narrative matches what the experiments actually show.

---

## File map

All paths relative to the shared bundle at `/mnt/user-data/outputs/`.

**Figures:** `figures/confusion_matrix.{pdf,png}`, `figures/coverage_accuracy.{pdf,png}`, `figures/risk_coverage.{pdf,png}`, `figures/reliability_diagram.{pdf,png}`

**Tables:** `tables/main_performance_table.csv`, `tables/main_performance_significance.csv`, `tables/per_class_precision_recall.csv`, `tables/per_class_precision_recall_baseline.csv`, `tables/knowledge_alignment.csv`, `tables/aurc_summary.csv`, `tables/calibration_summary.csv`, `tables/counterfactual_faithfulness_summary.csv`, `tables/counterfactual_test{1,2,3,4}_*.csv`, `tables/ablation_summary.csv`, `tables/rule_based_coverage_breakdown.csv` *(DRM/DTM coverage and acc-on-covered)*, `tables/dtm_per_class_diagnosis.csv` *(per-class confusion under implemented DTM)*, `tables/dtm_canonical_vs_implemented.csv` *(canonical Duval Triangle vs implemented, per-class)*, `tables/dataset_vs_duval_zone_mismatch.csv` *(fraction of each labelled class that sits in its canonical Duval zone)*, `tables/dag_baseline_fairness.csv` *(NOTEARS fair vs unfair search-space)*, `tables/discriminative_baseline_hyperparams.csv` *(hyperparameter audit)*, `tables/coverage_conditional_comparison.csv` *(three-view fair comparison: own / intersection / matched coverage — Views A, B, C of §6.3.1-ter)*, `tables/matched_coverage_comparison.csv` *(PICL evaluated on the exact samples DRM/DTM picked — View D of §6.3.1-ter)*

**Per-seed artifacts (project tree, not in shared bundle):** `results/seeds/seed_<42..46>/results_summary.json`, `scores.npz`, `model_and_meta.pt`

---

## Verification log

A programmatic check parses this report, locates every numeric claim, and compares each one against its source CSV/JSON.

- **175 numeric claims checked**; all match their source files to within rounding.
- **One earlier draft issue caught and fixed before this release:** the report previously wrote "0.252 ± 0.018" for the Test 3 non-cause-over-cause ratio. The source CSV stores only the aggregate ratio (0.252) and the underlying effect-size means/stds; it does not contain a ratio std. Fixed by reporting the constituent means and stds (causal 0.957 ± 0.008, non-causal 0.242 ± 0.016) and the ratio of means (0.252) as a point value.
- **One real codebase inconsistency identified and disclosed in §6.3.1:** `main_performance_table.csv` and `results_summary.json` both label a column "acc_all" but compute predictions by different decision rules (composite-score argmax vs classifier-head argmax). Means agree to 4 decimals; stds differ (0.23 vs 0.12 pp). Both numbers retained; the distinction is now explicit in the report.
- **Cross-checks performed against raw data** (not just self-consistency of the CSVs):
  - Test 1: 11/11 IEC hard edges pass in all 5 seeds (verified by direct count over `counterfactual_test1_cause_activation.csv`).
  - Test 2: aggregate ratio "1.525×" recomputed from per-fault ratios = 1.5247, matches the summary CSV.
  - Test 2: 4 of 6 faults have ratio > 1 (D1, D2, PD, T2 pass; T1, T3 fail) — recomputed directly from the CSV.
  - 5-seed acc_all from JSON files: mean 94.79, std 0.12 — matches the report.
  - Knowledge-alignment edge counts (NOTEARS 18, DCDI* 16, PICL 13) recomputed from `knowledge_alignment.csv` — match.
- **Two ECE numbers in the report, different objects.** The calibration table reports the classifier-head's raw vs temperature-scaled ECE (0.1536 → 0.0268, file `tables/calibration_summary.csv`). The per-seed JSON reports the full pipeline's ECE after temperature scaling (0.0217 ± 0.0067 over 5 seeds). Both are temperature-scaled — they are computed on different scoring objects (classifier head only vs full pipeline output), so they do not need to match. The `HOW_TO_RUN.md` headline "ECE 0.0284" on seed 42 is the pipeline ECE and matches `results/seeds/seed_42/results_summary.json` exactly.
- **Baseline scrutiny (second pass, on user challenge):** the user questioned the DRM (19.70 %), DTM (22.12 %), NOTEARS (66.97 %), DCDI* (65.33 %), SVM (92.73 %) numbers. Investigation produced four supporting CSVs (`rule_based_coverage_breakdown.csv`, `dtm_per_class_diagnosis.csv`, `dag_baseline_fairness.csv`, `discriminative_baseline_hyperparams.csv`) and section §6.3.1-bis. Initial findings (subsequently corrected for DTM — see fifth-pass entry below): (i) DRM 19.70 % is abstention-counted-as-wrong; on its 97 covered samples it scores 67.0 %; (ii) DTM 22.12 % low partly from abstention and partly from a perceived implementation issue (later determined to be primarily a label-convention mismatch — see below); (iii) NOTEARS/DCDI* are unfairly handicapped by allowing gas → gas edges in their search space while HillClimb/GES are constrained to fault → gas only — re-running NOTEARS with the same constraint yields 0.85 instead of 0.67; (iv) SVM 92.73 % is legitimate (RBF SVM, sklearn defaults, no tuning).
- **Coverage-conditional comparison (added in third pass, on user challenge):** the user pointed out that the DRM/DTM headline accuracies are misleading because they count abstentions as errors. Computed three fair views (own coverage, three-way intersection, and PICL matched to DRM/DTM coverage by sweeping its threshold) — full evidence in `tables/coverage_conditional_comparison.csv`. Key results: on the intersection of all three methods' covered samples (87 samples on average), DRM 69.6 %, DTM 25.3 %, PICL 97.7 %. With PICL forced to DRM's 29.4 % coverage, PICL is 100.0 % (all 5 seeds). With PICL forced to DTM's 83.6 % coverage, PICL is 96.9 %.
- **View D (PICL on the exact same samples DRM/DTM picked) added in fourth pass:** complementary to View C. View C reranks samples by PICL's own confidence; View D fixes the sample set to whatever DRM/DTM picked and asks what PICL would have predicted on those specific samples. Results: PICL gets 95.7 ± 0.4 % on DRM's 97 samples (DRM gets 67.0 %), and 94.1 ± 0.3 % on DTM's 276 samples (DTM gets 26.4 %). All four values were independently re-derived from raw scores; the existing `tables/coverage_conditional_comparison.csv` and the newly-created `tables/matched_coverage_comparison.csv` agree on every overlapping number (verified by a programmatic cross-check, 0 mismatches over 12 checks).
- **DTM finding corrected in fifth pass (user challenge: "are you sure DTM is that low?"):** the earlier draft claimed DTM's 26.45 % was driven by an implementation bug (missing `%C₂H₂ < 15` in the T3 rule of `_dtm_predict`). I re-implemented canonical Duval Triangle 1 per Duval & DePablo 2001 / IEC 60599 Annex C and ran it on the same test set. Canonical DTM achieves 30.07 % — only 3.6 pp better than the implemented version. The implementation difference is real but minor. The actual cause of the low score, established by inspecting the test set's per-class Duval percentages (`tables/dataset_vs_duval_zone_mismatch.csv`): **the dataset's fault labels do not correspond to canonical Duval Triangle zones.** Only 4.9 % of PD labels, 5.4 % of T1 labels, and 3.9 % of T3 labels sit in their canonical Duval zone; only D2 has a majority (81 %) match. The dataset was almost certainly labelled by a different procedure than Duval Triangle (direct inspection, IEEE C57.104 ratios, or expert judgment). The corrected explanation now sits in §6.3.1-bis under "DTM at 22.12 % — the real story", and two new CSVs (`dtm_canonical_vs_implemented.csv`, `dataset_vs_duval_zone_mismatch.csv`) provide the supporting evidence.
