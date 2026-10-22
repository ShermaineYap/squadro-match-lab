# BWF champion model card — Symmetric Glicko Logistic Regression, uncalibrated

| | |
| --- | --- |
| Production version | `3.0-symmetric-glicko-lr-uncalibrated` |
| Internal audited-bundle version | `3.0-candidate-symmetric-uncalibrated` |
| Display name | BWF Symmetric Glicko Logistic Regression, uncalibrated |
| Bundle | `models/champion/bwf_champion_bundle.joblib` |
| SHA-256 | `e7a3cc069d0d7897d692fdf1cc276f1cec6d3f953e02ca11e152665f2fc97d39` (unchanged by promotion) |
| Prediction function | `symmetric_margin_averaging.v1` |
| Decision | Project author, 21 August 2026 (`docs/BWF_CHAMPION_DECISION_RECORD.md`). Not supervisor approval. |
| Baseline retained | XGBoost (`models/baseline/bwf_baseline_bundle.joblib`), unchanged |

The two version strings are both correct and are not interchangeable. The
production version is what the service reports and what every response carries.
The internal version is the string written inside the bundle file, which was
written before promotion and was deliberately **not** edited: rewriting it would
change the file's SHA-256, and that hash is the only thing tying the served
artefact to the audit that measured it. Anything user-facing shows the
production version; the internal one appears only in technical detail, beside
the hash it protects.

## What it does

Estimates the probability that team 1 beats team 2 in a match that has not been
played. Inputs are pre-match only: ratings, form, experience, rest, head-to-head
counts and competition context.

## How the probability is formed

    m_sym(A, B) = [ m(A, B) - m(B, A) ] / 2
    p(A beats B) = sigmoid( m_sym(A, B) )

Scoring both orientations and averaging in log-odds makes the model exactly
antisymmetric, so `P(A beats B) = 1 - P(B beats A)` holds to floating-point
precision. This is a correctness property, not a tuning choice. The baseline
lacked it: about 3% of fixtures named a different winner depending on which
player was entered first, and `P(A) + P(B)` ranged 0.88-1.08. Measured violation
for the champion across both audit seasons is below 3e-16.

## Inputs

Twenty numeric and five categorical features. Fifteen Elo-era columns
(`individual_elo_diff`, form, experience, career win rate, rest, head-to-head,
cold-start flags), five Glicko-2 columns
(`glicko_certainty_weighted_rating_diff`, `glicko_deviation_diff`,
`glicko_volatility_diff`, `glicko_volatility_mean`, `glicko_deviation_mean`),
and `discipline`, `tier_category`, `round`, `is_qualification`, `host_country`.

Glicko-2 system constant tau = 0.5. Logistic regression `C = 0.1172102298`,
selected on 2022-2023 development folds before any 2024 access; lineage recorded
in `outputs/research/bwf_champion_lock/c_value_lineage_addendum.json`.

## Training

Fitted on the complete 2022-2024 development period, 17,193 matches.
Uncalibrated. Features are built date-batched: every match on a date is rated
from state that predates the date entirely, so two matches on the same day
cannot inform one another.

## Results

> **Audit integrity.** The figures below are reproducible: both prediction files
> and both subgroup files match their recorded hashes, and every metric in this
> table recomputes from them to within 1e-9. Two *metadata* files do not match
> their recorded hashes — the pre-registered protocol and the audit result
> record — so the pre-registration seal can no longer be demonstrated
> cryptographically. This is disclosed in full, with every hash, in
> `docs/BWF_AUDIT_INTEGRITY_CORRECTION.md`. Read it before citing this table as
> pre-registered evidence.

| | 2025 corrected audit | 2026 temporal audit |
| --- | --- | --- |
| Matches / tournaments | 7,030 / 40 | 3,822 / 22 |
| **ROC-AUC** | **0.8084** | **0.7977** |
| 95% tournament-cluster interval | [0.7946, 0.8219] | [0.7823, 0.8136] |
| PR-AUC | 0.8140 | 0.7999 |
| Accuracy / balanced accuracy | 0.7289 / 0.7288 | 0.7174 / 0.7172 |
| F1 | 0.7337 | 0.7236 |
| Log loss | 0.5298 | 0.5481 |
| Brier | 0.1788 | 0.1849 |
| ECE | 0.0142 | 0.0382 |
| Improvement over the baseline | +0.0108 | +0.0127 |

**The 0.80 criterion was met on 2025 and not on 2026. Performance above 0.80 was
not consistently replicated across seasons, and must not be described as if it
were.** Both intervals contain 0.80, so the data are compatible with true
performance below the criterion in either season. The improvement over the baseline is
consistent in direction across development CV (+0.0133), 2024 validation
(+0.0115) and both audit seasons; that repetition, not any single interval, is
what supports it.

## Limitations

- **Men's Singles is the weakest discipline** — 0.740 in 2025, 0.720 in 2026,
  against Women's Doubles at 0.877 and 0.846. A single headline figure overstates
  reliability for MS and understates it for WD.
- **Qualification matches** score 0.726 in 2025 and 0.773 in 2026.
- **The 2025 World Tour Finals subgroup is 72 matches with ECE 0.160.** Too small
  to quote without saying so.
- **2026 calibration is materially worse than 2025** (ECE 0.038 against 0.014).
- **Uncertainty: four separate claims, routinely collapsed into one.** They are
  stated apart because they are not the same statement.
  - *No fitted calibrator.* Probabilities are served exactly as audited; no
    isotonic or Platt calibration is applied.
  - *No conformal prediction set.* The baseline's conformal thresholds were
    calibrated on the XGBoost model over a different feature set. They do not
    describe the champion
    and are not reused.
  - *No validated operating point.* Nothing licenses the interface to accept or
    decline a champion call. Every fixture returns a probability. The claim that both
    model streams can decline to commit was true of the baseline and is not true
    of the champion.
  - *Descriptive calibration evidence does exist.* The locked audit measured
    Brier 0.1788 / ECE 0.0142 on 2025 and Brier 0.1849 / ECE 0.0382 on 2026,
    with per-segment figures in the audit subgroup tables. Those are real
    measurements of the probabilities in aggregate. "No calibration evidence"
    understates what is known; quoting the baseline's conformal coverage overstates it.
    They are descriptions, not thresholds: deriving an accept/decline rule from
    them would be fitting on the audit.
- **Rating uncertainty is not new information.** The Glicko deviation columns
  correlate 0.67-0.72 with the existing cold-start and experience features; the
  gain is a finer, decaying encoding of something the model partly had.
- **Ratings are per player per discipline**, so a player moving between singles
  and doubles starts each discipline afresh.
- **Serving state is a snapshot.** No speculative rating-deviation decay is
  applied between the snapshot and a future fixture, so deviations understate
  present uncertainty by an amount that grows with state age. Age is disclosed
  with every prediction and dates on or before the snapshot are refused.
- **The Stage 1 hyperparameter search sampled 13 of 30 grid points.** One omitted
  value was later scored and came out ~0.0002 lower; the other 16 are unevaluated,
  so the cost of the incomplete search is unknown.

## Governance

- Selected on development evidence. The audit seasons were read once, after the
  bundle was locked and hashed, and chose nothing.
- **The 2024 partition was accessed twice.** The first run built the wrong model
  through a harness defect and was invalid; the corrected run was the first
  genuine evaluation. Selection was closed before either. Recorded in
  `outputs/research/bwf_champion_stage1/stage1_governance_addendum.json`.
- No further candidate selection, retuning, recalibration or audit re-reading is
  permitted.

## Explanations

Contributions are computed for both orientations and combined:

    phi_i_sym = [ phi_i(A, B) - phi_sigma(i)(B, A) ] / 2

`sigma` exchanges the side-specific columns only. Contributions are **log-odds**,
sum exactly to the symmetric margin, and their sigmoid equals the displayed
probability to within 1e-9. The base value is exactly zero — before evidence, a
contest between two sides is a coin toss.

**These are not SHAP values and are not to be described as SHAP.** The baseline was
a tree model and needed SHAP sampled against a background population. The
champion is linear in
its transformed inputs, so a feature's share of the margin is the product of its
coefficient and its value: an exact symmetric log-odds contribution, computed
rather than estimated. That statement is made here, in the technical record, and
in the `/bwf/explain` payload's `attribution_note`.

**What the interface states, exactly.** The match panel prints one sentence
beneath the factor bars: *"Bar length ranks each factor against the largest
contribution shown, it is not a percentage share of the win probability."* That
is the scale claim, and it is there because the bars are drawn directly beneath
a percentage, where a bar reads as a slice of the number above it.

Each factor row additionally carries the limit declared for that feature in
`rules/bwf_factor_phrasing.json` — its "does not mean" clause — so what a
particular factor cannot be taken to mean is stated per factor rather than once
for all of them. A render test pins both, and a second test holds the layout to
the sentence by failing if a percentage is ever printed beside a bar.

The panel does **not** use the words "exact" or "computed rather than
estimated", and it does not mention SHAP — an earlier version of this card
claimed it did. Those are properties of the method, stated above and in the
payload's `attribution_note`, not claims the interface makes.

**Factor wording is declared, not written in code.** Every one of the 25
features has an entry in `rules/bwf_factor_phrasing.json` giving its label, its
sentence template, what it means and what it must not be taken to mean. The
explanation service reads that file and has no wording of its own to fall back
on; `scripts/check_performance_endpoints.py` compares the bundle's feature list
against the file's keys, so a feature added without an entry fails a check rather
than reaching a reader as a column name. Sentences name the athletes or the
partnership, falling back to "Team 1"/"Team 2" only where an id cannot be
resolved to exactly one name.

Declared review prompts (`rules/bwf_rules.json`) are selected by these exact
contributions: a prompt appears only where the feature has a non-negligible
contribution to the probability being explained, in the direction that
probability favours, and only where a rule for it has been written down. It is a
rule-based recommendation layer driven by model attribution — not a causal
claim, and not generated text.

**The match panel does not display them.** `/bwf/explain` still returns
`review_areas`, the rule engine still selects them by exact contribution, and
they are still written to the audit record with every explanation — but as of
v12.6 the dashboard's match panel does not draw them. The prompts are written
against a feature rather than against a fixture, so each one restates the factor
that raised it. Any figure or claim in this card about the prompt layer is a
claim about the API and the record, not about what a coach currently sees.
