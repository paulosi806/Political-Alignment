# Political Speech Partisanship Pipeline

This repository contains the code for a master’s thesis project that uses partisan language in congressional speech to identify partisan corporate speech in the form of tweets. The pipeline follows the general methodological logic of Gentzkow et al. (2019) and estimates phrase-level partisanship using a penalized Poisson model.

The central outputs are a measure of average partisanship by Congress and tables of top partisan phrases. These phrase-level estimates are intended to serve as the basis for detecting and interpreting partisan language in corporate tweets. The repository also includes supporting validation and interpretation outputs, including subsampling confidence intervals, permutation validation, and out-of-sample validation.

---

## 1. Project idea

The core idea is that political parties may differ not only in what they advocate, but also in which phrases they use more frequently. For each phrase, the model estimates whether its usage is more associated with Republican or Democratic speech.

The pipeline works at the speaker-session level. A speaker-session is an observation unit corresponding to a speaker in a specific Congress. For each speaker-session, phrase counts are computed and merged with party information and speaker covariates.

---

## 2. Model overview

The main model is a pooled penalized Poisson model estimated across all included Congresses. For phrase `j`, speaker-session `i`, and Congress `t`, the model can be summarized as:

```text
c_ij ~ Poisson(exp(log(m_i) + alpha_j + baseCov_i' gamma_j + partyInteraction_i' phi_j))
```

where:

- `c_ij`: count of phrase `j` in speaker-session `i`
- `m_i`: exposure, defined as the total number of retained bigrams/phrases in speaker-session `i`
- `alpha_j`: phrase-specific intercept
- `baseCov_i`: design matrix containing Congress fixed effects and static covariates
- `gamma_j`: phrase-specific coefficients for the base covariates
- `partyInteraction_i`: design matrix for `Republican × Congress`
- `phi_j`: Congress-specific party loading for phrase `j`

The current main model therefore does **not** estimate one separate model per Congress. Instead, it estimates one pooled model across all Congresses while allowing the party effect of each phrase to vary by Congress through `Republican × Congress` interactions.

---

## 3. Main data inputs

The pipeline starts from raw congressional speech data and legislator metadata.

The congressional speech input consists of Congressional Record JSON files from GovInfo. These files are expected in the following directory:

```text
data/raw/congress_speeches/
```

The legislator metadata comes from the GitHub repository unitedstates/congress-legislators. The relevant files are expected in:

```text
data/raw/politicians/
```

Typical legislator metadata files include:

```text
data/raw/politicians/legislators-current.json
data/raw/politicians/legislators-historical.json
data/raw/politicians/legislators-social-media.json
data/raw/politicians/executive.json
```

The early pipeline scripts transform these raw inputs into the processed files used by the estimation step:

```text
data/processed/speaker_sessions.parquet
data/processed/phrase_counts_long.parquet
```

### `speaker_sessions.parquet`

This file contains speaker-session metadata. Important columns include:

```text
speaker_session_id
speaker_bioguide
congress
party
state
chamber_member
gender
```

### `phrase_counts_long.parquet`

This file contains phrase counts in long format. Important columns include:

```text
speaker_session_id
phrase
count
congress
```

---

## 4. Main covariates

The main model uses the following covariates and interactions:

```text
Congress fixed effects
state
chamber_member
gender
census_region
party_in_majority
census_region × Congress
Republican × Congress
```

`census_region` is derived from the speaker’s state. `party_in_majority` indicates whether the party of the speaker-session holds the majority in the relevant chamber and Congress.

---

## 5. Typical project structure

A typical project structure is:

```text
project_root/
├── config/
│   └── config.yaml
├── data/
│   └── interim/
│   └── processed/
│   └── raw/
│       ├── congress_speeches/
│       ├── politicians/
│       └── pocedural_language/
├── outputs/
│   └── tables/
│       ├── descriptive/
│       ├── penalized_poisson/
│       └── penalized_poisson_overall/
├── scripts/
│   └── congress_speeches/
│       ├── 00_run_pipeline.py
│       ├── 01_load_raw_speeches.py
│       ├── 02_match_speakers.py
│       ├── 03_preprocess_text.py
│       ├── 04_build_phrase_counts.py
│       ├── 05_descriptive_statistics.py
│       ├── 06_penalized_poisson_partisanship.py
│       ├── 07_inference.py
│       ├── 08_permutation_validation.py
│       ├── 09_out_of_sample_validation.py
│       └── 10_overall_period_partisan_phrases.py
└── src/
    └── political_speech/
        ├── utils.py
        └── utils_analysis.py
```

The exact folder structure may differ slightly depending on the local setup.

---

## 6. Script overview

Before running any of the pipeline scripts, make sure that all Python dependencies listed in requirements.txt are installed in your active environment.

### `00_run_pipeline.py`

Optional driver script for running multiple pipeline steps in sequence.

### `01_load_raw_speeches.py`

Loads the raw congressional speech data.

### `02_match_speakers.py`

Merges speech data with speaker metadata, especially party, chamber, state, and Bioguide ID.

### `03_preprocess_text.py`

Cleans and tokenizes the text. This step creates the basis for bigram counting.

### `04_build_phrase_counts.py`

Creates phrase counts in long format. The typical output is:

```text
processed/phrase_counts_long.parquet
```

### `05_descriptive_statistics.py`

Creates descriptive statistics, such as the number of speaker-sessions, phrase counts, party distributions, and coverage by Congress.

### `06_penalized_poisson_partisanship.py`

Main analysis script. It estimates the pooled penalized Poisson model across all Congresses.

Important outputs:

```text
outputs/tables/penalized_poisson/average_partisanship_penalized_poisson.csv
outputs/tables/penalized_poisson/phrase_parameters_penalized_poisson.parquet
outputs/tables/penalized_poisson/top_partisan_phrases_penalized_poisson.csv
outputs/tables/penalized_poisson/top_partisan_phrases_by_direction_penalized_poisson.csv
```

### `07_inference.py`

Computes subsampling confidence intervals for `average_partisanship`. For each subsampling draw, a pooled model is estimated on a subsample of speaker-sessions. Confidence intervals are then computed by Congress.

### `08_permutation_validation.py`

Runs a permutation validation. Party labels are randomly permuted within each Congress while the language data remain unchanged. If the estimator has little finite-sample bias, the estimated partisanship in the permuted data should be close to 0.5.

### `09_out_of_sample_validation.py`

Runs an out-of-sample validation. Speaker-sessions are assigned to folds. For each fold, the model is estimated on the training data and applied to the held-out test data.

### `10_overall_period_partisan_phrases.py`

Creates top partisan phrases for the full sample period. This script does not need to estimate a new model. Instead, it aggregates the Congress-specific phrase results from script 6.

The aggregation weights each Congress-specific phrase partisanship estimate by the observed frequency of that phrase in the corresponding Congress. In other words, phrases that occur more often in a Congress receive greater weight in the full-period average. The resulting overall phrase partisanship measure is therefore a usage-weighted average of Congress-specific phrase partisanship



---

## 7. Important functions in `utils_analysis.py`

### `add_census_region(...)`

Derives a Census region from the speaker’s state.

### `add_majority_party_indicator(...)`

Creates `party_in_majority`. This variable indicates whether the party of the speaker-session holds the majority in the corresponding chamber and Congress.

### `build_static_covariate_time_varying_party_design(...)`

Builds the design matrices for the new pooled model:

The base matrix contains Congress fixed effects, static covariates, and time-varying interactions such as `census_region × Congress`. The party interaction matrix contains `Republican × Congress`.

### `fit_one_phrase_static_covariates_time_varying_party(...)`

Estimates the penalized Poisson model for a single phrase. The party coefficients are regularized using an L1 penalty.

### `compute_choice_probabilities_static_covariates_time_varying_party(...)`

Computes choice probabilities and partisanship measures by Congress from the estimated phrase parameters.

### `estimate_static_covariate_partisanship_model(...)`

Central wrapper function for the current main model. It handles:

- data filtering
- design matrix construction
- phrase limiting
- phrase-level estimation
- computation of `average_partisanship`
- creation of output tables

This function should be used in the main model, inference, permutation validation, and out-of-sample validation scripts.

### `compute_gentzkow_subsampling_ci(...)`

Computes subsampling confidence intervals from full-sample and subsample estimates.

### `compute_out_of_sample_partisanship_static_for_fold(...)`

Applies training parameters from one fold to held-out test data and computes out-of-sample partisanship by Congress.

---

## 8. Configuration

The main settings are stored in the configuration file, typically `config/config.yaml`.

Example relevant parameters:

```yaml
penalized_poisson:
  lambda_path_steps: 100
  lambda_path_min_ratio: 1e-5
  min_penalty_alpha: 1e-5
  max_phrases_per_congress: null
  top_n_phrases: 50
  top_n_phrases_by_direction: 10
  maxiter: 1000
  majority_party_by_congress_chamber: {}

validation:
  n_permutations: 10
  random_seed: 9

subsampling_inference:
  n_subsamples: 100
  subsample_fraction: 0.1
  max_draw_attempts: 100
  lower_order: 3
  upper_order: 98

out_of_sample_validation:
  n_folds: 5
  random_seed: 9
```

For test runs, use smaller values, for example:

```yaml
penalized_poisson:
  max_phrases_per_congress: 100
  lambda_path_steps: 20

validation:
  n_permutations: 1

subsampling_inference:
  n_subsamples: 3

out_of_sample_validation:
  n_folds: 2
```

---

## 9. Recommended execution order

A typical execution order is:

```bash
python scripts/analysis/01_load_raw_speeches.py
python scripts/analysis/02_match_speakers.py
python scripts/analysis/03_preprocess_text.py
python scripts/analysis/04_build_phrase_counts.py
python scripts/analysis/05_descriptive_statistics.py
python scripts/analysis/06_penalized_poisson_partisanship.py
python scripts/analysis/07_inference.py
python scripts/analysis/08_permutation_validation.py
python scripts/analysis/09_out_of_sample_validation.py
python scripts/analysis/10_overall_period_partisan_phrases.py
```

For quick technical tests, first run only up to script 6. Then test inference and validation with reduced parameters.

---

## 10. Interpretation of the main outputs

### `average_partisanship_penalized_poisson.csv`

Contains `average_partisanship` by Congress. Values above 0.5 indicate greater partisan distinguishability of language. Values close to 0.5 mean that language distinguishes less strongly between parties.

### `phrase_parameters_penalized_poisson.parquet`

Contains phrase results by Congress. Important columns include:

```text
congress
phrase
phi
phrase_partisanship
abs_phrase_partisanship
q_republican_mean
q_democrat_mean
predicted_per_100k_republican
predicted_per_100k_democrat
```

### `raw_phrase_parameters_penalized_poisson.parquet`

Contains the raw estimated parameters, especially `gamma` and `phi_by_congress`. This file is needed for out-of-sample applications and technical diagnostics.

### `top_partisan_phrases_penalized_poisson.csv`

Contains the strongest partisan phrases by Congress, sorted by `abs_phrase_partisanship`.

### `top_partisan_phrases_by_direction_penalized_poisson.csv`

Contains separate top phrase lists for the Republican and Democratic directions.

### `top_partisan_phrases_penalized_poisson_overall.csv`

Contains the top partisan phrases over the full sample period. These are aggregated from the Congress-specific phrase results, typically using `phrase_count` as the weight.

---

## 11. Overall top partisan phrases

No separate model is needed to obtain the top partisan phrases for the full sample period. Instead, the Congress-specific phrase results from the main model are aggregated.

For phrase `j`, compute:

```text
overall_phrase_partisanship_j = sum_t(phrase_partisanship_jt * phrase_count_jt) / sum_t(phrase_count_jt)
```

where:

- `phrase_partisanship_jt`: partisanship of phrase `j` in Congress `t`
- `phrase_count_jt`: count of phrase `j` in Congress `t`

Then rank phrases by the absolute value of the aggregated score:

```text
abs_phrase_partisanship = abs(overall_phrase_partisanship)
```

This approach ranks phrases highly when they consistently point in a partisan direction over the full period.

---

## 12. Runtime notes

The main estimation is computationally intensive because a penalized Poisson model is estimated for many phrases along a lambda path. Runtime increases especially with:

- many phrases
- many lambda steps
- many subsampling draws
- many out-of-sample folds
- many permutations

For initial tests, use small settings:

```yaml
max_phrases_per_congress: 100
lambda_path_steps: 20
n_subsamples: 3
n_permutations: 1
n_folds: 2
```

---

## 13. Reproducibility

Random procedures are controlled by `random_seed`. This applies especially to:

- subsampling inference
- permutation validation
- out-of-sample fold assignment

With the same seed, random-dependent results should be reproducible.

---

## 14. Key methodological choices

### Pooled model instead of separate Congress models

The main model estimates all Congresses jointly and allows Congress-specific party effects through `Republican × Congress`. This uses information from the full sample while still allowing temporal variation in party language.

### Phrase limiting by Congress

If `max_phrases_per_congress` is set, the most frequent phrases are selected separately within each Congress. This is useful for test runs and runtime control.

### Limited search for Lamba
To reduce computation time during model estimation, the search over the regularization path is stopped early if the model selection criterion does not improve for several consecutive lambda values. This avoids evaluating the full lambda path in cases where additional penalty values no longer lead to a better selected model.


### Overall phrase ranking from main-model results

For top phrases over the full sample period, no separate overall model is estimated. Instead, the Congress-specific results from the main model are aggregated. This keeps the interpretation consistent with the main specification and saves computation time.

---

## 15. Common issues

### Data issues

This project is explicitly written for congressional speech data from the 2015–2026 sample period, corresponding to the Congresses covered by the date ranges defined in utils.py. If the raw data contains speeches outside these Congress/session date ranges, the pipeline may raise an error when assigning speeches to Congresses and sessions.

If this happens, check whether the input data include dates outside the intended sample period. Either remove these observations before running the phrase-count construction step or extend the Congress/session date ranges in utils.py accordingly.

### Missing covariates

If columns such as `state`, `gender`, `chamber_member`, or `party` are missing, check the upstream preprocessing and speaker-matching scripts.

### Only one party in a Congress

The model requires both Democrats and Republicans in every included Congress. If a Congress contains only one party, estimation for that data subset becomes problematic.

### Very large test runs

With many phrases and many lambda steps, estimation can take a long time. Use small test parameters for debugging.

---

## 16. Short summary

The main analysis is performed in script 6. Scripts 7 to 9 validate the results. Script 10 creates aggregated top partisan phrases for the full sample period from the main-model outputs.

The central workflow is:

```text
Script 6: pooled model across all Congresses
Script 7: subsampling confidence intervals
Script 8: permutation validation
Script 9: out-of-sample validation
Script 10: overall top partisan phrases from script-6 outputs
```
