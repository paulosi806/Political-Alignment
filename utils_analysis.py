#Utils package for statistical analysis of speeches

from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import json

from scipy.optimize import minimize
from scipy.special import logsumexp

def print_section(title: str) -> None:
    """
    Print a formatted section title.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def save_table(df: pd.DataFrame, path: Path) -> None:
    """
    Save a DataFrame as CSV and create parent folders if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print("Saved:", path)


def make_overall_summary(
    speaker_sessions: pd.DataFrame,
    vocabulary: pd.DataFrame,
    phrase_counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one-row summary of the phrase-count dataset.
    """
    if "keep_phrase" in vocabulary.columns:
        n_kept_phrases = int(vocabulary["keep_phrase"].sum())
    else:
        n_kept_phrases = None

    return pd.DataFrame([
        {
            "n_speaker_sessions": speaker_sessions["speaker_session_id"].nunique(),
            "n_speakers": speaker_sessions["speaker_bioguide"].nunique(),
            "n_congresses": speaker_sessions["congress"].nunique(),
            "n_unique_bigrams_in_vocabulary": vocabulary["phrase"].nunique(),
            "n_kept_bigrams_after_filters": n_kept_phrases,
            "kept_bigrams_total_spoken_count": phrase_counts["count"].sum(),
        }
    ])


def summarize_speaker_sessions_by_congress(
    speaker_sessions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize speaker-sessions by Congress.
    """
    return (
        speaker_sessions
        .groupby("congress", as_index=False)
        .agg(
            n_speaker_sessions=("speaker_session_id", "nunique"),
            n_speakers=("speaker_bioguide", "nunique"),
            total_speeches=("n_speeches", "sum"),
            total_tokens_clean=("n_tokens_clean", "sum"),
            avg_tokens_per_speaker_session=("n_tokens_clean", "mean"),
            median_tokens_per_speaker_session=("n_tokens_clean", "median"),
        )
        .sort_values("congress")
    )


def summarize_speaker_sessions_by_congress_party(
    speaker_sessions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize speaker-sessions by Congress and party.
    """
    return (
        speaker_sessions
        .groupby(["congress", "party"], as_index=False)
        .agg(
            n_speaker_sessions=("speaker_session_id", "nunique"),
            n_speakers=("speaker_bioguide", "nunique"),
            total_speeches=("n_speeches", "sum"),
            total_tokens_clean=("n_tokens_clean", "sum"),
        )
        .sort_values(["congress", "party"])
    )


def summarize_phrase_counts_by_congress(
    phrase_counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize phrase counts by Congress.
    """
    return (
        phrase_counts
        .groupby("congress", as_index=False)
        .agg(
            total_bigram_count=("count", "sum"),
            n_unique_kept_bigrams=("phrase", "nunique"),
            n_speaker_sessions=("speaker_session_id", "nunique"),
        )
        .sort_values("congress")
    )


def summarize_phrase_counts_by_congress_party(
    phrase_counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize phrase counts by Congress and party.
    """
    return (
        phrase_counts
        .groupby(["congress", "party"], as_index=False)
        .agg(
            total_bigram_count=("count", "sum"),
            n_unique_kept_bigrams=("phrase", "nunique"),
            n_speaker_sessions=("speaker_session_id", "nunique"),
        )
        .sort_values(["congress", "party"])
    )


def get_top_phrases_overall(
    phrase_counts: pd.DataFrame,
    n: int = 100,
) -> pd.DataFrame:
    """
    Return top n phrases by total count.
    """
    return (
        phrase_counts
        .groupby("phrase", as_index=False)
        .agg(
            total_count=("count", "sum"),
            n_speaker_sessions=("speaker_session_id", "nunique"),
            n_congresses=("congress", "nunique"),
        )
        .sort_values("total_count", ascending=False)
        .head(n)
    )


def get_top_phrases_by_party(
    phrase_counts: pd.DataFrame,
    n: int = 100,
) -> pd.DataFrame:
    """
    Return top n phrases within each party by total count.
    """
    return (
        phrase_counts
        .groupby(["party", "phrase"], as_index=False)
        .agg(
            total_count=("count", "sum"),
            n_speaker_sessions=("speaker_session_id", "nunique"),
            n_congresses=("congress", "nunique"),
        )
        .sort_values(["party", "total_count"], ascending=[True, False])
        .groupby("party", as_index=False)
        .head(n)
    )


def summarize_vocabulary_filter(
    vocabulary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize how many phrases were kept or dropped by vocabulary filters.
    """
    if "keep_phrase" not in vocabulary.columns:
        return pd.DataFrame([
            {
                "n_bigrams_total": len(vocabulary),
                "n_bigrams_kept": None,
                "n_bigrams_dropped": None,
                "share_kept": None,
                "min_total_count_kept": None,
                "median_total_count_kept": None,
                "max_total_count_kept": None,
            }
        ])

    kept = vocabulary["keep_phrase"]

    return pd.DataFrame([
        {
            "n_bigrams_total": len(vocabulary),
            "n_bigrams_kept": int(kept.sum()),
            "n_bigrams_dropped": int((~kept).sum()),
            "share_kept": kept.mean(),
            "min_total_count_kept": (
                vocabulary.loc[kept, "total_count"].min()
                if kept.any()
                else None
            ),
            "mean_total_count_kept": (
                vocabulary.loc[kept, "total_count"].mean()
                if kept.any()
                else None
            ),
            "median_total_count_kept": (
                vocabulary.loc[kept, "total_count"].median()
                if kept.any()
                else None
            ),
            "max_total_count_kept": (
                vocabulary.loc[kept, "total_count"].max()
                if kept.any()
                else None
            ),
        }
    ])


def check_congress_party_coverage(
    phrase_counts_by_congress_party: pd.DataFrame,
) -> pd.DataFrame:
    """
    Check whether each Congress has phrase counts for both Democrats and Republicans.
    """
    check = (
        phrase_counts_by_congress_party
        .pivot_table(
            index="congress",
            columns="party",
            values="total_bigram_count",
            fill_value=0,
        )
        .reset_index()
    )

    for party in ["Democrat", "Republican"]:
        if party not in check.columns:
            check[party] = 0

    check["has_democrat_counts"] = check["Democrat"] > 0
    check["has_republican_counts"] = check["Republican"] > 0

    check["has_both_parties"] = (
        check["has_democrat_counts"]
        & check["has_republican_counts"]
    )

    return check


def add_census_region(
    df: pd.DataFrame,
    state_col: str = "state",
) -> pd.DataFrame:
    """
    Add Census region based on two-letter state abbreviation.

    Regions follow the standard U.S. Census grouping:
    - Northeast
    - Midwest
    - South
    - West
    """
    df = df.copy()

    state_to_region = {
        # Northeast
        "CT": "Northeast",
        "ME": "Northeast",
        "MA": "Northeast",
        "NH": "Northeast",
        "RI": "Northeast",
        "VT": "Northeast",
        "NJ": "Northeast",
        "NY": "Northeast",
        "PA": "Northeast",

        # Midwest
        "IL": "Midwest",
        "IN": "Midwest",
        "MI": "Midwest",
        "OH": "Midwest",
        "WI": "Midwest",
        "IA": "Midwest",
        "KS": "Midwest",
        "MN": "Midwest",
        "MO": "Midwest",
        "NE": "Midwest",
        "ND": "Midwest",
        "SD": "Midwest",

        # South
        "DE": "South",
        "DC": "South",
        "FL": "South",
        "GA": "South",
        "MD": "South",
        "NC": "South",
        "SC": "South",
        "VA": "South",
        "WV": "South",
        "AL": "South",
        "KY": "South",
        "MS": "South",
        "TN": "South",
        "AR": "South",
        "LA": "South",
        "OK": "South",
        "TX": "South",

        # West
        "AZ": "West",
        "CO": "West",
        "ID": "West",
        "MT": "West",
        "NV": "West",
        "NM": "West",
        "UT": "West",
        "WY": "West",
        "AK": "West",
        "CA": "West",
        "HI": "West",
        "OR": "West",
        "WA": "West",
    }

    df[state_col] = df[state_col].astype("string").str.strip()
    df["census_region"] = df[state_col].map(state_to_region)

    df["census_region"] = (
        df["census_region"]
        .astype("string")
        .fillna("Unknown")
    )

    return df

def add_majority_party_indicator(
    df: pd.DataFrame,
    majority_party_by_congress_chamber: dict | None = None,
    congress_col: str = "congress",
    chamber_col: str = "chamber_member",
    party_col: str = "party",
) -> pd.DataFrame:
    """
    Add majority-party indicator based only on an explicit config mapping.

    Expected config format:
    {
        "115": {"House": "Republican", "Senate": "Republican"},
        "116": {"House": "Democrat", "Senate": "Republican"},
        ...
    }

    The function does not infer majority party from the data.
    If a Congress × Chamber combination is missing from the mapping,
    it raises an error.
    """
    df = df.copy()

    if majority_party_by_congress_chamber is None:
        majority_party_by_congress_chamber = {}

    if not majority_party_by_congress_chamber:
        raise ValueError(
            "majority_party_by_congress_chamber is empty. "
            "Please define it explicitly in config.yaml."
        )

    df[chamber_col] = df[chamber_col].astype("string").str.strip()
    df[party_col] = df[party_col].astype("string").str.strip()

    def lookup_majority_party(row):
        congress_key = str(int(row[congress_col]))
        chamber = str(row[chamber_col])

        if congress_key not in majority_party_by_congress_chamber:
            return pd.NA

        chamber_map = majority_party_by_congress_chamber[congress_key]

        if chamber not in chamber_map:
            return pd.NA

        return chamber_map[chamber]

    df["majority_party"] = df.apply(
        lookup_majority_party,
        axis=1,
    )

    missing_majority = df["majority_party"].isna()

    if missing_majority.any():
        missing_combinations = (
            df.loc[missing_majority, [congress_col, chamber_col]]
            .drop_duplicates()
            .sort_values([congress_col, chamber_col])
        )

        raise ValueError(
            "Missing majority-party mapping for the following "
            "Congress × Chamber combinations:\n"
            f"{missing_combinations.to_string(index=False)}\n\n"
            "Please add these combinations to config.yaml under "
            "'penalized_poisson.majority_party_by_congress_chamber'."
        )

    df["party_in_majority"] = (
        df[party_col] == df["majority_party"]
    ).astype(int)

    return df

def build_static_covariate_time_varying_party_design(
    speaker_data: pd.DataFrame,
    static_categorical_cols: list[str] | None = None,
    time_varying_categorical_cols: list[str] | None = None,
    static_binary_cols: list[str] | None = None,
    congress_col: str = "congress",
    republican_col: str = "republican",
    base_covariate_columns: list[str] | None = None,
    party_interaction_columns: list[str] | None = None,
    congress_values: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[int], np.ndarray]:
    """
    Build design matrices for a Gentzkow-style model with mostly static
    covariate coefficients, time-varying Census-region effects, and
    time-varying party loadings.

    Model for phrase j:

        c_ij ~ Poisson(exp(log(m_i)
                         + alpha_jt
                         + X_i_static gamma_j
                         + W_it delta_jt
                         + phi_jt * Republican_i))

    Default structure:
    - state, chamber_member, gender are static categorical covariates.
    - party_in_majority is a static binary covariate.
    - census_region is interacted with Congress.
    - Republican is interacted with Congress.

    Parameters
    ----------
    speaker_data:
        Speaker-session-level data. Must contain congress, republican,
        and all requested covariates.

    base_covariate_columns:
        Optional column list used to align a new design matrix to a training
        design matrix. This is needed for out-of-sample validation.

    party_interaction_columns:
        Optional column list used to align Republican × Congress columns to
        a training design matrix.

    congress_values:
        Optional list of Congress values from the training sample. This is
        needed when constructing a test design matrix.

    Returns
    -------
    base_covariate_matrix:
        Matrix containing Congress fixed effects, static covariates, and
        time-varying Census-region effects. The intercept is not included.

    party_interaction_matrix:
        Matrix containing Republican × Congress columns.

    base_covariate_columns:
        Names of columns in base_covariate_matrix.

    party_interaction_columns:
        Names of columns in party_interaction_matrix.

    congress_values:
        Sorted Congress values used by the party interaction block.

    speaker_congress_index:
        For each speaker-session row, the index of its Congress in
        congress_values.
    """
    speaker_data = speaker_data.copy()

    if static_categorical_cols is None:
        static_categorical_cols = [
            "state",
            "chamber_member",
            "gender",
        ]

    if time_varying_categorical_cols is None:
        time_varying_categorical_cols = [
            "census_region",
        ]

    if static_binary_cols is None:
        static_binary_cols = [
            "party_in_majority",
        ]

    required_cols = (
        [congress_col, republican_col]
        + static_categorical_cols
        + time_varying_categorical_cols
        + static_binary_cols
    )

    missing_cols = [
        col for col in required_cols
        if col not in speaker_data.columns
    ]

    if missing_cols:
        raise KeyError(
            "Missing required columns for static-covariate design: "
            f"{missing_cols}"
        )

    speaker_data[congress_col] = speaker_data[congress_col].astype(int)

    if congress_values is None:
        congress_values = sorted(
            speaker_data[congress_col]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
    else:
        congress_values = [int(x) for x in congress_values]

    congress_to_index = {
        congress: idx
        for idx, congress in enumerate(congress_values)
    }

    missing_congress_values = sorted(
        set(speaker_data[congress_col].astype(int).unique())
        - set(congress_values)
    )

    if missing_congress_values:
        raise ValueError(
            "speaker_data contains Congress values that are not in "
            f"congress_values: {missing_congress_values}"
        )

    speaker_congress_index = (
        speaker_data[congress_col]
        .astype(int)
        .map(congress_to_index)
        .to_numpy(dtype=int)
    )

    # ------------------------------------------------------------
    # Congress fixed effects for alpha_jt
    # ------------------------------------------------------------
    congress_dummy_data = {}

    for congress in congress_values[1:]:
        dummy_col = f"congress_{congress}"

        congress_dummy_data[dummy_col] = (
            speaker_data[congress_col].astype(int) == int(congress)
        ).astype(float)

    congress_dummies = pd.DataFrame(
        congress_dummy_data,
        index=speaker_data.index,
    )

    is_alignment_mode = base_covariate_columns is not None

    # ------------------------------------------------------------
    # Static categorical covariates
    # ------------------------------------------------------------
    static_categorical_data = speaker_data[static_categorical_cols].copy()

    for col in static_categorical_cols:
        if static_categorical_data[col].isna().any():
            n_missing = int(static_categorical_data[col].isna().sum())

            raise ValueError(
                f"Column '{col}' contains {n_missing} missing values."
            )

        static_categorical_data[col] = (
            static_categorical_data[col]
            .astype("string")
            .str.strip()
        )

        if (static_categorical_data[col] == "").any():
            n_empty = int((static_categorical_data[col] == "").sum())

            raise ValueError(
                f"Column '{col}' contains {n_empty} empty string values."
            )

    static_categorical_dummies = pd.get_dummies(
        static_categorical_data,
        columns=static_categorical_cols,
        drop_first=not is_alignment_mode,
        dtype=float,
    )

    # ------------------------------------------------------------
    # Time-varying categorical covariates
    # ------------------------------------------------------------
    # For census_region, this creates columns such as:
    # census_region_x_congress_116_South
    # census_region_x_congress_117_South
    #
    # One category is omitted within each Congress to avoid collinearity
    # with Congress fixed effects.
    time_varying_parts = []

    for col in time_varying_categorical_cols:
        col_data = speaker_data[col].copy()

        if col_data.isna().any():
            n_missing = int(col_data.isna().sum())

            raise ValueError(
                f"Column '{col}' contains {n_missing} missing values."
            )

        col_data = (
            col_data
            .astype("string")
            .str.strip()
        )

        if (col_data == "").any():
            n_empty = int((col_data == "").sum())

            raise ValueError(
                f"Column '{col}' contains {n_empty} empty string values."
            )

        category_dummies = pd.get_dummies(
            col_data,
            prefix=col,
            drop_first= not is_alignment_mode,
            dtype=float,
        )

        if category_dummies.empty:
            continue

        # Baseline region effects in the omitted baseline Congress.
        time_varying_parts.append(category_dummies.copy())

        interaction_data = {}

        for congress_dummy_col in congress_dummies.columns:
            for category_col in category_dummies.columns:
                interaction_col = (
                    f"{category_col}_x_{congress_dummy_col}"
                )

                interaction_data[interaction_col] = (
                    category_dummies[category_col]
                    * congress_dummies[congress_dummy_col]
                )

        if interaction_data:
            time_varying_parts.append(
                pd.DataFrame(
                    interaction_data,
                    index=speaker_data.index,
                )
            )

    if time_varying_parts:
        time_varying_df = pd.concat(
            time_varying_parts,
            axis=1,
        )
    else:
        time_varying_df = pd.DataFrame(
            index=speaker_data.index,
        )
    # ------------------------------------------------------------
    # Static binary covariates
    # ------------------------------------------------------------
    static_binary_data = speaker_data[static_binary_cols].copy()

    for col in static_binary_cols:
        if static_binary_data[col].isna().any():
            n_missing = int(static_binary_data[col].isna().sum())

            raise ValueError(
                f"Column '{col}' contains {n_missing} missing values."
            )

        static_binary_data[col] = pd.to_numeric(
            static_binary_data[col],
            errors="coerce",
        )

        if static_binary_data[col].isna().any():
            n_invalid = int(static_binary_data[col].isna().sum())

            raise ValueError(
                f"Column '{col}' contains {n_invalid} non-numeric values."
            )

        invalid_values = ~static_binary_data[col].isin([0, 1])

        if invalid_values.any():
            invalid_examples = (
                static_binary_data.loc[invalid_values, col]
                .drop_duplicates()
                .head(10)
                .tolist()
            )

            raise ValueError(
                f"Column '{col}' contains values other than 0 and 1. "
                f"Examples: {invalid_examples}"
            )

        static_binary_data[col] = static_binary_data[col].astype(float)

    base_df = pd.concat(
        [
            congress_dummies,
            static_categorical_dummies,
            time_varying_df,
            static_binary_data,
        ],
        axis=1,
    )

    if base_covariate_columns is not None:
        base_df = base_df.reindex(
            columns=base_covariate_columns,
            fill_value=0.0,
        )

    base_covariate_columns = base_df.columns.tolist()

    # ------------------------------------------------------------
    # Time-varying party loading columns: Republican × Congress
    # ------------------------------------------------------------
    republican = pd.to_numeric(
        speaker_data[republican_col],
        errors="coerce",
    )

    if republican.isna().any():
        raise ValueError(
            f"Column '{republican_col}' contains non-numeric values."
        )

    if (~republican.isin([0, 1])).any():
        raise ValueError(
            f"Column '{republican_col}' must contain only 0/1 values."
        )

    party_interaction_data = {}

    for congress in congress_values:
        col = f"republican_x_congress_{congress}"

        party_interaction_data[col] = (
            republican.astype(float)
            * (speaker_data[congress_col].astype(int) == int(congress)).astype(float)
        )

    party_interaction_df = pd.DataFrame(
        party_interaction_data,
        index=speaker_data.index,
    )

    if party_interaction_columns is not None:
        party_interaction_df = party_interaction_df.reindex(
            columns=party_interaction_columns,
            fill_value=0.0,
        )

    party_interaction_columns = party_interaction_df.columns.tolist()

    return (
        base_df.to_numpy(dtype=float),
        party_interaction_df.to_numpy(dtype=float),
        base_covariate_columns,
        party_interaction_columns,
        congress_values,
        speaker_congress_index,
    )

def _soft_abs(
    x: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Smooth approximation to absolute value for stable optimization.

    We use this for the tiny numerical penalty on alpha and gamma.
    """
    return np.sqrt(x * x + eps)

def fit_one_phrase_static_covariates_time_varying_party(
    counts: np.ndarray,
    exposures: np.ndarray,
    base_covariate_matrix: np.ndarray,
    party_interaction_matrix: np.ndarray,
    lambda_path_steps: int = 100,
    lambda_path_min_ratio: float = 1e-5,
    min_penalty_alpha: float = 1e-5,
    l1_smooth_eps: float = 1e-8,
    maxiter: int = 1000,
) -> dict:
    """
    Estimate one phrase-specific Poisson model with static covariates,
    time-varying Census-region controls, and time-varying party loadings.

    The model is:

        c_i ~ Poisson(exp(log(m_i)
                         + alpha
                         + base_covariates_i beta
                         + party_interactions_i phi))

    where:
    - base_covariates includes Congress fixed effects, static covariates,
      and Census-region × Congress interactions.
    - party_interactions contains Republican × Congress columns.
    - phi is L1-penalized.
    - the intercept and base covariates receive only a tiny numerical penalty.
    """
    counts = np.asarray(counts, dtype=float)
    exposures = np.asarray(exposures, dtype=float)

    base_covariate_matrix = np.asarray(
        base_covariate_matrix,
        dtype=float,
    )

    party_interaction_matrix = np.asarray(
        party_interaction_matrix,
        dtype=float,
    )

    if len(counts) != len(exposures):
        raise ValueError("counts and exposures must have the same length.")

    if base_covariate_matrix.shape[0] != len(counts):
        raise ValueError(
            "base_covariate_matrix must have one row per speaker-session."
        )

    if party_interaction_matrix.shape[0] != len(counts):
        raise ValueError(
            "party_interaction_matrix must have one row per speaker-session."
        )

    if np.any(exposures <= 0):
        raise ValueError("All exposures must be positive.")

    if counts.sum() <= 0:
        raise ValueError("This phrase has zero total count.")

    n_obs = len(counts)
    log_exposure = np.log(exposures)

    intercept = np.ones((n_obs, 1), dtype=float)

    restricted_design = np.hstack([
        intercept,
        base_covariate_matrix,
    ])

    full_design = np.hstack([
        restricted_design,
        party_interaction_matrix,
    ])

    n_base_params = restricted_design.shape[1]
    n_party_params = party_interaction_matrix.shape[1]

    phrase_rate = counts.sum() / exposures.sum()

    beta_restricted_start = np.zeros(
        restricted_design.shape[1],
        dtype=float,
    )

    beta_restricted_start[0] = np.log(phrase_rate)

    def restricted_objective_and_gradient(
        beta: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        eta = log_exposure + restricted_design @ beta
        mu = np.exp(eta)
        residual = mu - counts

        nll = float(np.sum(mu - counts * eta))
        grad = restricted_design.T @ residual

        beta_abs = _soft_abs(beta, eps=l1_smooth_eps)

        penalty = min_penalty_alpha * np.sum(beta_abs)

        grad_penalty = (
            min_penalty_alpha
            * beta
            / beta_abs
        )

        objective_value = nll + penalty
        grad= grad + grad_penalty

        return objective_value, grad

    restricted_fit = minimize(
        fun=restricted_objective_and_gradient,
        x0=beta_restricted_start,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": maxiter,
        },
    )

    beta_restricted_hat = restricted_fit.x

    if not restricted_fit.success:
        print("Warning: restricted model did not converge.")
        print("Restricted message:", restricted_fit.message)

    eta_restricted = (
        log_exposure
        + restricted_design @ beta_restricted_hat
    )

    mu_restricted = np.exp(eta_restricted)

    # Gradient of the unpenalized negative log likelihood with respect
    # to each Republican × Congress coefficient at phi = 0.
    score_phi_at_zero = (
        party_interaction_matrix.T
        @ (mu_restricted - counts)
    )

    lambda_max = float(
        np.max(np.abs(score_phi_at_zero))
    )

    lambda_path = build_lambda_path_from_lambda_max(
        lambda_max=lambda_max,
        lambda_path_steps=lambda_path_steps,
        lambda_path_min_ratio=lambda_path_min_ratio,
    )

    if lambda_path[-1] > 0:
        lambda_values = np.append(lambda_path, 0.0)
    else:
        lambda_values = lambda_path

    beta_full_start = np.zeros(
        full_design.shape[1],
        dtype=float,
    )

    beta_full_start[:n_base_params] = beta_restricted_hat

    best_result = None
    current_start = beta_full_start.copy()

    def compute_unpenalized_nll(
        beta: np.ndarray,
    ) -> float:
        eta = log_exposure + full_design @ beta
        mu = np.exp(eta)

        return float(np.sum(mu - counts * eta))

    def full_objective_and_gradient(
        beta: np.ndarray,
        lambda_phi: float,
    ) -> tuple[float, np.ndarray]:
        eta = log_exposure + full_design @ beta
        mu = np.exp(eta)
        residual = mu - counts

        nll = float(np.sum(mu - counts * eta))
        grad = full_design.T @ residual

        base_beta = beta[:n_base_params]
        phi_beta = beta[n_base_params:]

        base_abs = _soft_abs(
            base_beta,
            eps=l1_smooth_eps,
        )

        phi_abs = _soft_abs(
            phi_beta,
            eps=l1_smooth_eps,
        )

        penalty_base = min_penalty_alpha * np.sum(base_abs)
        penalty_phi = lambda_phi * np.sum(phi_abs)

        grad[:n_base_params] += (
            min_penalty_alpha
            * base_beta
            / base_abs
        )

        grad[n_base_params:] += (
            lambda_phi
            * phi_beta
            / phi_abs
        )

        objective_value = (
            nll
            + penalty_base
            + penalty_phi
        )

        return objective_value, grad
    
    bic_patience = 8
    bic_tolerance = 1e-8
    no_bic_improvement_count = 0
    best_bic_so_far = np.inf
    n_lamba_values_evaluated = 0

    for lambda_phi in lambda_values:
        n_lamba_values_evaluated += 1
        result = minimize(
            fun=lambda beta: full_objective_and_gradient(
                beta,
                float(lambda_phi),
            ),
            x0=current_start,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
            },
        )

        beta_hat = result.x
        current_start = beta_hat.copy()

        unpenalized_nll = compute_unpenalized_nll(beta_hat)

        df = int(np.sum(np.abs(beta_hat) > 1e-8))

        bic = 2.0 * unpenalized_nll + df * np.log(n_obs)

        candidate = {
            "alpha": float(beta_hat[0]),
            "gamma": beta_hat[1:n_base_params].tolist(),
            "phi_by_congress": beta_hat[n_base_params:].tolist(),
            "lambda_phi": float(lambda_phi),
            "lambda_max": float(lambda_max),
            "lambda_path_steps": int(lambda_path_steps),
            "lambda_path_min_ratio": float(lambda_path_min_ratio),
            "bic": float(bic),
            "unpenalized_nll": float(unpenalized_nll),
            "df": int(df),
            "n_obs": int(n_obs),
            "total_count": float(counts.sum()),
            "restricted_converged": bool(restricted_fit.success),
            "restricted_optimizer_message": str(restricted_fit.message),
            "converged": bool(result.success),
            "optimizer_message": str(result.message),
        }

        if best_result is None or candidate["bic"] < best_result["bic"]:
            best_result = candidate

        if bic < best_bic_so_far - bic_tolerance:
            best_bic_so_far = bic
            no_bic_improvement_count = 0
        else:
            no_bic_improvement_count += 1

        if no_bic_improvement_count >= bic_patience and n_lamba_values_evaluated > 50:
            break

    best_result["n_lambda_values_evaluated"] = int(n_lamba_values_evaluated)
    best_result["n_party_parameters"] = int(n_party_params)

    return best_result

def compute_choice_probabilities_static_covariates_time_varying_party(
    phrase_parameters: pd.DataFrame,
    base_covariate_matrix: np.ndarray,
    speaker_congress_index: np.ndarray,
    congress_values: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute party-specific choice probabilities and average partisanship
    for the static-covariate / time-varying-region / time-varying-party model.

    Returns
    -------
    phrase_probabilities_long:
        One row per Congress × phrase.

    partisanship_by_congress:
        One row per Congress with average partisanship.
    """
    if phrase_parameters.empty:
        raise ValueError("phrase_parameters is empty.")

    params = phrase_parameters.copy().reset_index(drop=True)

    X = np.asarray(base_covariate_matrix, dtype=float)

    speaker_congress_index = np.asarray(
        speaker_congress_index,
        dtype=int,
    )

    if X.shape[0] != len(speaker_congress_index):
        raise ValueError(
            "base_covariate_matrix and speaker_congress_index must have "
            "the same number of rows."
        )

    if np.any(speaker_congress_index < 0) or np.any(
        speaker_congress_index >= len(congress_values)
    ):
        raise ValueError(
            "speaker_congress_index contains invalid Congress indices."
        )

    alphas = params["alpha"].to_numpy(dtype=float)

    gamma_matrix = np.vstack(
        params["gamma"]
        .apply(lambda x: np.asarray(x, dtype=float))
        .to_list()
    )

    phi_matrix = np.vstack(
        params["phi_by_congress"]
        .apply(lambda x: np.asarray(x, dtype=float))
        .to_list()
    )

    if X.shape[1] != gamma_matrix.shape[1]:
        raise ValueError(
            "base_covariate_matrix has incompatible number of columns."
        )

    if phi_matrix.shape[1] != len(congress_values):
        raise ValueError(
            "phi_by_congress has incompatible number of Congress columns."
        )

    base_eta = X @ gamma_matrix.T
    base_eta = base_eta + alphas.reshape(1, -1)

    phi_for_speaker_phrase = (
        phi_matrix[:, speaker_congress_index]
        .T
    )

    eta_democrat = base_eta
    eta_republican = base_eta + phi_for_speaker_phrase

    log_denom_democrat = logsumexp(
        eta_democrat,
        axis=1,
        keepdims=True,
    )

    log_denom_republican = logsumexp(
        eta_republican,
        axis=1,
        keepdims=True,
    )

    q_democrat = np.exp(
        eta_democrat - log_denom_democrat
    )

    q_republican = np.exp(
        eta_republican - log_denom_republican
    )

    posterior_republican = (
        q_republican
        / (q_republican + q_democrat)
    )

    speaker_partisanship = 0.5 * np.sum(
        q_republican * posterior_republican
        + q_democrat * (1.0 - posterior_republican),
        axis=1,
    )

    weighted_rho_republican_total = np.sum(
        q_republican * posterior_republican,
        axis=1,
        keepdims=True,
    )

    weighted_rho_democrat_total = np.sum(
        q_democrat * posterior_republican,
        axis=1,
        keepdims=True,
    )

    epsilon = 1e-15

    denom_republican_without_phrase = np.maximum(
        1.0 - q_republican,
        epsilon,
    )

    denom_democrat_without_phrase = np.maximum(
        1.0 - q_democrat,
        epsilon,
    )

    expected_posterior_without_phrase = 0.5 * (
        (
            weighted_rho_republican_total
            - q_republican * posterior_republican
        )
        / denom_republican_without_phrase
        +
        (
            weighted_rho_democrat_total
            - q_democrat * posterior_republican
        )
        / denom_democrat_without_phrase
    )

    phrase_rows = []
    partisanship_rows = []

    for congress_idx, congress in enumerate(congress_values):
        mask = speaker_congress_index == congress_idx

        if not np.any(mask):
            continue

        average_partisanship = float(
            np.mean(speaker_partisanship[mask])
        )

        partisanship_rows.append({
            "congress": congress,
            "average_partisanship": average_partisanship,
            "n_speaker_sessions": int(mask.sum()),
        })

        phrase_partisanship = np.mean(
            0.5 - expected_posterior_without_phrase[mask],
            axis=0,
        )

        congress_params = params.copy()

        congress_params["congress"] = congress
        congress_params["phi"] = phi_matrix[:, congress_idx]

        congress_params["q_republican_mean"] = (
            q_republican[mask]
            .mean(axis=0)
        )

        congress_params["q_democrat_mean"] = (
            q_democrat[mask]
            .mean(axis=0)
        )

        congress_params["posterior_republican_mean"] = (
            posterior_republican[mask]
            .mean(axis=0)
        )

        congress_params["predicted_per_100k_republican"] = (
            congress_params["q_republican_mean"]
            * 100000
        )

        congress_params["predicted_per_100k_democrat"] = (
            congress_params["q_democrat_mean"]
            * 100000
        )

        congress_params["phrase_partisanship"] = phrase_partisanship

        congress_params["abs_phrase_partisanship"] = (
            congress_params["phrase_partisanship"]
            .abs()
        )

        congress_params["gamma_json"] = (
            congress_params["gamma"]
            .apply(json.dumps)
        )

        congress_params["phi_by_congress_json"] = (
            congress_params["phi_by_congress"]
            .apply(json.dumps)
        )

        phrase_rows.append(congress_params)

    phrase_probabilities_long = pd.concat(
        phrase_rows,
        ignore_index=True,
    )

    partisanship_by_congress = pd.DataFrame(partisanship_rows)

    return phrase_probabilities_long, partisanship_by_congress

def estimate_static_covariate_partisanship_model(
    speaker_data: pd.DataFrame,
    phrase_counts: pd.DataFrame,
    lambda_path_steps: int,
    lambda_path_min_ratio: float,
    min_penalty_alpha: float,
    maxiter: int,
    max_phrases: int | None = None,
    return_phrase_parameters: bool = False,
    progress_label: str | None = None,
    n_jobs: int = 1,
    parallel_backend: str = "threading",
) -> dict:
    """
    Estimate the pooled static-covariate / time-varying-party model.

    This is the central model function used by scripts 06-09.

    It estimates one pooled model per phrase over all Congresses in the
    supplied data. The resulting average partisanship is then computed
    separately by Congress.

    Parameters
    ----------
    speaker_data:
        Speaker-session-level data after exposure and covariates have already
        been added. Must include speaker_session_id, congress, party,
        republican, exposure, state, chamber_member, gender, census_region,
        and party_in_majority.

    phrase_counts:
        Long phrase-count table.

    max_phrases:
        Optional pooled phrase limit. If provided, the top phrases by total
        count over the supplied sample are kept.

    return_phrase_parameters:
        If True, return raw phrase parameters and design metadata. This is
        needed for out-of-sample validation.

    progress_label:
        Optional text prefix for progress messages.

    Returns
    -------
    dict
        Contains success flag, partisanship table, phrase probabilities,
        and optionally raw phrase parameters and design metadata.
    """
    speaker_data = speaker_data.copy()
    phrase_counts = phrase_counts.copy()

    required_speaker_cols = [
        "speaker_session_id",
        "congress",
        "party",
        "republican",
        "exposure",
        "state",
        "chamber_member",
        "gender",
        "census_region",
        "party_in_majority",
    ]

    missing_speaker_cols = [
        col for col in required_speaker_cols
        if col not in speaker_data.columns
    ]

    if missing_speaker_cols:
        return {
            "success": False,
            "reason": (
                "missing required speaker columns: "
                f"{missing_speaker_cols}"
            ),
        }

    if phrase_counts.empty:
        return {
            "success": False,
            "reason": "phrase_counts is empty",
        }

    speaker_data["party"] = (
        speaker_data["party"]
        .astype("string")
        .str.strip()
    )

    speaker_data["congress"] = (
        speaker_data["congress"]
        .astype(int)
    )

    speaker_data = speaker_data[
        speaker_data["party"].isin(["Democrat", "Republican"])
        & (speaker_data["exposure"] > 0)
    ].copy()

    if speaker_data.empty:
        return {
            "success": False,
            "reason": "no usable speaker-sessions",
        }

    # Keep phrase counts only for supplied usable speaker-sessions.
    phrase_counts = phrase_counts.merge(
        speaker_data[["speaker_session_id"]],
        on="speaker_session_id",
        how="inner",
    )

    if phrase_counts.empty:
        return {
            "success": False,
            "reason": "no phrase counts for usable speaker-sessions",
        }

    # Require both parties within every Congress included in the sample.
    congress_party_counts = (
        speaker_data
        .groupby(["congress", "party"], as_index=False)
        .agg(n_speaker_sessions=("speaker_session_id", "nunique"))
    )

    party_coverage = (
        congress_party_counts
        .pivot_table(
            index="congress",
            columns="party",
            values="n_speaker_sessions",
            fill_value=0,
        )
        .reset_index()
    )

    for party in ["Democrat", "Republican"]:
        if party not in party_coverage.columns:
            party_coverage[party] = 0

    bad_congresses = party_coverage.loc[
        (party_coverage["Democrat"] <= 0)
        | (party_coverage["Republican"] <= 0),
        "congress",
    ].astype(int).tolist()

    if bad_congresses:
        return {
            "success": False,
            "reason": (
                "not all Congresses contain both parties. "
                f"bad_congresses={bad_congresses}"
            ),
        }

    # Optional phrase limit per Congress.
    if max_phrases is not None:
        top_phrases_by_congress = (
            phrase_counts
            .groupby(["congress", "phrase"], as_index=False)
            .agg(total_count=("count", "sum"))
            .sort_values(
                ["congress", "total_count"],
                ascending=[True, False],
            )
            .groupby("congress", group_keys=False)
            .head(max_phrases)
        )

        phrase_counts = phrase_counts.merge(
            top_phrases_by_congress[["congress", "phrase"]],
            on=["congress", "phrase"],
            how="inner",
        )

        if phrase_counts.empty:
            return {
                "success": False,
                "reason": "no phrase counts after per-Congress phrase limit",
            }

    speaker_data = (
        speaker_data
        .sort_values("speaker_session_id")
        .copy()
    )

    speaker_ids = speaker_data["speaker_session_id"].tolist()

    speaker_index = {
        speaker_id: idx
        for idx, speaker_id in enumerate(speaker_ids)
    }

    exposures = speaker_data["exposure"].to_numpy(dtype=float)

    print(
        "Building static-covariate / time-varying-party design matrices..."
        )

    (
        base_covariate_matrix,
        party_interaction_matrix,
        base_covariate_columns,
        party_interaction_columns,
        congress_values,
        speaker_congress_index,
    ) = build_static_covariate_time_varying_party_design(
        speaker_data=speaker_data,
        static_categorical_cols=[
            "state",
            "chamber_member",
            "gender",
        ],
        time_varying_categorical_cols=[
            "census_region",
        ],
        static_binary_cols=[
            "party_in_majority",
        ],
    )

    print(
        "Design matrices built. Fitting phrase-specific models."
        )

    # --------------------------------------------------------------
    # Prepare phrase jobs.
    # We first map speaker_session_id to a numeric speaker_index.
    # This avoids repeated dictionary lookups inside every phrase loop.
    # --------------------------------------------------------------
    phrase_counts["speaker_index"] = (
        phrase_counts["speaker_session_id"]
        .map(speaker_index)
    )

    phrase_counts = phrase_counts[
        phrase_counts["speaker_index"].notna()
    ].copy()

    phrase_counts["speaker_index"] = (
        phrase_counts["speaker_index"]
        .astype(int)
    )

    n_speakers = len(speaker_ids)

    phrase_jobs = []

    for phrase, group in phrase_counts.groupby("phrase", sort=True):
        if group["speaker_index"].duplicated().any():
            duplicated_rows = group[
                group["speaker_index"].duplicated(keep=False)
            ][
                ["phrase", "speaker_session_id", "speaker_index", "count"]
            ].sort_values("speaker_index")

            raise ValueError(
                "Duplicate phrase-speaker rows found while preparing phrase jobs. "
                f"Phrase: '{phrase}'. "
                f"First duplicated rows:\n"
                f"{duplicated_rows.head(20).to_string(index=False)}"
            )
        
        speaker_indices = group["speaker_index"].to_numpy(dtype=int)
        count_values = group["count"].to_numpy(dtype=float)

        phrase_jobs.append(
            (
                phrase,
                speaker_indices,
                count_values,
            )
        )


    def fit_phrase_job(
        phrase: str,
        speaker_indices: np.ndarray,
        count_values: np.ndarray,
    ) -> dict | None:
        counts = np.zeros(n_speakers, dtype=float)

        counts[speaker_indices] = count_values

        if counts.sum() <= 0:
            return None

        result = fit_one_phrase_static_covariates_time_varying_party(
            counts=counts,
            exposures=exposures,
            base_covariate_matrix=base_covariate_matrix,
            party_interaction_matrix=party_interaction_matrix,
            lambda_path_steps=lambda_path_steps,
            lambda_path_min_ratio=lambda_path_min_ratio,
            min_penalty_alpha=min_penalty_alpha,
            maxiter=maxiter,
        )

        result["phrase"] = phrase
        result["base_covariate_columns"] = base_covariate_columns
        result["party_interaction_columns"] = party_interaction_columns
        result["congress_values"] = congress_values

        return result


    # --------------------------------------------------------------
    # Fit phrase-specific models.
    # If n_jobs == 1, run sequentially.
    # If n_jobs != 1, run phrase models in parallel.
    # --------------------------------------------------------------
    if n_jobs == 1:
        phrase_parameters = []

        for phrase_number, (
            phrase,
            speaker_indices,
            count_values,
        ) in enumerate(phrase_jobs, start=1):

            result = fit_phrase_job(
                phrase=phrase,
                speaker_indices=speaker_indices,
                count_values=count_values,
            )

            if result is not None:
                phrase_parameters.append(result)

            if phrase_number % 250 == 0:
                if progress_label is None:
                    print(f"Estimated {phrase_number:,} phrases...")
                else:
                    print(
                        f"{progress_label}: "
                        f"estimated {phrase_number:,} phrases..."
                    )

    else:
        print(
            f"Fitting {len(phrase_jobs):,} phrase models "
            f"with n_jobs={n_jobs}, backend={parallel_backend}."
        )

        phrase_parameters = Parallel(
            n_jobs=n_jobs,
            backend=parallel_backend,
            verbose=10,
        )(
            delayed(fit_phrase_job)(
                phrase,
                speaker_indices,
                count_values,
            )
            for phrase, speaker_indices, count_values in phrase_jobs
        )

        phrase_parameters = [
            result
            for result in phrase_parameters
            if result is not None
        ]

    phrase_parameters = pd.DataFrame(phrase_parameters)

    if phrase_parameters.empty:
        return {
            "success": False,
            "reason": "no phrase parameters estimated",
        }
    
    n_best_models = int(len(phrase_parameters))

    n_best_models_converged = int(
        phrase_parameters["converged"].fillna(False).sum()
    )

    n_best_models_not_converged = (
        n_best_models - n_best_models_converged
    )

    best_model_convergence_rate = (
        n_best_models_converged / n_best_models
        if n_best_models > 0
        else np.nan
    )

    best_model_convergence_percent = (
        100.0 * best_model_convergence_rate
        if n_best_models > 0
        else np.nan
    )

    print_section("Best-model convergence diagnostics")

    print("Best selected phrase models:", n_best_models)
    print("Converged best models:", n_best_models_converged)
    print("Non-converged best models:", n_best_models_not_converged)
    print(
        "Converged best models:",
        f"{best_model_convergence_percent:.2f}%",
    )

    phrase_probabilities, partisanship_df = (
        compute_choice_probabilities_static_covariates_time_varying_party(
            phrase_parameters=phrase_parameters,
            base_covariate_matrix=base_covariate_matrix,
            speaker_congress_index=speaker_congress_index,
            congress_values=congress_values,
        )
    )

    # Add Congress-level metadata.
    congress_metadata = (
        speaker_data
        .groupby("congress", as_index=False)
        .agg(
            n_speaker_sessions=("speaker_session_id", "nunique"),
            n_democrat_speaker_sessions=(
                "party",
                lambda x: int((x == "Democrat").sum()),
            ),
            n_republican_speaker_sessions=(
                "party",
                lambda x: int((x == "Republican").sum()),
            ),
            total_exposure=("exposure", "sum"),
        )
    )

    phrase_metadata = (
        phrase_counts
        .groupby("congress", as_index=False)
        .agg(
            total_bigram_count=("count", "sum"),
            n_phrases=("phrase", "nunique"),
        )
    )

    partisanship_df = (
        partisanship_df
        .drop(columns=["n_speaker_sessions"], errors="ignore")
        .merge(congress_metadata, on="congress", how="left")
        .merge(phrase_metadata, on="congress", how="left")
    )

    partisanship_df["n_base_covariates"] = len(base_covariate_columns)
    partisanship_df["n_party_parameters"] = len(party_interaction_columns)

    phrase_probabilities["base_covariate_columns_json"] = json.dumps(
        base_covariate_columns
    )

    phrase_probabilities["party_interaction_columns_json"] = json.dumps(
        party_interaction_columns
    )

    phrase_probabilities["congress_values_json"] = json.dumps(
        [int(x) for x in congress_values]
    )

    phrase_probabilities_for_output = (
        phrase_probabilities
        .drop(
            columns=[
                "gamma",
                "phi_by_congress",
                "base_covariate_columns",
                "party_interaction_columns",
                "congress_values",
            ],
            errors="ignore",
        )
        .copy()
    )

    output = {
        "success": True,
        "reason": "",
        "partisanship": partisanship_df,
        "phrase_probabilities": phrase_probabilities_for_output,
        "n_phrases": len(phrase_parameters),
        "n_speaker_sessions": len(speaker_data),
        "base_covariate_columns": base_covariate_columns,
        "party_interaction_columns": party_interaction_columns,
        "congress_values": congress_values,
        "best_model_convergence_percent": best_model_convergence_percent,
    }

    if return_phrase_parameters:
        output["raw_phrase_parameters"] = phrase_parameters
        output["speaker_data"] = speaker_data
        output["phrase_counts"] = phrase_counts
        output["base_covariate_matrix"] = base_covariate_matrix
        output["party_interaction_matrix"] = party_interaction_matrix
        output["speaker_congress_index"] = speaker_congress_index

    return output


def get_top_partisan_phrases(
    phrase_probabilities: pd.DataFrame,
    n: int = 50,
) -> pd.DataFrame:
    """
    Return most partisan phrases by absolute phrase partisanship.
    """
    required_cols = [
        "congress",
        "phrase",
        "phrase_partisanship",
        "abs_phrase_partisanship",
        "predicted_per_100k_republican",
        "predicted_per_100k_democrat",
        "posterior_republican_mean",
        "lambda_phi",
        "bic",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in phrase_probabilities.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns for top phrases: {missing_cols}"
        )

    return (
        phrase_probabilities[required_cols]
        .sort_values("abs_phrase_partisanship", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def get_top_partisan_phrases_by_direction(
    phrase_probabilities: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return top Republican and Democratic phrases separately.
    """
    republican = (
        phrase_probabilities[
            phrase_probabilities["phrase_partisanship"] > 0
        ]
        .sort_values("phrase_partisanship", ascending=False)
        .head(n)
        .copy()
    )

    republican["direction"] = "Republican"

    democratic = (
        phrase_probabilities[
            phrase_probabilities["phrase_partisanship"] < 0
        ]
        .sort_values("phrase_partisanship", ascending=True)
        .head(n)
        .copy()
    )

    democratic["direction"] = "Democratic"

    cols = [
        "congress",
        "direction",
        "phrase",
        "phrase_partisanship",
        "abs_phrase_partisanship",
        "predicted_per_100k_republican",
        "predicted_per_100k_democrat",
        "posterior_republican_mean",
        "lambda_phi",
        "bic",
    ]

    return (
        pd.concat([republican, democratic], ignore_index=True)
        [cols]
        .reset_index(drop=True)
    )

def build_lambda_path_from_lambda_max(
    lambda_max: float,
    lambda_path_steps: int = 100,
    lambda_path_min_ratio: float = 1e-5,
) -> np.ndarray:
    """
    Build a Gentzkow-style decreasing regularization path.

    The path starts at lambda_max, where the party loading phi is set to zero,
    and decreases geometrically down to lambda_max * lambda_path_min_ratio.

    Parameters
    ----------
    lambda_max:
        Phrase-specific starting penalty.
    lambda_path_steps:
        Number of values on the path.
    lambda_path_min_ratio:
        Smallest lambda as a fraction of lambda_max.

    Returns
    -------
    np.ndarray
        Decreasing lambda path.
    """
    lambda_max = float(lambda_max)

    if lambda_max <= 0 or not np.isfinite(lambda_max):
        return np.array([0.0], dtype=float)

    if lambda_path_steps < 2:
        return np.array([lambda_max], dtype=float)

    path = lambda_max * np.geomspace(
        1.0,
        lambda_path_min_ratio,
        num=lambda_path_steps,
    )

    return path.astype(float)



def compute_gentzkow_subsampling_ci(
    full_estimate: float,
    subsample_estimates: np.ndarray,
    full_n: int,
    subsample_n: int,
    lower_order: int = 1,
    upper_order: int = 99,
) -> dict:
    """
    Compute Gentzkow-style subsampling confidence interval.

    Implements the formula based on:

        Q_k = sqrt(tau_k) * [
            log(pi_hat_k - 1/2) - log(pi_hat - 1/2)
        ]

    and

        CI = 1/2 + (
            exp(log(pi_hat - 1/2) - Q(upper_order) / sqrt(tau)),
            exp(log(pi_hat - 1/2) - Q(lower_order) / sqrt(tau))
        )

    where:
    - pi_hat is the full-sample estimate
    - pi_hat_k are the subsample estimates
    - tau_k is the subsample size
    - tau is the full-sample size
    - Q(b) is the b-th order statistic of Q_k
    """

    estimates = np.asarray(subsample_estimates, dtype=float)
    estimates = estimates[np.isfinite(estimates)]

    if len(estimates) == 0:
        print(
            "Warning: no valid subsample estimates available. "
            "Gentzkow-style confidence interval cannot be computed."
        )

        return {
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "subsample_mean": np.nan,
            "subsample_sd": np.nan,
            "n_successful_subsamples": 0,
            "q_lower_order_statistic": np.nan,
            "q_upper_order_statistic": np.nan,
            "ci_method": "gentzkow_log_subsampling",
        }

    subsample_mean_all = float(np.mean(estimates))
    subsample_sd_all = (
        float(np.std(estimates, ddof=1))
        if len(estimates) > 1
        else np.nan
    )

    if full_estimate <= 0.5:
        print(
            "Warning: full_estimate is <= 0.5. "
            "Gentzkow log(pi - 0.5) confidence interval cannot be computed. "
            f"full_estimate={full_estimate:.6f}."
        )

        return {
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "subsample_mean": subsample_mean_all,
            "subsample_sd": subsample_sd_all,
            "n_successful_subsamples": int(len(estimates)),
            "q_lower_order_statistic": np.nan,
            "q_upper_order_statistic": np.nan,
            "ci_method": "gentzkow_log_subsampling",
        }

    valid_estimates = estimates[estimates > 0.5]

    if len(valid_estimates) < len(estimates):
        print(
            "Warning: some subsample estimates are <= 0.5 and were removed "
            "before computing the Gentzkow log(pi - 0.5) confidence interval. "
            f"removed={len(estimates) - len(valid_estimates)}, "
            f"kept={len(valid_estimates)}."
        )

    estimates = valid_estimates

    if len(estimates) == 0:
        print(
            "Warning: no subsample estimates above 0.5 available. "
            "Gentzkow-style confidence interval cannot be computed."
        )

        return {
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "subsample_mean": np.nan,
            "subsample_sd": np.nan,
            "n_successful_subsamples": 0,
            "q_lower_order_statistic": np.nan,
            "q_upper_order_statistic": np.nan,
            "ci_method": "gentzkow_log_subsampling",
        }

    log_full_excess = np.log(full_estimate - 0.5)

    q_statistics = (
        np.sqrt(subsample_n)
        * (
            np.log(estimates - 0.5)
            - log_full_excess
        )
    )

    q_statistics_sorted = np.sort(q_statistics)

    n_q = len(q_statistics_sorted)

    if n_q < upper_order:
        print(
            "Warning: fewer successful subsamples than requested upper order. "
            f"n_successful={n_q}, requested upper_order={upper_order}. "
            "Using the largest available order statistic instead."
        )

    lower_index = max(lower_order - 1, 0)
    upper_index = min(upper_order - 1, n_q - 1)

    q_lower_order_statistic = float(q_statistics_sorted[lower_index])
    q_upper_order_statistic = float(q_statistics_sorted[upper_index])

    ci_lower = float(
        0.5
        + np.exp(
            log_full_excess
            - q_upper_order_statistic / np.sqrt(full_n)
        )
    )

    ci_upper = float(
        0.5
        + np.exp(
            log_full_excess
            - q_lower_order_statistic / np.sqrt(full_n)
        )
    )

    return {
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "subsample_mean": subsample_mean_all,
        "subsample_sd": subsample_sd_all,
        "n_successful_subsamples": int(len(estimates)),
        "q_lower_order_statistic": q_lower_order_statistic,
        "q_upper_order_statistic": q_upper_order_statistic,
        "ci_method": "gentzkow_log_subsampling",
    }


def compute_empirical_party_choice_probabilities(
    phrase_counts_t: pd.DataFrame,
    speakers_t: pd.DataFrame,
    phrases: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical q_R and q_D in a held-out speaker subset.

    q_R and q_D are phrase probability vectors over the provided phrase list.
    """

    speaker_party = (
        speakers_t[["speaker_session_id", "party"]]
        .drop_duplicates()
    )

    phrase_counts_for_merge = phrase_counts_t.drop(
        columns=["party"],
        errors="ignore",
    )

    counts_with_party = phrase_counts_for_merge.merge(
        speaker_party,
        on="speaker_session_id",
        how="inner",
    )

    party_phrase_counts = (
        counts_with_party
        .groupby(["party", "phrase"], as_index=False)
        .agg(count=("count", "sum"))
    )

    def make_q_for_party(party_name: str) -> np.ndarray:
        party_counts = (
            party_phrase_counts[
                party_phrase_counts["party"] == party_name
            ]
            .set_index("phrase")["count"]
            .reindex(phrases, fill_value=0.0)
            .to_numpy(dtype=float)
        )

        total = party_counts.sum()

        if total <= 0:
            return np.full(len(phrases), np.nan, dtype=float)

        return party_counts / total

    q_republican_empirical = make_q_for_party("Republican")
    q_democrat_empirical = make_q_for_party("Democrat")

    return q_republican_empirical, q_democrat_empirical

def compute_out_of_sample_partisanship_static_for_fold(
    train_phrase_parameters: pd.DataFrame,
    test_speakers: pd.DataFrame,
    test_phrase_counts: pd.DataFrame,
    base_covariate_columns: list[str],
    party_interaction_columns: list[str],
    congress_values: list[int],
) -> pd.DataFrame:
    """
    Compute out-of-sample partisanship for a global fold under the
    static-covariate / time-varying-party model.

    The trained phrase posterior p_j(x) is computed using the training
    phrase parameters applied to test speakers. The q_R and q_D terms are
    computed empirically in the held-out fold, separately by Congress.

    Returns
    -------
    pd.DataFrame
        One row per Congress in the test fold.
    """
    if train_phrase_parameters.empty:
        return pd.DataFrame([
            {
                "congress": np.nan,
                "success": False,
                "reason": "empty train phrase parameters",
                "out_of_sample_partisanship": np.nan,
                "n_test_speaker_sessions": len(test_speakers),
                "n_phrases": 0,
            }
        ])

    test_speakers = test_speakers.copy()
    test_phrase_counts = test_phrase_counts.copy()

    test_speakers["congress"] = (
        test_speakers["congress"]
        .astype(int)
    )

    (
        X_test,
        party_interaction_test,
        _base_columns_test,
        _party_columns_test,
        _congress_values_test,
        speaker_congress_index_test,
    ) = build_static_covariate_time_varying_party_design(
        speaker_data=test_speakers,
        static_categorical_cols=[
            "state",
            "chamber_member",
            "gender",
        ],
        time_varying_categorical_cols=[
            "census_region",
        ],
        static_binary_cols=[
            "party_in_majority",
        ],
        base_covariate_columns=base_covariate_columns,
        party_interaction_columns=party_interaction_columns,
        congress_values=congress_values,
    )

    params = train_phrase_parameters.copy().reset_index(drop=True)

    phrases = params["phrase"].tolist()

    alphas = params["alpha"].to_numpy(dtype=float)

    gamma_matrix = np.vstack(
        params["gamma"]
        .apply(lambda x: np.asarray(x, dtype=float))
        .to_list()
    )

    phi_matrix = np.vstack(
        params["phi_by_congress"]
        .apply(lambda x: np.asarray(x, dtype=float))
        .to_list()
    )

    if X_test.shape[1] != gamma_matrix.shape[1]:
        raise ValueError(
            "Test base covariate matrix has incompatible number of columns."
        )

    if phi_matrix.shape[1] != len(congress_values):
        raise ValueError(
            "phi_by_congress has incompatible number of Congress values."
        )

    base_eta = X_test @ gamma_matrix.T
    base_eta = base_eta + alphas.reshape(1, -1)

    phi_for_test_speaker_phrase = (
        phi_matrix[:, speaker_congress_index_test]
        .T
    )

    eta_democrat = base_eta
    eta_republican = base_eta + phi_for_test_speaker_phrase

    log_denom_democrat = logsumexp(
        eta_democrat,
        axis=1,
        keepdims=True,
    )

    log_denom_republican = logsumexp(
        eta_republican,
        axis=1,
        keepdims=True,
    )

    q_democrat_train = np.exp(
        eta_democrat - log_denom_democrat
    )

    q_republican_train = np.exp(
        eta_republican - log_denom_republican
    )

    posterior_republican_train = (
        q_republican_train
        / (q_republican_train + q_democrat_train)
    )

    rows = []

    for congress in sorted(test_speakers["congress"].unique()):
        speakers_test_t = test_speakers[
            test_speakers["congress"] == congress
        ].copy()

        phrase_counts_test_t = test_phrase_counts[
            test_phrase_counts["congress"] == congress
        ].copy()

        parties_present = set(
            speakers_test_t["party"]
            .dropna()
            .unique()
        )

        if parties_present != {"Democrat", "Republican"}:
            rows.append({
                "congress": congress,
                "success": False,
                "reason": f"test parties present = {parties_present}",
                "out_of_sample_partisanship": np.nan,
                "n_test_speaker_sessions": len(speakers_test_t),
                "n_phrases": len(phrases),
            })

            continue

        q_republican_empirical, q_democrat_empirical = (
            compute_empirical_party_choice_probabilities(
                phrase_counts_t=phrase_counts_test_t,
                speakers_t=speakers_test_t,
                phrases=phrases,
            )
        )

        if (
            np.any(~np.isfinite(q_republican_empirical))
            or np.any(~np.isfinite(q_democrat_empirical))
        ):
            rows.append({
                "congress": congress,
                "success": False,
                "reason": "held-out empirical q could not be computed for both parties",
                "out_of_sample_partisanship": np.nan,
                "n_test_speaker_sessions": len(speakers_test_t),
                "n_phrases": len(phrases),
            })

            continue

        mask = (
            test_speakers["congress"]
            .astype(int)
            .to_numpy()
            == int(congress)
        )

        speaker_partisanship = 0.5 * np.sum(
            q_republican_empirical.reshape(1, -1)
            * posterior_republican_train[mask]
            +
            q_democrat_empirical.reshape(1, -1)
            * (1.0 - posterior_republican_train[mask]),
            axis=1,
        )

        out_of_sample_partisanship = float(
            np.mean(speaker_partisanship)
        )

        rows.append({
            "congress": congress,
            "success": True,
            "reason": "",
            "out_of_sample_partisanship": out_of_sample_partisanship,
            "n_test_speaker_sessions": len(speakers_test_t),
            "n_phrases": len(phrases),
        })

    return pd.DataFrame(rows)