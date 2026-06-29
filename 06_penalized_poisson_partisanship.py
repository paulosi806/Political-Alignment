# Estimate partisanship using Gentzkow-style penalized Poisson approximation
# with speaker covariates.

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import numpy as np
import pandas as pd

from political_speech.utils import load_config

from political_speech.utils_analysis import (
    print_section,
    save_table,
    add_census_region,
    add_majority_party_indicator,
    estimate_static_covariate_partisanship_model,
    get_top_partisan_phrases,
    get_top_partisan_phrases_by_direction,
)


def main():
    root = PROJECT_ROOT
    config = load_config(root)

    processed_dir = root / config["paths"]["processed"]

    output_dir = root / "outputs" / "tables" / "penalized_poisson"
    output_dir.mkdir(parents=True, exist_ok=True)

    speaker_sessions_path = processed_dir / "speaker_sessions.parquet"
    phrase_counts_path = processed_dir / "phrase_counts_long.parquet"

    speaker_sessions = pd.read_parquet(speaker_sessions_path)
    phrase_counts = pd.read_parquet(phrase_counts_path)

    print_section("Loaded data")
    print("Speaker sessions:", speaker_sessions.shape)
    print("Phrase counts:", phrase_counts.shape)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    poisson_config = config.get("penalized_poisson", {})

    lambda_path_steps = poisson_config.get(
        "lambda_path_steps",
        100,
    )

    lambda_path_min_ratio = poisson_config.get(
        "lambda_path_min_ratio",
        1e-5,
    )

    min_penalty_alpha = poisson_config.get(
        "min_penalty_alpha",
        1e-5,
    )

    max_phrases_per_congress = poisson_config.get(
        "max_phrases_per_congress",
        None,
    )

    top_n_phrases = poisson_config.get(
        "top_n_phrases",
        50,
    )

    top_n_phrases_by_direction = poisson_config.get(
        "top_n_phrases_by_direction",
        10,
    )

    majority_party_by_congress_chamber = poisson_config.get(
        "majority_party_by_congress_chamber",
        {},
    )

    maxiter = poisson_config.get("maxiter", 1000)

    n_jobs = poisson_config.get(
        "n_jobs",
        1,
    )

    parallel_backend = poisson_config.get(
        "parallel_backend",
        "threading",
    )

    print_section("Estimator settings")
    print("lambda_path_steps:", lambda_path_steps)
    print("lambda_path_min_ratio:", lambda_path_min_ratio)
    print("min_penalty_alpha:", min_penalty_alpha)
    print("max_phrases_per_congress:", max_phrases_per_congress)
    print("maxiter:", maxiter)
    print("top_n_phrases:", top_n_phrases)
    print("top_n_phrases_by_direction:", top_n_phrases_by_direction)
    print("n_jobs:", n_jobs )
    print("parallel_backend:", parallel_backend)
    

    # ------------------------------------------------------------------
    # Required speaker metadata
    # ------------------------------------------------------------------
    required_speaker_cols = [
        "speaker_session_id",
        "speaker_bioguide",
        "congress",
        "party",
        "state",
        "chamber_member",
        "gender",
    ]

    missing_speaker_cols = [
        col for col in required_speaker_cols
        if col not in speaker_sessions.columns
    ]

    if missing_speaker_cols:
        raise KeyError(
            "Missing required columns in speaker_sessions: "
            f"{missing_speaker_cols}. "
            "Make sure script 02 and 04 include state, chamber_member, and gender."
        )

    # ------------------------------------------------------------------
    # Prepare speaker-session exposure m_i
    # ------------------------------------------------------------------
    # m_i is the total count of all retained phrases in a speaker-session.
    speaker_exposure = (
        phrase_counts
        .groupby("speaker_session_id", as_index=False)
        .agg(exposure=("count", "sum"))
    )

    speaker_meta_cols = [
        "speaker_session_id",
        "speaker_bioguide",
        "congress",
        "party",
        "state",
        "chamber_member",
        "gender",
    ]

    speaker_data = (
        speaker_sessions[speaker_meta_cols]
        .drop_duplicates()
        .copy()
    )

    speaker_data = add_census_region(
        speaker_data,
        state_col="state",
    )

    speaker_data = add_majority_party_indicator(
        speaker_data,
        majority_party_by_congress_chamber=majority_party_by_congress_chamber,
        congress_col="congress",
        chamber_col="chamber_member",
        party_col="party",
    )

    speaker_data = speaker_data.merge(
        speaker_exposure,
        on="speaker_session_id",
        how="inner",
    )

    speaker_data["party"] = (
        speaker_data["party"]
        .astype("string")
        .str.strip()
    )

    speaker_data = speaker_data[
        speaker_data["party"].isin(["Democrat", "Republican"])
        & (speaker_data["exposure"] > 0)
    ].copy()

    speaker_data["republican"] = (
        speaker_data["party"] == "Republican"
    ).astype(int)

    print_section("Speaker exposure and covariates")
    print("Speaker-sessions with positive exposure:", len(speaker_data))
    print("Congresses:", sorted(speaker_data["congress"].unique()))

    print("\nParty counts:")
    print(speaker_data["party"].value_counts(dropna=False))

    print("\nCovariate missing values:")
    covariate_check_cols = [
        "state",
        "chamber_member",
        "gender",
        "census_region",
        "party_in_majority",
    ]
    print(speaker_data[covariate_check_cols].isna().sum())

    # Keep phrase counts only for usable speaker-sessions
    phrase_counts_shape_before = phrase_counts.shape
    phrase_counts_rows_before = len(phrase_counts)
    phrase_counts_speaker_sessions_before = phrase_counts["speaker_session_id"].nunique()

    usable_speaker_sessions = speaker_data["speaker_session_id"].nunique()

    phrase_counts = phrase_counts.merge(
        speaker_data[["speaker_session_id"]],
        on="speaker_session_id",
        how="inner",
    )

    phrase_counts_shape_after = phrase_counts.shape
    phrase_counts_rows_after = len(phrase_counts)
    phrase_counts_speaker_sessions_after = phrase_counts["speaker_session_id"].nunique()

    print("\nPhrase counts before restricting to usable speaker-sessions:")
    print(phrase_counts_shape_before)
    print("Speaker-sessions in phrase_counts before:", phrase_counts_speaker_sessions_before)

    print("\nUsable speaker-sessions in speaker_data:")
    print(usable_speaker_sessions)

    print("\nPhrase counts after restricting to usable speaker-sessions:")
    print(phrase_counts_shape_after)
    print("Speaker-sessions in phrase_counts after:", phrase_counts_speaker_sessions_after)

    print("\nRows removed from phrase_counts:")
    print(phrase_counts_rows_before - phrase_counts_rows_after)

    print("\nSpeaker-sessions removed from phrase_counts:")
    print(phrase_counts_speaker_sessions_before - phrase_counts_speaker_sessions_after)

    # ------------------------------------------------------------------
    # Estimate pooled static-covariate / time-varying-party model
    # ------------------------------------------------------------------
    print_section("Estimating pooled static-covariate / time-varying-party model")

    model_result = estimate_static_covariate_partisanship_model(
        speaker_data=speaker_data,
        phrase_counts=phrase_counts,
        lambda_path_steps=lambda_path_steps,
        lambda_path_min_ratio=lambda_path_min_ratio,
        min_penalty_alpha=min_penalty_alpha,
        maxiter=maxiter,
        max_phrases=max_phrases_per_congress,
        return_phrase_parameters=True,
        progress_label="Main model",
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
    )

    if not model_result["success"]:
        raise ValueError(
            "Pooled model estimation failed: "
            f"{model_result['reason']}"
        )

    partisanship_df = model_result["partisanship"]
    partisanship_df["best_model_convergence_percent"] = model_result["best_model_convergence_percent"]
    phrase_parameters_df = model_result["phrase_probabilities"]

    raw_phrase_parameters = model_result["raw_phrase_parameters"]

    base_covariate_columns = model_result["base_covariate_columns"]
    party_interaction_columns = model_result["party_interaction_columns"]
    congress_values = model_result["congress_values"]

    print_section("Pooled model results")
    print("Estimated phrases:", model_result["n_phrases"])
    print("Convergence rate:", model_result['best_model_convergence_percent'])
    print("Speaker-sessions:", model_result["n_speaker_sessions"])
    print("Congresses:", congress_values)
    print("Base covariates:", len(base_covariate_columns))
    print("Party interaction parameters:", len(party_interaction_columns))

    print("\nAverage partisanship by Congress:")
    print(
        partisanship_df
        .sort_values("congress")
        .to_string(index=False)
    )

    top_phrase_tables = []
    top_phrase_direction_tables = []

    for congress, phrase_parameters_t in phrase_parameters_df.groupby(
        "congress",
        sort=True,
    ):
        phrase_parameters_t = phrase_parameters_t.copy()
        phrase_parameters_t["congress"] = congress

        top_phrase_tables.append(
            get_top_partisan_phrases(
                phrase_parameters_t,
                n=top_n_phrases,
            )
        )

        top_phrase_direction_tables.append(
            get_top_partisan_phrases_by_direction(
                phrase_parameters_t,
                n=top_n_phrases_by_direction,
            )
        )

    top_phrases_df = pd.concat(
        top_phrase_tables,
        ignore_index=True,
    )

    top_phrases_by_direction_df = pd.concat(
        top_phrase_direction_tables,
        ignore_index=True,
    )

    print("\nTop partisan phrases overall:")
    print(
        top_phrases_df
        .head(30)
        .to_string(index=False)
    )

    print("\nTop partisan phrases by direction:")
    print(
        top_phrases_by_direction_df
        .head(40)
        .to_string(index=False)
    )
    
    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print_section("Saving outputs")

    partisanship_csv = (
        output_dir / "average_partisanship_penalized_poisson.csv"
    )

    phrase_parameters_parquet = (
        output_dir / "phrase_parameters_penalized_poisson.parquet"
    )

    raw_phrase_parameters_parquet = (
        output_dir / "raw_phrase_parameters_penalized_poisson.parquet"
    )

    top_phrases_csv = (
        output_dir / "top_partisan_phrases_penalized_poisson.csv"
    )

    top_phrases_by_direction_csv = (
        output_dir
        / "top_partisan_phrases_by_direction_penalized_poisson.csv"
    )

    design_metadata_json = (
        output_dir / "design_metadata_penalized_poisson.json"
    )

    save_table(partisanship_df, partisanship_csv)

    phrase_parameters_df.to_parquet(
        phrase_parameters_parquet,
        index=False,
    )
    print("Saved:", phrase_parameters_parquet)

    raw_phrase_parameters.to_parquet(
        raw_phrase_parameters_parquet,
        index=False,
    )
    print("Saved:", raw_phrase_parameters_parquet)

    save_table(top_phrases_df, top_phrases_csv)

    save_table(
        top_phrases_by_direction_df,
        top_phrases_by_direction_csv,
    )

    design_metadata = {
        "base_covariate_columns": base_covariate_columns,
        "party_interaction_columns": party_interaction_columns,
        "congress_values": [int(x) for x in congress_values],
        "lambda_path_steps": int(lambda_path_steps),
        "lambda_path_min_ratio": float(lambda_path_min_ratio),
        "min_penalty_alpha": float(min_penalty_alpha),
        "max_phrases": (
            None if max_phrases_per_congress is None else int(max_phrases_per_congress)
        ),
        "maxiter": int(maxiter),
    }

    with open(design_metadata_json, "w", encoding="utf-8") as f:
        import json
        json.dump(design_metadata, f, indent=2)

    print("Saved:", design_metadata_json)

    print_section("Done")
    print("Outputs saved in:", output_dir)

if __name__ == "__main__":
    main()