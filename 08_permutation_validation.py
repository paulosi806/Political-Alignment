# Permutation validation for Gentzkow-style penalized Poisson partisanship.
#
# This script mirrors the main penalized Poisson script, but randomly
# reassigns party labels to speaker-sessions before estimation.
#
# In the permuted data, party and language are unrelated by construction.
# Therefore, average partisanship should be close to 0.5 if the estimator
# has little finite-sample bias.

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


    validation_config = config.get("validation", {})

    n_permutations = validation_config.get(
        "n_permutations",
        1,
    )

    random_seed = validation_config.get(
        "random_seed",
        9,
    )

    rng = np.random.default_rng(random_seed)

    print_section("Permutation validation settings")
    print("n_permutations:", n_permutations)
    print("random_seed:", random_seed)
    print("lambda_path_steps:", lambda_path_steps)
    print("lambda_path_min_ratio:", lambda_path_min_ratio)
    print("min_penalty_alpha:", min_penalty_alpha)
    print("max_phrases_per_congress:", max_phrases_per_congress)
    print("maxiter:", maxiter)
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

    print("\nOriginal party counts:")
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
    phrase_counts = phrase_counts.merge(
        speaker_data[["speaker_session_id"]],
        on="speaker_session_id",
        how="inner",
    )

    # ------------------------------------------------------------------
    # Permutation validation with pooled model
    # ------------------------------------------------------------------
    permutation_draws = []
    permutation_summary = []

    speaker_data["congress"] = speaker_data["congress"].astype(int)
    phrase_counts["congress"] = phrase_counts["congress"].astype(int)

    congresses = sorted(speaker_data["congress"].astype(int).unique())

    for permutation_id in range(1, n_permutations + 1):
        print_section(f"Permutation {permutation_id}/{n_permutations}")

        speaker_data_perm = speaker_data.copy()

        # --------------------------------------------------------------
        # Randomly reassign party labels within each Congress.
        # This preserves the number of Democrats and Republicans in each
        # Congress, but breaks the link between party and language.
        # --------------------------------------------------------------
        speaker_data_perm["party_original"] = speaker_data_perm["party"]

        permuted_party_values = []

        for congress in congresses:
            mask = speaker_data_perm["congress"].astype(int) == int(congress)

            parties_t = (
                speaker_data_perm
                .loc[mask, "party"]
                .to_numpy()
            )

            permuted_parties_t = rng.permutation(parties_t)

            permuted_party_values.append(
                pd.Series(
                    permuted_parties_t,
                    index=speaker_data_perm.loc[mask].index,
                )
            )

        speaker_data_perm["party"] = (
            pd.concat(permuted_party_values)
            .sort_index()
            .astype("string")
        )

        speaker_data_perm["republican"] = (
            speaker_data_perm["party"] == "Republican"
        ).astype(int)

        # party_in_majority must be recomputed because party was permuted.
        speaker_data_perm["party_in_majority"] = (
            speaker_data_perm["party"]
            == speaker_data_perm["majority_party"]
        ).astype(int)

        original_party_counts = (
            speaker_data
            .groupby(["congress", "party"])
            .size()
            .reset_index(name="n")
        )

        permuted_party_counts = (
            speaker_data_perm
            .groupby(["congress", "party"])
            .size()
            .reset_index(name="n")
        )

        print("\nOriginal party counts by Congress:")
        print(original_party_counts.to_string(index=False))

        print("\nPermuted party counts by Congress:")
        print(permuted_party_counts.to_string(index=False))

        # --------------------------------------------------------------
        # Estimate pooled model on permuted speaker_data.
        # phrase_counts stay unchanged: only party labels are randomized.
        # --------------------------------------------------------------
        model_result_perm = estimate_static_covariate_partisanship_model(
            speaker_data=speaker_data_perm,
            phrase_counts=phrase_counts,
            lambda_path_steps=lambda_path_steps,
            lambda_path_min_ratio=lambda_path_min_ratio,
            min_penalty_alpha=min_penalty_alpha,
            maxiter=maxiter,
            max_phrases=max_phrases_per_congress,
            return_phrase_parameters=False,
            progress_label=f"Permutation {permutation_id}",
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

        if not model_result_perm["success"]:
            print(
                "Permutation model failed:",
                model_result_perm["reason"],
            )

            for congress in congresses:
                speakers_perm_t = speaker_data_perm[
                    speaker_data_perm["congress"].astype(int) == int(congress)
                ]

                phrase_counts_t = phrase_counts[
                    phrase_counts["congress"].astype(int) == int(congress)
                ]

                permutation_draws.append({
                    "permutation_id": permutation_id,
                    "congress": int(congress),
                    "success": False,
                    "reason": model_result_perm["reason"],
                    "permutation_partisanship": np.nan,
                    "bias_from_0_5": np.nan,
                    "n_speaker_sessions": len(speakers_perm_t),
                    "n_democrat_speaker_sessions": int(
                        (speakers_perm_t["party"] == "Democrat").sum()
                    ),
                    "n_republican_speaker_sessions": int(
                        (speakers_perm_t["party"] == "Republican").sum()
                    ),
                    "n_phrases": 0,
                    "n_base_covariates": np.nan,
                    "n_party_parameters": np.nan,
                    "total_bigram_count": int(phrase_counts_t["count"].sum()),
                })

            continue

        partisanship_perm = model_result_perm["partisanship"].copy()

        all_partisanship_this_permutation = []

        for congress in congresses:
            row_t = partisanship_perm[
                partisanship_perm["congress"].astype(int) == int(congress)
            ]

            if row_t.empty:
                speakers_perm_t = speaker_data_perm[
                    speaker_data_perm["congress"].astype(int) == int(congress)
                ]

                phrase_counts_t = phrase_counts[
                    phrase_counts["congress"].astype(int) == int(congress)
                ]

                permutation_draws.append({
                    "permutation_id": permutation_id,
                    "congress": int(congress),
                    "success": False,
                    "reason": "Congress missing from permutation model output",
                    "permutation_partisanship": np.nan,
                    "bias_from_0_5": np.nan,
                    "n_speaker_sessions": len(speakers_perm_t),
                    "n_democrat_speaker_sessions": int(
                        (speakers_perm_t["party"] == "Democrat").sum()
                    ),
                    "n_republican_speaker_sessions": int(
                        (speakers_perm_t["party"] == "Republican").sum()
                    ),
                    "n_phrases": 0,
                    "n_base_covariates": np.nan,
                    "n_party_parameters": np.nan,
                    "total_bigram_count": int(phrase_counts_t["count"].sum()),
                })

                continue

            row_t = row_t.iloc[0]

            permutation_partisanship = float(
                row_t["average_partisanship"]
            )

            bias_from_0_5 = permutation_partisanship - 0.5

            permutation_draws.append({
                "permutation_id": permutation_id,
                "congress": int(congress),
                "success": True,
                "reason": "",
                "permutation_partisanship": permutation_partisanship,
                "bias_from_0_5": bias_from_0_5,
                "n_speaker_sessions": int(row_t["n_speaker_sessions"]),
                "n_democrat_speaker_sessions": int(
                    row_t["n_democrat_speaker_sessions"]
                ),
                "n_republican_speaker_sessions": int(
                    row_t["n_republican_speaker_sessions"]
                ),
                "n_phrases": int(row_t["n_phrases"]),
                "n_base_covariates": int(row_t["n_base_covariates"]),
                "n_party_parameters": int(row_t["n_party_parameters"]),
                "total_bigram_count": int(row_t["total_bigram_count"]),
            })

            all_partisanship_this_permutation.append({
                "congress": int(congress),
                "permutation_partisanship": permutation_partisanship,
                "bias_from_0_5": bias_from_0_5,
            })

            print(
                f"Congress {congress} permutation partisanship:",
                round(permutation_partisanship, 6),
            )

            print(
                f"Congress {congress} bias from 0.5:",
                round(bias_from_0_5, 6),
            )

        if all_partisanship_this_permutation:
            permutation_partisanship_df = pd.DataFrame(
                all_partisanship_this_permutation
            )

            permutation_summary.append({
                "permutation_id": permutation_id,
                "n_congresses_estimated": len(permutation_partisanship_df),
                "mean_permutation_partisanship": float(
                    permutation_partisanship_df[
                        "permutation_partisanship"
                    ].mean()
                ),
                "mean_bias_from_0_5": float(
                    permutation_partisanship_df[
                        "bias_from_0_5"
                    ].mean()
                ),
            })

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print_section("Saving permutation validation outputs")

    if len(permutation_draws) == 0:
        raise ValueError("No permutation validation results were produced.")

    permutation_draws_df = pd.DataFrame(permutation_draws)

    permutation_summary_df = pd.DataFrame(permutation_summary)

    # ------------------------------------------------------------------
    # Print summary over all permutation runs
    # ------------------------------------------------------------------
    print_section("Summary over all permutation runs")

    successful_draws = permutation_draws_df[
        permutation_draws_df["success"]
    ].copy()

    failed_draws = permutation_draws_df[
        ~permutation_draws_df["success"]
    ].copy()

    print("Requested permutations:", n_permutations)
    print("Congresses:", congresses)
    print("Permutation × Congress rows:", len(permutation_draws_df))
    print("Successful rows:", len(successful_draws))
    print("Failed rows:", len(failed_draws))

    if len(successful_draws) > 0:
        overall_summary = pd.DataFrame([
            {
                "n_successful_rows": len(successful_draws),
                "mean_permutation_partisanship": successful_draws[
                    "permutation_partisanship"
                ].mean(),
                "sd_permutation_partisanship": successful_draws[
                    "permutation_partisanship"
                ].std(ddof=1),
                "mean_bias_from_0_5": successful_draws[
                    "bias_from_0_5"
                ].mean(),
                "sd_bias_from_0_5": successful_draws[
                    "bias_from_0_5"
                ].std(ddof=1),
                "min_permutation_partisanship": successful_draws[
                    "permutation_partisanship"
                ].min(),
                "max_permutation_partisanship": successful_draws[
                    "permutation_partisanship"
                ].max(),
            }
        ])

        print("\nOverall summary:")
        print(overall_summary.to_string(index=False))

        summary_by_congress = (
            successful_draws
            .groupby("congress", as_index=False)
            .agg(
                n_successful_permutations=("permutation_id", "nunique"),
                mean_permutation_partisanship=(
                    "permutation_partisanship",
                    "mean",
                ),
                sd_permutation_partisanship=(
                    "permutation_partisanship",
                    "std",
                ),
                mean_bias_from_0_5=("bias_from_0_5", "mean"),
                sd_bias_from_0_5=("bias_from_0_5", "std"),
                min_permutation_partisanship=(
                    "permutation_partisanship",
                    "min",
                ),
                max_permutation_partisanship=(
                    "permutation_partisanship",
                    "max",
                ),
            )
            .sort_values("congress")
        )

        print("\nSummary by Congress:")
        print(summary_by_congress.to_string(index=False))

    if len(failed_draws) > 0:
        failure_summary = (
            failed_draws
            .groupby("reason", as_index=False)
            .agg(n_failures=("permutation_id", "size"))
            .sort_values("n_failures", ascending=False)
        )

        print("\nFailure summary:")
        print(failure_summary.to_string(index=False))

    permutation_draws_csv = (
        output_dir / "validation_permutation_draws_penalized_poisson.csv"
    )

    permutation_summary_csv = (
        output_dir / "validation_permutation_summary_penalized_poisson.csv"
    )

    save_table(permutation_draws_df, permutation_draws_csv)
    save_table(permutation_summary_df, permutation_summary_csv)

    print_section("Done")
    print("Outputs saved in:", output_dir)


if __name__ == "__main__":
    main()