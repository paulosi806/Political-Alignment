# Out-of-sample validation for Gentzkow-style penalized Poisson partisanship.
#
# This script divides speaker-sessions into folds. For each fold, it estimates
# phrase partisanship terms on the training speaker-sessions and evaluates
# average partisanship using empirical party-specific phrase frequencies in
# the held-out fold.

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
    compute_out_of_sample_partisanship_static_for_fold,
)


def main():
    root = PROJECT_ROOT
    config = load_config(root)

    processed_dir = root / config["paths"]["processed"]

    output_dir = root / "outputs" / "tables" / "penalized_poisson"
    output_dir.mkdir(parents=True, exist_ok=True)

    speaker_sessions_path = processed_dir / "speaker_sessions.parquet"
    phrase_counts_path = processed_dir / "phrase_counts_long.parquet"

    full_partisanship_path = (
        output_dir / "average_partisanship_penalized_poisson.csv"
    )

    if not full_partisanship_path.exists():
        raise FileNotFoundError(
            f"Full-sample partisanship file not found: "
            f"{full_partisanship_path}. "
            "Run the main penalized Poisson script first."
        )

    speaker_sessions = pd.read_parquet(speaker_sessions_path)
    phrase_counts = pd.read_parquet(phrase_counts_path)
    full_partisanship = pd.read_csv(full_partisanship_path)

    print_section("Loaded data")
    print("Speaker sessions:", speaker_sessions.shape)
    print("Phrase counts:", phrase_counts.shape)
    print("Full-sample partisanship:", full_partisanship.shape)

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

    maxiter = poisson_config.get(
        "maxiter",
        1000,
    )

    n_jobs = poisson_config.get(
        "n_jobs",
        1,
    )

    parallel_backend = poisson_config.get(
        "parallel_backend",
        "threading",
    )


    oos_config = config.get("out_of_sample_validation", {})

    n_folds = oos_config.get(
        "n_folds",
        5,
    )

    random_seed = oos_config.get(
        "random_seed",
        9,
    )

    rng = np.random.default_rng(random_seed)

    print_section("Out-of-sample validation settings")
    print("n_folds:", n_folds)
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
            f"{missing_speaker_cols}."
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

    phrase_counts = phrase_counts.merge(
        speaker_data[["speaker_session_id"]],
        on="speaker_session_id",
        how="inner",
    )

    print_section("Prepared data")
    print("Usable speaker-sessions:", len(speaker_data))
    print("Phrase counts after restriction:", phrase_counts.shape)

    # ------------------------------------------------------------------
    # Out-of-sample validation with pooled model
    # ------------------------------------------------------------------
    fold_results = []
    summary_results = []

    congresses = sorted(full_partisanship["congress"].astype(int).unique())

    print_section("Building stratified folds by Congress")

    speaker_data["fold_id"] = -1

    for congress in congresses:
        speakers_t = speaker_data[
            speaker_data["congress"].astype(int) == int(congress)
        ].copy()

        if speakers_t.empty:
            print(f"Skipping fold assignment for Congress {congress}: no speakers.")
            continue

        speaker_ids_t = speakers_t["speaker_session_id"].to_numpy()

        shuffled_ids_t = speaker_ids_t.copy()
        rng.shuffle(shuffled_ids_t)

        folds_t = np.array_split(
            shuffled_ids_t,
            min(n_folds, len(shuffled_ids_t)),
        )

        for fold_id, test_ids_t in enumerate(folds_t, start=1):
            speaker_data.loc[
                speaker_data["speaker_session_id"].isin(test_ids_t),
                "fold_id",
            ] = fold_id

    assigned_folds = sorted(
        speaker_data.loc[speaker_data["fold_id"] > 0, "fold_id"]
        .astype(int)
        .unique()
    )

    print("Requested folds:", n_folds)
    print("Assigned folds:", assigned_folds)

    for fold_id in assigned_folds:
        print_section(f"Out-of-sample fold {fold_id}/{len(assigned_folds)}")

        speakers_train = speaker_data[
            speaker_data["fold_id"].astype(int) != int(fold_id)
        ].copy()

        speakers_test = speaker_data[
            speaker_data["fold_id"].astype(int) == int(fold_id)
        ].copy()

        train_ids = set(speakers_train["speaker_session_id"])
        test_ids = set(speakers_test["speaker_session_id"])

        phrase_counts_train = phrase_counts[
            phrase_counts["speaker_session_id"].isin(train_ids)
        ].copy()

        phrase_counts_test = phrase_counts[
            phrase_counts["speaker_session_id"].isin(test_ids)
        ].copy()

        print("Train speaker-sessions:", len(speakers_train))
        print("Test speaker-sessions:", len(speakers_test))
        print("Train phrase count rows:", len(phrase_counts_train))
        print("Test phrase count rows:", len(phrase_counts_test))

        # --------------------------------------------------------------
        # Basic fold validity checks.
        # Training must contain both parties in every Congress, because
        # the pooled training model estimates Republican × Congress terms.
        # Test must also contain both parties in a Congress for empirical
        # q_R and q_D to be computable there.
        # --------------------------------------------------------------
        train_party_coverage = (
            speakers_train
            .groupby(["congress", "party"], as_index=False)
            .agg(n=("speaker_session_id", "nunique"))
            .pivot_table(
                index="congress",
                columns="party",
                values="n",
                fill_value=0,
            )
            .reset_index()
        )

        for party in ["Democrat", "Republican"]:
            if party not in train_party_coverage.columns:
                train_party_coverage[party] = 0

        bad_train_congresses = (
            train_party_coverage.loc[
                (train_party_coverage["Democrat"] <= 0)
                | (train_party_coverage["Republican"] <= 0),
                "congress",
            ]
            .astype(int)
            .tolist()
        )

        if bad_train_congresses:
            reason = (
                "training fold does not contain both parties in all Congresses: "
                f"{bad_train_congresses}"
            )

            print("Fold failed:", reason)

            for congress in congresses:
                full_row = full_partisanship[
                    full_partisanship["congress"].astype(int) == int(congress)
                ]

                full_estimate = (
                    float(full_row["average_partisanship"].iloc[0])
                    if not full_row.empty
                    else np.nan
                )

                n_train_t = int(
                    (speakers_train["congress"].astype(int) == int(congress)).sum()
                )

                n_test_t = int(
                    (speakers_test["congress"].astype(int) == int(congress)).sum()
                )

                fold_results.append({
                    "congress": int(congress),
                    "fold_id": int(fold_id),
                    "success": False,
                    "reason": reason,
                    "out_of_sample_partisanship": np.nan,
                    "full_sample_partisanship": full_estimate,
                    "out_of_sample_minus_full_sample": np.nan,
                    "n_train_speaker_sessions": n_train_t,
                    "n_test_speaker_sessions": n_test_t,
                    "n_phrases": 0,
                })

            continue

        # --------------------------------------------------------------
        # Estimate the pooled model on the training fold.
        # --------------------------------------------------------------
        train_result = estimate_static_covariate_partisanship_model(
            speaker_data=speakers_train,
            phrase_counts=phrase_counts_train,
            lambda_path_steps=lambda_path_steps,
            lambda_path_min_ratio=lambda_path_min_ratio,
            min_penalty_alpha=min_penalty_alpha,
            maxiter=maxiter,
            max_phrases=max_phrases_per_congress,
            return_phrase_parameters=True,
            progress_label=f"OOS fold {fold_id}",
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

        if not train_result["success"]:
            reason = f"training failed: {train_result['reason']}"

            print("Fold failed:", reason)

            for congress in congresses:
                full_row = full_partisanship[
                    full_partisanship["congress"].astype(int) == int(congress)
                ]

                full_estimate = (
                    float(full_row["average_partisanship"].iloc[0])
                    if not full_row.empty
                    else np.nan
                )

                n_train_t = int(
                    (speakers_train["congress"].astype(int) == int(congress)).sum()
                )

                n_test_t = int(
                    (speakers_test["congress"].astype(int) == int(congress)).sum()
                )

                fold_results.append({
                    "congress": int(congress),
                    "fold_id": int(fold_id),
                    "success": False,
                    "reason": reason,
                    "out_of_sample_partisanship": np.nan,
                    "full_sample_partisanship": full_estimate,
                    "out_of_sample_minus_full_sample": np.nan,
                    "n_train_speaker_sessions": n_train_t,
                    "n_test_speaker_sessions": n_test_t,
                    "n_phrases": 0,
                })

            continue

        # --------------------------------------------------------------
        # Apply training parameters to held-out speakers and compute
        # out-of-sample partisanship by Congress.
        # --------------------------------------------------------------
        fold_oos = compute_out_of_sample_partisanship_static_for_fold(
            train_phrase_parameters=train_result["raw_phrase_parameters"],
            test_speakers=speakers_test,
            test_phrase_counts=phrase_counts_test,
            base_covariate_columns=train_result["base_covariate_columns"],
            party_interaction_columns=train_result["party_interaction_columns"],
            congress_values=train_result["congress_values"],
        )

        for congress in congresses:
            full_row = full_partisanship[
                full_partisanship["congress"].astype(int) == int(congress)
            ]

            if full_row.empty:
                print(f"Skipping Congress {congress}: no full-sample estimate.")
                continue

            full_estimate = float(
                full_row["average_partisanship"].iloc[0]
            )

            row_t = fold_oos[
                fold_oos["congress"].astype(float) == float(congress)
            ]

            n_train_t = int(
                (speakers_train["congress"].astype(int) == int(congress)).sum()
            )

            n_test_t = int(
                (speakers_test["congress"].astype(int) == int(congress)).sum()
            )

            if row_t.empty:
                fold_results.append({
                    "congress": int(congress),
                    "fold_id": int(fold_id),
                    "success": False,
                    "reason": "Congress missing from OOS fold output",
                    "out_of_sample_partisanship": np.nan,
                    "full_sample_partisanship": full_estimate,
                    "out_of_sample_minus_full_sample": np.nan,
                    "n_train_speaker_sessions": n_train_t,
                    "n_test_speaker_sessions": n_test_t,
                    "n_phrases": 0,
                })

                continue

            row_t = row_t.iloc[0]

            if bool(row_t["success"]):
                out_of_sample_partisanship = float(
                    row_t["out_of_sample_partisanship"]
                )

                out_of_sample_minus_full_sample = (
                    out_of_sample_partisanship
                    - full_estimate
                )

                print(
                    f"Congress {congress} OOS partisanship:",
                    round(out_of_sample_partisanship, 6),
                )
                print(
                    f"Congress {congress} full-sample partisanship:",
                    round(full_estimate, 6),
                )
                print(
                    f"Congress {congress} OOS minus full:",
                    round(out_of_sample_minus_full_sample, 6),
                )

            else:
                out_of_sample_partisanship = np.nan
                out_of_sample_minus_full_sample = np.nan

                print(
                    f"Congress {congress} fold failed:",
                    row_t["reason"],
                )

            fold_results.append({
                "congress": int(congress),
                "fold_id": int(fold_id),
                "success": bool(row_t["success"]),
                "reason": row_t["reason"],
                "out_of_sample_partisanship": out_of_sample_partisanship,
                "full_sample_partisanship": full_estimate,
                "out_of_sample_minus_full_sample": out_of_sample_minus_full_sample,
                "n_train_speaker_sessions": n_train_t,
                "n_test_speaker_sessions": n_test_t,
                "n_phrases": int(row_t["n_phrases"]),
            })

    # ------------------------------------------------------------------
    # Summarize OOS results by Congress
    # ------------------------------------------------------------------
    fold_results_df = pd.DataFrame(fold_results)

    for congress in congresses:
        full_row = full_partisanship[
            full_partisanship["congress"].astype(int) == int(congress)
        ]

        if full_row.empty:
            continue

        full_estimate = float(
            full_row["average_partisanship"].iloc[0]
        )

        successful_fold_estimates = (
            fold_results_df[
                (fold_results_df["congress"].astype(int) == int(congress))
                & (fold_results_df["success"])
            ]["out_of_sample_partisanship"]
            .to_numpy(dtype=float)
        )

        summary_results.append({
            "congress": int(congress),
            "full_sample_partisanship": full_estimate,
            "n_folds_requested": int(n_folds),
            "n_successful_folds": int(len(successful_fold_estimates)),
            "out_of_sample_mean": (
                float(np.mean(successful_fold_estimates))
                if len(successful_fold_estimates) > 0
                else np.nan
            ),
            "out_of_sample_sd": (
                float(np.std(successful_fold_estimates, ddof=1))
                if len(successful_fold_estimates) > 1
                else np.nan
            ),
            "out_of_sample_minus_full_sample": (
                float(np.mean(successful_fold_estimates) - full_estimate)
                if len(successful_fold_estimates) > 0
                else np.nan
            ),
        })

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print_section("Saving out-of-sample validation outputs")

    if len(fold_results) == 0:
        raise ValueError("No out-of-sample validation results were produced.")

    summary_results_df = pd.DataFrame(summary_results)

        # ------------------------------------------------------------------
    # Print summary over all out-of-sample folds
    # ------------------------------------------------------------------
    print_section("Summary over all out-of-sample folds")

    successful_folds = fold_results_df[
        fold_results_df["success"]
    ].copy()

    failed_folds = fold_results_df[
        ~fold_results_df["success"]
    ].copy()

    print("Requested folds:", n_folds)
    print("Assigned folds:", assigned_folds)
    print("Congresses:", congresses)
    print("Fold × Congress rows:", len(fold_results_df))
    print("Successful rows:", len(successful_folds))
    print("Failed rows:", len(failed_folds))

    if len(successful_folds) > 0:
        overall_oos_summary = pd.DataFrame([
            {
                "n_successful_rows": len(successful_folds),
                "mean_out_of_sample_partisanship": successful_folds[
                    "out_of_sample_partisanship"
                ].mean(),
                "sd_out_of_sample_partisanship": successful_folds[
                    "out_of_sample_partisanship"
                ].std(ddof=1),
                "mean_full_sample_partisanship": successful_folds[
                    "full_sample_partisanship"
                ].mean(),
                "mean_oos_minus_full_sample": successful_folds[
                    "out_of_sample_minus_full_sample"
                ].mean(),
                "sd_oos_minus_full_sample": successful_folds[
                    "out_of_sample_minus_full_sample"
                ].std(ddof=1),
                "min_oos_minus_full_sample": successful_folds[
                    "out_of_sample_minus_full_sample"
                ].min(),
                "max_oos_minus_full_sample": successful_folds[
                    "out_of_sample_minus_full_sample"
                ].max(),
            }
        ])

        print("\nOverall OOS summary:")
        print(overall_oos_summary.to_string(index=False))

        print("\nOOS summary by Congress:")
        print(
            summary_results_df
            .sort_values("congress")
            .to_string(index=False)
        )

        summary_by_fold = (
            successful_folds
            .groupby("fold_id", as_index=False)
            .agg(
                n_successful_congresses=("congress", "nunique"),
                mean_out_of_sample_partisanship=(
                    "out_of_sample_partisanship",
                    "mean",
                ),
                mean_full_sample_partisanship=(
                    "full_sample_partisanship",
                    "mean",
                ),
                mean_oos_minus_full_sample=(
                    "out_of_sample_minus_full_sample",
                    "mean",
                ),
                sd_oos_minus_full_sample=(
                    "out_of_sample_minus_full_sample",
                    "std",
                ),
                mean_n_train_speaker_sessions=(
                    "n_train_speaker_sessions",
                    "mean",
                ),
                mean_n_test_speaker_sessions=(
                    "n_test_speaker_sessions",
                    "mean",
                ),
                mean_n_phrases=("n_phrases", "mean"),
            )
            .sort_values("fold_id")
        )

        print("\nOOS summary by fold:")
        print(summary_by_fold.to_string(index=False))

    if len(failed_folds) > 0:
        failure_summary = (
            failed_folds
            .groupby("reason", as_index=False)
            .agg(n_failures=("fold_id", "size"))
            .sort_values("n_failures", ascending=False)
        )

        print("\nFailure summary:")
        print(failure_summary.to_string(index=False))

    fold_results_csv = (
        output_dir / "validation_oos_folds_penalized_poisson.csv"
    )

    summary_results_csv = (
        output_dir / "validation_oos_summary_penalized_poisson.csv"
    )

    save_table(fold_results_df, fold_results_csv)
    save_table(summary_results_df, summary_results_csv)

    print_section("Done")
    print("Outputs saved in:", output_dir)


if __name__ == "__main__":
    main()