# Subsampling inference for Gentzkow-style penalized Poisson partisanship.
#
# This script draws random subsets without replacement of size one-tenth
# the number of speaker-sessions and re-estimates average partisanship.

from pathlib import Path
import sys
import random

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import numpy as np
import pandas as pd

from political_speech.utils import load_config

from political_speech.utils_analysis import (
    compute_gentzkow_subsampling_ci,
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

    inference_config = config.get("subsampling_inference", {})

    n_subsamples = inference_config.get(
        "n_subsamples",
        100,
    )

    subsample_fraction = inference_config.get(
        "subsample_fraction",
        0.1,
    )

    random_seed = inference_config.get(
        "random_seed",
        42,
    )

    lower_order = inference_config.get(
        "lower_order",
        1,
    )

    upper_order = inference_config.get(
        "upper_order",
        99,
    )

    max_draw_attempts = inference_config.get(
        "max_draw_attempts",
        500,
    )

    random.seed(random_seed)
    rng = np.random.default_rng(random_seed)

    print_section("Inference settings")
    print("n_subsamples:", n_subsamples)
    print("subsample_fraction:", subsample_fraction)
    print("random_seed:", random_seed)
    print("lower_order:", lower_order)
    print("upper_order:", upper_order)
    print("max_draw_attempts:", max_draw_attempts)
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
    print("Phrase counts after restricting to usable speakers:", phrase_counts.shape)

    # ------------------------------------------------------------------
    # Pooled subsampling inference
    # ------------------------------------------------------------------
    subsample_results = []
    ci_results = []

    congresses = sorted(full_partisanship["congress"].astype(int).unique())

    speaker_data["congress"] = speaker_data["congress"].astype(int)
    phrase_counts["congress"] = phrase_counts["congress"].astype(int)

    full_n_by_congress = (
        speaker_data
        .groupby("congress")
        .size()
        .to_dict()
    )

    subsample_n_by_congress = {
        congress: max(
            int(np.floor(subsample_fraction * full_n_by_congress[congress])),
            2,
        )
        for congress in congresses
    }

    print_section("Pooled subsampling")
    print("Congresses:", congresses)
    print("Full n by Congress:", full_n_by_congress)
    print("Subsample n by Congress:", subsample_n_by_congress)

    for draw_id in range(1, n_subsamples + 1):
        print_section(f"Pooled subsample {draw_id}/{n_subsamples}")

        draw_success = False
        sampled_ids_all = []

        for attempt in range(1, max_draw_attempts + 1):
            sampled_ids_attempt = []
            valid_attempt = True

            for congress in congresses:
                speakers_full_t = speaker_data[
                    speaker_data["congress"] == congress
                ].copy()

                if speakers_full_t.empty:
                    valid_attempt = False
                    break

                speaker_ids_t = (
                    speakers_full_t["speaker_session_id"]
                    .to_numpy()
                )

                subsample_n_t = subsample_n_by_congress[congress]

                if subsample_n_t > len(speaker_ids_t):
                    valid_attempt = False
                    break

                sampled_ids_t = rng.choice(
                    speaker_ids_t,
                    size=subsample_n_t,
                    replace=False,
                )

                speakers_sub_t = speakers_full_t[
                    speakers_full_t["speaker_session_id"].isin(sampled_ids_t)
                ].copy()

                parties_present = set(
                    speakers_sub_t["party"]
                    .dropna()
                    .unique()
                )

                if parties_present != {"Democrat", "Republican"}:
                    valid_attempt = False
                    break

                sampled_ids_attempt.extend(sampled_ids_t.tolist())

            if valid_attempt:
                sampled_ids_all = sampled_ids_attempt
                draw_success = True
                break

        if not draw_success:
            print(
                f"Skipping pooled draw {draw_id}: could not draw both parties "
                f"in every Congress after {max_draw_attempts} attempts."
            )

            for congress in congresses:
                subsample_results.append({
                    "congress": congress,
                    "draw_id": draw_id,
                    "success": False,
                    "reason": "could not draw both parties in every Congress",
                    "average_partisanship": np.nan,
                    "n_speaker_sessions": subsample_n_by_congress[congress],
                    "n_democrat_speaker_sessions": np.nan,
                    "n_republican_speaker_sessions": np.nan,
                    "n_phrases": 0,
                    "n_base_covariates": np.nan,
                    "n_party_parameters": np.nan,
                    "total_bigram_count": 0,
                    "full_n_speaker_sessions": full_n_by_congress[congress],
                    "subsample_n_speaker_sessions": subsample_n_by_congress[congress],
                    "subsample_fraction": subsample_fraction,
                    "random_seed": random_seed,
                })

            continue

        sampled_ids_all = set(sampled_ids_all)

        speaker_data_sub = speaker_data[
            speaker_data["speaker_session_id"].isin(sampled_ids_all)
        ].copy()

        phrase_counts_sub = phrase_counts[
            phrase_counts["speaker_session_id"].isin(sampled_ids_all)
        ].copy()

        print("Subsample speaker-sessions:", len(speaker_data_sub))
        print("Subsample phrase-count rows:", len(phrase_counts_sub))

        model_result_sub = estimate_static_covariate_partisanship_model(
            speaker_data=speaker_data_sub,
            phrase_counts=phrase_counts_sub,
            lambda_path_steps=lambda_path_steps,
            lambda_path_min_ratio=lambda_path_min_ratio,
            min_penalty_alpha=min_penalty_alpha,
            maxiter=maxiter,
            max_phrases=max_phrases_per_congress,
            return_phrase_parameters=False,
            progress_label=f"Subsample {draw_id}",
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
        )

        if not model_result_sub["success"]:
            print(
                "Subsample model failed:",
                model_result_sub["reason"],
            )

            for congress in congresses:
                subsample_results.append({
                    "congress": congress,
                    "draw_id": draw_id,
                    "success": False,
                    "reason": model_result_sub["reason"],
                    "average_partisanship": np.nan,
                    "n_speaker_sessions": subsample_n_by_congress[congress],
                    "n_democrat_speaker_sessions": np.nan,
                    "n_republican_speaker_sessions": np.nan,
                    "n_phrases": 0,
                    "n_base_covariates": np.nan,
                    "n_party_parameters": np.nan,
                    "total_bigram_count": 0,
                    "full_n_speaker_sessions": full_n_by_congress[congress],
                    "subsample_n_speaker_sessions": subsample_n_by_congress[congress],
                    "subsample_fraction": subsample_fraction,
                    "random_seed": random_seed,
                })

            continue

        partisanship_sub = model_result_sub["partisanship"].copy()

        for congress in congresses:
            row_t = partisanship_sub[
                partisanship_sub["congress"].astype(int) == int(congress)
            ]

            if row_t.empty:
                subsample_results.append({
                    "congress": congress,
                    "draw_id": draw_id,
                    "success": False,
                    "reason": "Congress missing from subsample model output",
                    "average_partisanship": np.nan,
                    "n_speaker_sessions": subsample_n_by_congress[congress],
                    "n_democrat_speaker_sessions": np.nan,
                    "n_republican_speaker_sessions": np.nan,
                    "n_phrases": 0,
                    "n_base_covariates": np.nan,
                    "n_party_parameters": np.nan,
                    "total_bigram_count": 0,
                    "full_n_speaker_sessions": full_n_by_congress[congress],
                    "subsample_n_speaker_sessions": subsample_n_by_congress[congress],
                    "subsample_fraction": subsample_fraction,
                    "random_seed": random_seed,
                })

                continue

            row_t = row_t.iloc[0]

            subsample_results.append({
                "congress": congress,
                "draw_id": draw_id,
                "success": True,
                "reason": "",
                "average_partisanship": float(row_t["average_partisanship"]),
                "n_speaker_sessions": int(row_t["n_speaker_sessions"]),
                "n_democrat_speaker_sessions": int(row_t["n_democrat_speaker_sessions"]),
                "n_republican_speaker_sessions": int(row_t["n_republican_speaker_sessions"]),
                "n_phrases": int(row_t["n_phrases"]),
                "n_base_covariates": int(row_t["n_base_covariates"]),
                "n_party_parameters": int(row_t["n_party_parameters"]),
                "total_bigram_count": int(row_t["total_bigram_count"]),
                "full_n_speaker_sessions": full_n_by_congress[congress],
                "subsample_n_speaker_sessions": subsample_n_by_congress[congress],
                "subsample_fraction": subsample_fraction,
                "random_seed": random_seed,
            })

            print(
                f"Congress {congress} subsample average partisanship:",
                round(float(row_t["average_partisanship"]), 6),
            )

    subsample_results_df = pd.DataFrame(subsample_results)

    # ------------------------------------------------------------------
    # Compute CIs by Congress
    # ------------------------------------------------------------------
    for congress in congresses:
        full_row = full_partisanship[
            full_partisanship["congress"].astype(int) == int(congress)
        ]

        if full_row.empty:
            print(f"Skipping CI for Congress {congress}: no full estimate.")
            continue

        full_estimate = float(
            full_row["average_partisanship"].iloc[0]
        )

        estimates_t = (
            subsample_results_df[
                (subsample_results_df["congress"].astype(int) == int(congress))
                & (subsample_results_df["success"])
            ]["average_partisanship"]
            .to_numpy(dtype=float)
        )

        full_n = int(full_n_by_congress[congress])
        subsample_n = int(subsample_n_by_congress[congress])

        ci = compute_gentzkow_subsampling_ci(
            full_estimate=full_estimate,
            subsample_estimates=estimates_t,
            full_n=full_n,
            subsample_n=subsample_n,
            lower_order=lower_order,
            upper_order=upper_order,
        )

        ci_results.append({
            "congress": congress,
            "average_partisanship": full_estimate,
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "n_requested_subsamples": n_subsamples,
            "n_successful_subsamples": ci["n_successful_subsamples"],
            "subsample_mean": ci["subsample_mean"],
            "subsample_sd": ci["subsample_sd"],
            "q_lower_order_statistic": ci["q_lower_order_statistic"],
            "q_upper_order_statistic": ci["q_upper_order_statistic"],
            "ci_method": ci["ci_method"],
            "lower_order": lower_order,
            "upper_order": upper_order,
            "full_n_speaker_sessions": full_n,
            "subsample_n_speaker_sessions": subsample_n,
            "subsample_fraction": subsample_fraction,
            "random_seed": random_seed,
        })

        print_section(f"Gentzkow-style subsampling CI: Congress {congress}")
        print("Estimate:", round(full_estimate, 6))
        print("CI lower:", round(ci["ci_lower"], 6))
        print("CI upper:", round(ci["ci_upper"], 6))
        print("Successful subsamples:", ci["n_successful_subsamples"])

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print_section("Saving outputs")

    if len(subsample_results) == 0:
        raise ValueError("No subsampling results were produced.")

    ci_results_df = pd.DataFrame(ci_results)

    subsample_results_csv = (
        output_dir / "subsampling_draws_penalized_poisson.csv"
    )

    ci_results_csv = (
        output_dir / "subsampling_ci_penalized_poisson.csv"
    )

    save_table(subsample_results_df, subsample_results_csv)
    save_table(ci_results_df, ci_results_csv)

    print_section("Done")
    print("Outputs saved in:", output_dir)


if __name__ == "__main__":
    main()