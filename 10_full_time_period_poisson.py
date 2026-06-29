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
)


def main():
    root = PROJECT_ROOT
    config = load_config(root)

    processed_dir = root / config["paths"]["processed"]

    input_dir = root / "outputs" / "tables" / "penalized_poisson"
    output_dir = root / "outputs" / "tables" / "penalized_poisson_overall"
    output_dir.mkdir(parents=True, exist_ok=True)

    phrase_parameters_path = (
        input_dir / "phrase_parameters_penalized_poisson.parquet"
    )

    phrase_counts_path = (
        processed_dir / "phrase_counts_long.parquet"
    )

    phrase_parameters = pd.read_parquet(phrase_parameters_path)
    phrase_counts = pd.read_parquet(phrase_counts_path)

    print_section("Loaded data")
    print("Phrase parameters:", phrase_parameters.shape)
    print("Phrase counts:", phrase_counts.shape)

    phrase_parameters["congress"] = phrase_parameters["congress"].astype(int)
    phrase_counts["congress"] = phrase_counts["congress"].astype(int)

    # --------------------------------------------------------------
    # Count phrase usage by Congress.
    # This becomes the weight for overall aggregation.
    # --------------------------------------------------------------
    phrase_usage = (
        phrase_counts
        .groupby(["congress", "phrase"], as_index=False)
        .agg(
            phrase_count=("count", "sum"),
            n_speaker_sessions_using_phrase=("speaker_session_id", "nunique"),
        )
    )

    phrase_parameters = phrase_parameters.merge(
        phrase_usage,
        on=["congress", "phrase"],
        how="left",
    )

    phrase_parameters["phrase_count"] = (
        phrase_parameters["phrase_count"]
        .fillna(0)
        .astype(float)
    )

    phrase_parameters = phrase_parameters[
        phrase_parameters["phrase_count"] > 0
    ].copy()

    # --------------------------------------------------------------
    # Weighted overall phrase-level partisanship.
    # --------------------------------------------------------------
    phrase_parameters["weighted_phrase_partisanship"] = (
        phrase_parameters["phrase_partisanship"]
        * phrase_parameters["phrase_count"]
    )

    phrase_parameters["weighted_predicted_per_100k_republican"] = (
        phrase_parameters["predicted_per_100k_republican"]
        * phrase_parameters["phrase_count"]
    )

    phrase_parameters["weighted_predicted_per_100k_democrat"] = (
        phrase_parameters["predicted_per_100k_democrat"]
        * phrase_parameters["phrase_count"]
    )

    overall_phrases = (
        phrase_parameters
        .groupby("phrase", as_index=False)
        .agg(
            total_phrase_count=("phrase_count", "sum"),
            n_congresses_observed=("congress", "nunique"),
            n_speaker_sessions_using_phrase=(
                "n_speaker_sessions_using_phrase",
                "sum",
            ),
            phrase_partisanship_weighted_sum=(
                "weighted_phrase_partisanship",
                "sum",
            ),
            predicted_per_100k_republican_weighted_sum=(
                "weighted_predicted_per_100k_republican",
                "sum",
            ),
            predicted_per_100k_democrat_weighted_sum=(
                "weighted_predicted_per_100k_democrat",
                "sum",
            ),
        )
    )

    overall_phrases["phrase_partisanship"] = (
        overall_phrases["phrase_partisanship_weighted_sum"]
        / overall_phrases["total_phrase_count"]
    )

    overall_phrases["predicted_per_100k_republican"] = (
        overall_phrases["predicted_per_100k_republican_weighted_sum"]
        / overall_phrases["total_phrase_count"]
    )

    overall_phrases["predicted_per_100k_democrat"] = (
        overall_phrases["predicted_per_100k_democrat_weighted_sum"]
        / overall_phrases["total_phrase_count"]
    )

    overall_phrases["abs_phrase_partisanship"] = (
        overall_phrases["phrase_partisanship"].abs()
    )

    # --------------------------------------------------------------
    # Top phrases overall.
    # --------------------------------------------------------------
    top_n_phrases = config.get("penalized_poisson", {}).get(
        "top_n_phrases",
        50,
    )

    top_n_phrases_by_direction = config.get("penalized_poisson", {}).get(
        "top_n_phrases_by_direction",
        10,
    )

    top_partisan_phrases = (
        overall_phrases
        .sort_values("abs_phrase_partisanship", ascending=False)
        .head(top_n_phrases)
        .copy()
    )

    top_republican_phrases = (
        overall_phrases
        .sort_values("phrase_partisanship", ascending=False)
        .head(top_n_phrases_by_direction)
        .assign(direction="Republican")
    )

    top_democratic_phrases = (
        overall_phrases
        .sort_values("phrase_partisanship", ascending=True)
        .head(top_n_phrases_by_direction)
        .assign(direction="Democratic")
    )

    top_partisan_phrases_by_direction = pd.concat(
        [top_republican_phrases, top_democratic_phrases],
        ignore_index=True,
    )

    print_section("Top overall partisan phrases")
    print(top_partisan_phrases.head(20).to_string(index=False))

    print_section("Top overall partisan phrases by direction")
    print(top_partisan_phrases_by_direction.to_string(index=False))

    # --------------------------------------------------------------
    # Save outputs.
    # --------------------------------------------------------------
    save_table(
        overall_phrases,
        output_dir / "phrase_parameters_penalized_poisson_overall_aggregated.csv",
    )

    save_table(
        top_partisan_phrases,
        output_dir / "top_partisan_phrases_penalized_poisson_overall.csv",
    )

    save_table(
        top_partisan_phrases_by_direction,
        output_dir / "top_partisan_phrases_by_direction_penalized_poisson_overall.csv",
    )

    print_section("Done")
    print("Outputs saved in:", output_dir)


if __name__ == "__main__":
    main()