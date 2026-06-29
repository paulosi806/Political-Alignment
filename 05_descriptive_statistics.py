# Descriptive statistics for phrase-count dataset

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import pandas as pd

from political_speech.utils import load_config

from political_speech.utils_analysis import (
    print_section,
    save_table,
    make_overall_summary,
    summarize_speaker_sessions_by_congress,
    summarize_speaker_sessions_by_congress_party,
    summarize_phrase_counts_by_congress,
    summarize_phrase_counts_by_congress_party,
    get_top_phrases_overall,
    get_top_phrases_by_party,
    summarize_vocabulary_filter,
    check_congress_party_coverage,
)


def main():
    root = PROJECT_ROOT
    config = load_config(root)

    processed_dir = root / config["paths"]["processed"]
    output_dir = root / "outputs" / "tables" / "descriptive"
    output_dir.mkdir(parents=True, exist_ok=True)

    speaker_sessions_path = processed_dir / "speaker_sessions.parquet"
    vocabulary_path = processed_dir / "vocabulary_bigrams.parquet"
    phrase_counts_path = processed_dir / "phrase_counts_long.parquet"

    speaker_sessions = pd.read_parquet(speaker_sessions_path)
    vocabulary = pd.read_parquet(vocabulary_path)
    phrase_counts = pd.read_parquet(phrase_counts_path)

    print_section("Loaded data")
    print("Speaker sessions:", speaker_sessions.shape)
    print("Vocabulary:", vocabulary.shape)
    print("Phrase counts:", phrase_counts.shape)

    # Overall summary
    print_section("Overall summary")

    overall_summary = make_overall_summary(
        speaker_sessions=speaker_sessions,
        vocabulary=vocabulary,
        phrase_counts=phrase_counts,
    )

    print(overall_summary.to_string(index=False))
    save_table(overall_summary, output_dir / "overall_summary.csv")

    # Speaker-sessions by Congress
    print_section("Speaker-sessions by Congress")

    speaker_sessions_by_congress = summarize_speaker_sessions_by_congress(
        speaker_sessions
    )

    print(speaker_sessions_by_congress.to_string(index=False))
    save_table(
        speaker_sessions_by_congress,
        output_dir / "speaker_sessions_by_congress.csv",
    )

    # Speaker-sessions by Congress and Party
    print_section("Speaker-sessions by Congress and Party")

    speaker_sessions_by_congress_party = (
        summarize_speaker_sessions_by_congress_party(speaker_sessions)
    )

    print(speaker_sessions_by_congress_party.to_string(index=False))
    save_table(
        speaker_sessions_by_congress_party,
        output_dir / "speaker_sessions_by_congress_party.csv",
    )

    # Phrase counts by Congress
    print_section("Phrase counts by Congress")

    phrase_counts_by_congress = summarize_phrase_counts_by_congress(
        phrase_counts
    )

    print(phrase_counts_by_congress.to_string(index=False))
    save_table(
        phrase_counts_by_congress,
        output_dir / "phrase_counts_by_congress.csv",
    )

    # Phrase counts by Congress and Party
    print_section("Phrase counts by Congress and Party")

    phrase_counts_by_congress_party = (
        summarize_phrase_counts_by_congress_party(phrase_counts)
    )

    print(phrase_counts_by_congress_party.to_string(index=False))
    save_table(
        phrase_counts_by_congress_party,
        output_dir / "phrase_counts_by_congress_party.csv",
    )

    # Top bigrams overall
    print_section("Top bigrams overall")

    top_phrases_overall = get_top_phrases_overall(
        phrase_counts=phrase_counts,
        n=100,
    )

    print(top_phrases_overall.head(30).to_string(index=False))
    save_table(
        top_phrases_overall,
        output_dir / "top_phrases_overall.csv",
    )

    # Top bigrams by party
    print_section("Top bigrams by Party")

    top_phrases_by_party = get_top_phrases_by_party(
        phrase_counts=phrase_counts,
        n=100,
    )

    print(top_phrases_by_party.head(60).to_string(index=False))
    save_table(
        top_phrases_by_party,
        output_dir / "top_phrases_by_party.csv",
    )

    # Vocabulary filter summary
    print_section("Vocabulary filter summary")

    vocabulary_filter_summary = summarize_vocabulary_filter(vocabulary)

    print(vocabulary_filter_summary.to_string(index=False))
    save_table(
        vocabulary_filter_summary,
        output_dir / "vocabulary_filter_summary.csv",
    )

    # Potential data issues
    print_section("Potential data issues")

    congress_party_check = check_congress_party_coverage(
        phrase_counts_by_congress_party
    )

    print(congress_party_check.to_string(index=False))
    save_table(
        congress_party_check,
        output_dir / "congress_party_data_check.csv",
    )

    problem_congresses = congress_party_check[
        ~congress_party_check["has_both_parties"]
    ].copy()

    if len(problem_congresses) > 0:
        print("\nWarning: Some Congresses do not have counts for both parties:")
        print(problem_congresses.to_string(index=False))
    else:
        print("\nAll Congresses have phrase counts for both parties.")

    print_section("Done")
    print("Descriptive tables saved in:", output_dir)


if __name__ == "__main__":
    main()