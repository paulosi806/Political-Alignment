# Build speaker-session bigram counts with Gentzkow Appendix-B phrase filtering

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import pandas as pd

from political_speech.utils import (
    load_config,
    load_stopwords,
    get_stemmer,
    congress_session_from_date,
    load_procedural_manual_phrase_sets,
    build_speech_level_phrase_counts,
    build_bigram_counts_for_speaker_sessions,
    classify_procedural_speeches,
    identify_cooccurrence_procedural_phrases,
    build_vocabulary_stats,
    build_appendix_b_exclusion_resources,
    add_appendix_b_filter_flags_to_vocabulary,
    apply_phrase_filters,
)


def main():
    root = PROJECT_ROOT
    config = load_config(root)

    processed_dir = root / config["paths"]["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    politicians_dir = root / config["paths"]["raw"] / "politicians"

    input_path = processed_dir / "speeches_clean.parquet"

    speaker_sessions_path = processed_dir / "speaker_sessions.parquet"
    vocabulary_path = processed_dir / "vocabulary_bigrams.parquet"
    phrase_counts_path = processed_dir / "phrase_counts_long.parquet"

    speech_scores_path = processed_dir / "speech_procedural_scores.parquet"
    cooccurrence_stats_path = processed_dir / "procedural_phrase_cooccurrence_stats.parquet"

    df = pd.read_parquet(input_path)

    print("Loaded:", input_path)
    print("Rows:", len(df))

    required_columns = [
        "date",
        "speaker_bioguide",
        "party",
        "text_clean",
        "n_tokens_clean",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")
    
    if "chamber_member" in df.columns:
        chamber_col = "chamber_member"
    elif "chamber" in df.columns:
        chamber_col = "chamber"
    else:
        raise KeyError(
            "Missing required chamber column: expected either "
            "'chamber_member' or 'chamber'."
        )

    df = df.copy()

    # ------------------------------------------------------------------
    # Congress/session variable
    # ------------------------------------------------------------------

    df["date"] = pd.to_datetime(df["date"])

    congress_session = (
        df["date"]
        .apply(congress_session_from_date)
        .apply(pd.Series)
    )

    df["congress"] = congress_session["congress"]
    df["session"] = congress_session["session"]

    # ------------------------------------------------------------------
    # Clean key columns
    # ------------------------------------------------------------------
    df["speaker_bioguide"] = (
        df["speaker_bioguide"]
        .astype("string")
        .str.strip()
    )

    df["party"] = (
        df["party"]
        .astype("string")
        .str.strip()
    )

    df["text_clean"] = (
        df["text_clean"]
        .astype("string")
        .fillna("")
        .str.strip()
    )

    df = df[
        (df["speaker_bioguide"].notna())
        & (df["speaker_bioguide"] != "")
        & (df["text_clean"] != "")
    ].copy()

    print("Rows after dropping missing speaker/text:", len(df))

    # ------------------------------------------------------------------
    # Speaker-session ID
    # ------------------------------------------------------------------

    df[chamber_col] = (
        df[chamber_col]
        .astype("string")
        .str.strip()
    )

    df["speaker_session_id"] = (
        df["speaker_bioguide"].astype(str)
        + "_"
        + df["congress"].astype(str)
        + "_"
        + df[chamber_col].astype(str)
    )

    # Speech ID for Appendix-B filtering
    df = df.reset_index(drop=True)
    df["speech_id"] = (
        "speech_"
        + df.index.astype(str)
    )

    print("Unique speaker-sessions:", df["speaker_session_id"].nunique())
    print("Unique speeches:", df["speech_id"].nunique())

    # ------------------------------------------------------------------
    # Speaker-session metadata table
    # ------------------------------------------------------------------
    metadata_cols = [
        "speaker_session_id",
        "speaker_bioguide",
        "congress",
        "party",
        chamber_col,
    ]

    optional_metadata_cols = [
        "full_name",
        "first_name",
        "last_name",
        "gender",
        "state",
        "district",
    ]

    for col in optional_metadata_cols:
        metadata_cols.append(col)

    speaker_sessions = (
        df.groupby(metadata_cols, dropna=False)
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            n_speeches=("text_clean", "size"),
            n_tokens_clean=("n_tokens_clean", "sum"),
        )
        .reset_index()
    )

    speaker_sessions.to_parquet(speaker_sessions_path, index=False)

    print("Saved speaker sessions to:", speaker_sessions_path)

    # ------------------------------------------------------------------
    # Build speech-level phrase counts
    # ------------------------------------------------------------------
    print("\nBuilding speech-level bigram counts...")

    speech_phrase_counts = build_speech_level_phrase_counts(df)

    print("Speech-level phrase rows:", len(speech_phrase_counts))
    print("Unique raw phrases:", speech_phrase_counts["phrase"].nunique())

    # ------------------------------------------------------------------
    # Load procedural manuals and build procedural phrase sets
    # ------------------------------------------------------------------
    procedural_config = config.get("procedural_language", {})

    roberts_file = procedural_config.get(
        "roberts_file",
        "data/raw/procedural_language/Roberts Rules of Order.txt",
    )

    riddick_file = procedural_config.get(
        "riddick_file",
        "data/raw/procedural_language/Riddick Senate Procedure_Appendix.pdf",
    )

    roberts_path = root / roberts_file
    riddick_path = root / riddick_file

    if not roberts_path.exists():
        raise FileNotFoundError(f"Roberts file not found: {roberts_path}")

    if not riddick_path.exists():
        raise FileNotFoundError(f"Riddick file not found: {riddick_path}")

    stopwords_path = root / "config" / "stopwords_snowball_english.txt"

    stopwords = load_stopwords(stopwords_path)
    stemmer = get_stemmer()

    print("\nLoading procedural language manuals...")
    print("Roberts:", roberts_path)
    print("Riddick:", riddick_path)

    roberts_phrases, riddick_phrases = load_procedural_manual_phrase_sets(
        roberts_path=roberts_path,
        riddick_path=riddick_path,
        stemmer=stemmer,
        stopwords=stopwords,
    )

    print("Roberts procedural phrases:", len(roberts_phrases))
    print("Riddick procedural phrases:", len(riddick_phrases))
    print(
        "Manual procedural phrases:",
        len(roberts_phrases | riddick_phrases),
    )

    # ------------------------------------------------------------------
    # Classify speeches according to 30 percent procedural threshold
    # ------------------------------------------------------------------
    print("\nClassifying procedural speeches...")

    speech_scores = classify_procedural_speeches(
        speech_phrase_counts=speech_phrase_counts,
        roberts_phrases=roberts_phrases,
        riddick_phrases=riddick_phrases,
        threshold=0.30,
    )

    print("Highly Robert speeches:", int(speech_scores["is_highly_robert_speech"].sum()))
    print("Highly Riddick speeches:", int(speech_scores["is_highly_riddick_speech"].sum()))
    print("Procedural speeches:", int(speech_scores["is_procedural_speech"].sum()))

    speech_scores.to_parquet(speech_scores_path, index=False)
    print("Saved speech procedural scores to:", speech_scores_path)

    # ------------------------------------------------------------------
    # Identify additional procedural phrases by co-occurrence rules
    # ------------------------------------------------------------------
    print("\nIdentifying co-occurrence procedural phrases...")

    cooccurrence_phrase_stats = identify_cooccurrence_procedural_phrases(
        speech_phrase_counts=speech_phrase_counts,
        speech_scores=speech_scores,
    )

    print(
        "Co-occurrence procedural phrases:",
        int(cooccurrence_phrase_stats["is_cooccurrence_procedural_phrase"].sum()),
    )

    cooccurrence_phrase_stats.to_parquet(
        cooccurrence_stats_path,
        index=False,
    )

    print("Saved co-occurrence stats to:", cooccurrence_stats_path)

    # ------------------------------------------------------------------
    # Aggregate to speaker-session phrase counts
    # ------------------------------------------------------------------
    print("\nAggregating to speaker-session phrase counts...")

    phrase_counts_raw = build_bigram_counts_for_speaker_sessions(
        speech_phrase_counts
    )

    print("Raw speaker-session phrase rows:", len(phrase_counts_raw))
    print("Raw unique phrases:", phrase_counts_raw["phrase"].nunique())

    # ------------------------------------------------------------------
    # Build vocabulary stats
    # ------------------------------------------------------------------
    vocab = build_vocabulary_stats(phrase_counts_raw)

    print("\nVocabulary before Appendix-B filtering:", len(vocab))

    # ------------------------------------------------------------------
    # Appendix-B exclusion resources
    # ------------------------------------------------------------------

    exclusion_resources = build_appendix_b_exclusion_resources(
        df=df,
        stemmer=stemmer,
        stopwords=stopwords,
        politicians_dir=politicians_dir,
    )

    print("Member surname patterns:", len(exclusion_resources["surname_patterns"]))
    print("State/month patterns:", len(exclusion_resources["state_month_patterns"]))
    print("Table 4 stems:", len(exclusion_resources["table4_stems"]))

    vocab = add_appendix_b_filter_flags_to_vocabulary(
        vocabulary=vocab,
        roberts_phrases=roberts_phrases,
        riddick_phrases=riddick_phrases,
        cooccurrence_phrase_stats=cooccurrence_phrase_stats,
        exclusion_resources=exclusion_resources,
    )

    # ------------------------------------------------------------------
    # Frequency filters
    # ------------------------------------------------------------------
    phrase_filters = config.get("phrase_filters", {})

    min_total_count = phrase_filters.get("min_total_count", 1)
    min_speaker_sessions = phrase_filters.get("min_speaker_sessions", 1)
    min_count_in_one_congress = phrase_filters.get(
        "min_count_in_one_congress",
        1,
    )

    print("\nPhrase frequency filters:")
    print("min_total_count:", min_total_count)
    print("min_speaker_sessions:", min_speaker_sessions)
    print("min_count_in_one_congress:", min_count_in_one_congress)

    vocab = apply_phrase_filters(
        vocab=vocab,
        min_total_count=min_total_count,
        min_speaker_sessions=min_speaker_sessions,
        min_count_in_one_congress=min_count_in_one_congress,
    )

    kept_vocab = vocab[vocab["keep_phrase"]].copy()

    print("\nVocabulary summary:")
    print("Total phrases:", len(vocab))
    print("Pass frequency filters:", int(vocab["passes_frequency_filters"].sum()))
    print("Excluded by Appendix B:", int(vocab["excluded_by_appendix_b"].sum()))
    print("Kept phrases:", len(kept_vocab))

    print("\nAppendix-B exclusion counts:")
    exclusion_cols = [
        "excluded_procedural_manual_or_cooccurrence",
        "excluded_number_or_symbol",
        "excluded_short_phrase",
        "excluded_one_letter_word",
        "excluded_table4_word",
        "excluded_member_surname",
        "excluded_state_or_month",
    ]

    for col in exclusion_cols:
        print(col + ":", int(vocab[col].sum()))

    # ------------------------------------------------------------------
    # Filter phrase counts
    # ------------------------------------------------------------------
    phrase_counts_filtered = phrase_counts_raw.merge(
        kept_vocab[["phrase"]],
        on="phrase",
        how="inner",
    )

    print("\nFiltered speaker-session phrase rows:", len(phrase_counts_filtered))
    print("Filtered unique phrases:", phrase_counts_filtered["phrase"].nunique())
    print("Filtered total bigram count:", int(phrase_counts_filtered["count"].sum()))

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    vocab.to_parquet(vocabulary_path, index=False)

    phrase_counts_filtered.to_parquet(
        phrase_counts_path,
        index=False,
    )

    print("\nSaved vocabulary to:", vocabulary_path)
    print("Saved filtered phrase counts to:", phrase_counts_path)

    if len(phrase_counts_filtered) > 0:
        print("\nPreview filtered phrase counts:")
        print(
            phrase_counts_filtered
            .head(20)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()