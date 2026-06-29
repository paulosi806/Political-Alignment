# Match speeches to legislators by bioguide and date

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import pandas as pd

from political_speech.utils import (
    load_config,
    build_legislator_terms,
    add_social_media,
    match_speeches_to_terms,
    remove_speaker_prefix_from_text,
)


def main():
    root = PROJECT_ROOT
    config = load_config(root)

    interim_dir = root / config["paths"]["interim"]
    processed_dir = root / config["paths"]["processed"]
    politicians_dir = root / "data" / "raw" / "politicians"
    logs_dir = root / config["paths"]["logs"]

    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    speeches_path = interim_dir / "speeches_raw.parquet"

    terms_output_path = processed_dir / "legislator_terms.parquet"
    matched_output_path = processed_dir / "speeches_with_members.parquet"
    unmatched_output_path = logs_dir / "unmatched_speeches_with_bioguide.csv"

    speeches_df = pd.read_parquet(speeches_path)

    print("Raw speeches:", len(speeches_df))

    # Drop speeches without Bioguide ID
    missing_bioguide = (
        speeches_df["speaker_bioguide"].isna()
        | (speeches_df["speaker_bioguide"].astype(str).str.strip() == "")
    )

    print("Speeches without speaker_bioguide:", missing_bioguide.sum())

    if missing_bioguide.sum() > 0:
        print("\nTop speakers without Bioguide:")
        print(
            speeches_df.loc[missing_bioguide, "speaker"]
            .value_counts(dropna=False)
            .head(30)
            .to_string()
        )

    speeches_df = speeches_df.loc[~missing_bioguide].copy()

    speeches_df["speaker_bioguide"] = (
        speeches_df["speaker_bioguide"]
        .astype("string")
        .str.strip()
    )

    # Remove speaker name from beginning of text
    speeches_df["text"] = speeches_df.apply(
        lambda row: remove_speaker_prefix_from_text(
            row["text"],
            row["speaker"]
        ),
        axis=1
    )

    print("Speeches after dropping missing Bioguide:", len(speeches_df))
    print(
        "Unique speaker_bioguide in speeches:",
        speeches_df["speaker_bioguide"].nunique()
    )

    terms_df = build_legislator_terms(politicians_dir)
    terms_df = add_social_media(terms_df, politicians_dir)

    print("Legislator term rows:", len(terms_df))
    print(
        "Unique bioguide in legislator terms:",
        terms_df["bioguide"].nunique()
    )

    terms_df.to_parquet(terms_output_path, index=False)

    speeches_with_members = match_speeches_to_terms(
        speeches_df=speeches_df,
        terms_df=terms_df
    )

    print("Matched speeches:", len(speeches_with_members))
    print(
        "Unique matched speakers:",
        speeches_with_members["speaker_bioguide"].nunique()
    )

    unmatched = speeches_df[
        speeches_df["speaker_bioguide"].notna()
        & ~speeches_df["speaker_bioguide"].isin(
            speeches_with_members["speaker_bioguide"]
        )
    ].copy()

    print("Speeches with bioguide but no matched term:", len(unmatched))

    if len(unmatched) > 0:
        print("\nTop unmatched speakers:")
        print(
            unmatched[["speaker", "speaker_bioguide"]]
            .drop_duplicates()
            .head(30)
            .to_string(index=False)
        )

    if len(unmatched) > 0:
        unmatched.to_csv(unmatched_output_path, index=False)
        print(f"Saved unmatched speeches to: {unmatched_output_path}")

    speeches_with_members.to_parquet(
        matched_output_path,
        index=False
    )

    print(f"\nSaved legislator terms to: {terms_output_path}")
    print(f"Saved matched speeches to: {matched_output_path}")


if __name__ == "__main__":
    main()