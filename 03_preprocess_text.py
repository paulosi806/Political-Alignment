# Preprocess congressional speeches

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
    clean_text_to_tokens,
)


def main():
    root = PROJECT_ROOT
    config = load_config(root)

    processed_dir = root / config["paths"]["processed"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    input_path = processed_dir / "speeches_with_members.parquet"
    output_path = processed_dir / "speeches_clean.parquet"

    stopwords_path = root / "config" / "stopwords_snowball_english.txt"

    df = pd.read_parquet(input_path)

    print("Loaded:", input_path)
    print("Rows before preprocessing:", len(df))

    # Keep only Democrats and Republicans
    df["party"] = df["party"].astype("string").str.strip()

    party_mask = df["party"].isin(["Democrat", "Republican"])

    print("Rows not Democrat/Republican:", (~party_mask).sum())

    df = df.loc[party_mask].copy()

    # Drop empty text rows
    df["text"] = df["text"].astype("string")

    missing_text = (
        df["text"].isna()
        | (df["text"].str.strip() == "")
    )

    print("Rows with missing/empty text:", missing_text.sum())

    df = df.loc[~missing_text].copy()

    # Load preprocessing resources
    stopwords = load_stopwords(stopwords_path)
    stemmer = get_stemmer()

    print("Stopwords loaded:", len(stopwords))

    # Clean text
    df["tokens_clean"] = df["text"].apply(
        lambda text: clean_text_to_tokens(
            text=text,
            stemmer=stemmer,
            stopwords=stopwords,
        )
    )

    df["text_clean"] = df["tokens_clean"].apply(
        lambda tokens: " ".join(tokens)
    )

    df["n_tokens_clean"] = df["tokens_clean"].apply(len)

    # Drop rows with no tokens left
    no_tokens = df["n_tokens_clean"] == 0

    print("Rows with zero clean tokens:", no_tokens.sum())

    df = df.loc[~no_tokens].copy()

    print("Rows after preprocessing:", len(df))
    print("Unique speakers:", df["speaker_bioguide"].nunique())

    print("\nParties:")
    print(df["party"].value_counts(dropna=False))

    if len(df) > 0:
        example = df.iloc[0]

        print("\nExample before/after:")
        print("Speaker:", example["speaker"])
        print("Party:", example["party"])

        print("\nOriginal text:")
        print(example["text"][:500])

        print("\nClean text:")
        print(example["text_clean"][:500])

    df.to_parquet(output_path, index=False)

    print(f"\nSaved cleaned speeches to: {output_path}")


if __name__ == "__main__":
    main()