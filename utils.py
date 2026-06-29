# Utils package for political speech

from pathlib import Path
from datetime import datetime
import json
import re
import string

import pandas as pd
import yaml


def load_config(root: Path | None = None) -> dict:
    """
    Load config/config.yaml from the project root.
    """
    config_path = root / "config" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_date(date_string: str, date_format: str = "%Y-%m-%d") -> datetime:
    """
    Parse a date string into a datetime object.
    """
    return datetime.strptime(date_string, date_format)


def read_json(path: Path) -> dict | list:
    """
    Read a JSON file and return its contents.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_date_from_crec_folder(folder_name: str) -> str | None:
    """
    Extract YYYY-MM-DD from folder names like CREC-2017-01-04.
    """
    match = re.match(r"CREC-(\d{4}-\d{2}-\d{2})$", folder_name)

    if match:
        return match.group(1)

    return None


def extract_speeches_from_json(data: dict) -> list[dict]:
    """
    Extract all speech entries from one Congressional Record JSON object.

    Returns one row per speech.
    """
    header = data.get("header", {})
    content = data.get("content", [])

    doc_id = data.get("id")
    doc_title = data.get("doc_title")
    title = data.get("title")

    rows = []

    for item in content:
        if not isinstance(item, dict):
            continue

        if item.get("kind") != "speech":
            continue

        text = item.get("text")

        if text is None or str(text).strip() == "":
            continue

        row = {
            # document-level metadata
            "doc_id": doc_id,
            "doc_title": doc_title,
            "title": title,

            # header metadata
            "volume": header.get("vol"),
            "number": header.get("num"),
            "weekday": header.get("wkday"),
            "month": header.get("month"),
            "day": header.get("day"),
            "year": header.get("year"),
            "chamber": header.get("chamber"),
            "pages": header.get("pages"),
            "extension": header.get("extension"),

            # speech-level content
            "kind": item.get("kind"),
            "speaker": item.get("speaker"),
            "speaker_bioguide": item.get("speaker_bioguide"),
            "text": text,
            "turn": item.get("turn"),
            "itemno": item.get("itemno"),
        }

        rows.append(row)

    return rows


def should_keep_by_chamber(row: dict, chamber_setting: str) -> bool:
    """
    Keep a row only if it has chamber information and matches the selected chamber.

    chamber_setting can be:
    - "House"
    - "Senate"
    - "both"
    """
    chamber_setting = chamber_setting.lower().strip()

    chamber_found = row.get("chamber")

    # Drop row if chamber information is missing
    if chamber_found is None or str(chamber_found).strip() == "":
        return False

    chamber_found = str(chamber_found).lower().strip()

    # Keep all rows with valid chamber information
    if chamber_setting == "both":
        return chamber_found in {"house", "senate"}

    # Keep only selected chamber
    return chamber_found == chamber_setting


def build_legislator_terms(politicians_dir: Path) -> pd.DataFrame:
    """
    Build one row per legislator term from legislators-current.json
    and legislators-historical.json.
    """
    current_file = politicians_dir / "legislators-current.json"
    historical_file = politicians_dir / "legislators-historical.json"

    current = read_json(current_file)
    historical = read_json(historical_file)

    all_legislators = current + historical

    rows = []

    for person in all_legislators:
        name = person.get("name", {})
        ids = person.get("id", {})
        bio = person.get("bio", {})

        first_name = name.get("first")
        last_name = name.get("last")
        official_full = name.get("official_full")

        if official_full:
            full_name = official_full
        else:
            full_name = " ".join(
                x for x in [first_name, last_name]
                if x is not None
            )

        bioguide = ids.get("bioguide")
        gender = bio.get("gender")

        for term in person.get("terms", []):
            term_type = term.get("type")

            if term_type == "rep":
                chamber = "House"
            elif term_type == "sen":
                chamber = "Senate"
            else:
                chamber = term_type

            rows.append({
                "bioguide": bioguide,
                "full_name": full_name,
                "first_name": first_name,
                "last_name": last_name,
                "party": term.get("party"),
                "state": term.get("state"),
                "district": term.get("district"),
                "chamber_member": chamber,
                "term_start": term.get("start"),
                "term_end": term.get("end"),
                "gender": gender,
            })

    terms_df = pd.DataFrame(rows)

    terms_df = terms_df[terms_df["bioguide"].notna()].copy()

    terms_df["term_start"] = pd.to_datetime(terms_df["term_start"])
    terms_df["term_end"] = pd.to_datetime(terms_df["term_end"])

    return terms_df


def add_social_media(
    terms_df: pd.DataFrame,
    politicians_dir: Path
) -> pd.DataFrame:
    """
    Add Twitter information if legislators-social-media.json exists.
    """
    social_file = politicians_dir / "legislators-social-media.json"

    if not social_file.exists():
        return terms_df

    social_media = read_json(social_file)

    social_rows = []

    for person in social_media:
        bioguide = person.get("id", {}).get("bioguide")
        social = person.get("social", {})

        social_rows.append({
            "bioguide": bioguide,
            "twitter": social.get("twitter"),
            "twitter_id": social.get("twitter_id"),
        })

    social_df = pd.DataFrame(social_rows)

    social_df = social_df.drop_duplicates(
        subset="bioguide",
        keep="first"
    )

    terms_df = terms_df.merge(
        social_df,
        on="bioguide",
        how="left"
    )

    return terms_df


def match_speeches_to_terms(
    speeches_df: pd.DataFrame,
    terms_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Match speeches to legislator terms by:
    - speaker_bioguide == bioguide
    - speech date between term_start and term_end
    """
    speeches = speeches_df.copy()

    speeches["speech_date"] = pd.to_datetime(speeches["date"])
    speeches["speaker_bioguide"] = (
        speeches["speaker_bioguide"]
        .astype("string")
        .str.strip()
    )

    terms = terms_df.copy()
    terms["bioguide"] = (
        terms["bioguide"]
        .astype("string")
        .str.strip()
    )

    merged = speeches.merge(
        terms,
        left_on="speaker_bioguide",
        right_on="bioguide",
        how="left"
    )

    matched = merged[
        (merged["speech_date"] >= merged["term_start"])
        & (merged["speech_date"] <= merged["term_end"])
    ].copy()

    return matched

def remove_speaker_prefix_from_text(text: str, speaker: str) -> str:
    """
    Remove speaker name from the beginning of a speech text.

    Example:
    speaker = "Mr. HILL"
    text = "Mr. HILL. Mr. Speaker, I rise today..."
    result = "Mr. Speaker, I rise today..."
    """
    if text is None or speaker is None:
        return text

    text = str(text).strip()
    speaker = str(speaker).strip()

    if text == "" or speaker == "":
        return text

    # Case 1: exact speaker prefix followed by a period
    prefix = speaker + "."

    if text.startswith(prefix):
        return text[len(prefix):].strip()

    # Case 2: exact speaker prefix without period
    if text.startswith(speaker):
        return text[len(speaker):].strip()

    return text

def load_stopwords(path: Path) -> set[str]:
    """
    Load stopwords from a text file.

    One stopword per line.
    Empty lines and lines starting with '#' are ignored.
    """
    with open(path, "r", encoding="utf-8") as f:
        return {
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        }


def get_stemmer():
    """
    Return Porter2/Snowball English stemmer.
    """
    try:
        import snowballstemmer
    except ImportError as exc:
        raise ImportError(
            "Package 'snowballstemmer' is missing. "
            "Install it with: python -m pip install snowballstemmer"
        ) from exc

    return snowballstemmer.stemmer("english")


def remove_parenthetical_insertions(text: str) -> str:
    """
    Remove parenthetical insertions such as '(Applause.)' or '(Laughter.)'.

    This approximates Gentzkow et al.'s step of removing non-spoken
    parenthetical insertions.
    """
    return re.sub(r"\([^)]*\)", " ", text)


def clean_text_to_tokens(
    text: str,
    stemmer,
    stopwords: set[str],
) -> list[str]:
    """
    Clean one speech text and return stemmed tokens.

    Steps:
    1. Lowercase
    2. Remove parenthetical insertions
    3. Delete hyphens and apostrophes
    4. Replace other punctuation with spaces
    5. Split into tokens
    6. Keep alphabetic tokens only
    7. Remove stopwords
    8. Apply Porter2/Snowball stemming
    """
    if text is None:
        return []

    text = str(text).lower()

    # Remove parenthetical insertions
    text = remove_parenthetical_insertions(text)

    # Delete hyphens and apostrophes
    text = re.sub(r"[-‐-‒–—]", "", text)
    text = re.sub(r"[']", "", text)

    # Replace all other punctuation with spaces
    punctuation_without_hyphen_apostrophe = (
        string.punctuation
        .replace("-", "")
        .replace("'", "")
    )

    translation_table = str.maketrans(
        {char: " " for char in punctuation_without_hyphen_apostrophe}
    )

    text = text.translate(translation_table)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if text == "":
        return []

    tokens = text.split()

    # Keep alphabetic tokens only
    tokens = [token for token in tokens if token.isalpha()]

    # Remove stopwords before stemming
    tokens = [token for token in tokens if token not in stopwords]

    # Porter2/Snowball stemming
    tokens = stemmer.stemWords(tokens)

    return tokens

def congress_session_from_date(date_value) -> dict:
    """
    Assign a date to Congress and annual Session using explicit date ranges.

    The intervals are defined as half-open:
    begin_date <= date < end_date

    Returns:
    {
        "congress": int,
        "session": int
    }
    """
    import pandas as pd

    date = pd.to_datetime(date_value)

    congress_sessions = [
        # 114th Congress
        {"congress": 114, "session": 1, "begin": "2015-01-06", "end": "2015-12-19"},
        {"congress": 114, "session": 2, "begin": "2016-01-04", "end": "2017-01-03"},

        # 115th Congress
        {"congress": 115, "session": 1, "begin": "2017-01-03", "end": "2018-01-03"},
        {"congress": 115, "session": 2, "begin": "2018-01-03", "end": "2019-01-03"},

        # 116th Congress
        {"congress": 116, "session": 1, "begin": "2019-01-03", "end": "2020-01-03"},
        {"congress": 116, "session": 2, "begin": "2020-01-03", "end": "2021-01-03"},

        # 117th Congress
        {"congress": 117, "session": 1, "begin": "2021-01-03", "end": "2022-01-03"},
        {"congress": 117, "session": 2, "begin": "2022-01-03", "end": "2023-01-03"},

        # 118th Congress
        {"congress": 118, "session": 1, "begin": "2023-01-03", "end": "2024-01-03"},
        {"congress": 118, "session": 2, "begin": "2024-01-03", "end": "2025-01-03"},

        # 119th Congress
        {"congress": 119, "session": 1, "begin": "2025-01-03", "end": "2026-01-03"},
        {"congress": 119, "session": 2, "begin": "2026-01-03", "end": "2027-01-03"},
    ]

    for item in congress_sessions:
        begin = pd.to_datetime(item["begin"])
        end = pd.to_datetime(item["end"])

        if begin <= date < end:
            return {
                "congress": item["congress"],
                "session": item["session"],
            }

    raise ValueError(f"Date is outside defined Congress-session ranges: {date_value}")


def make_bigrams(tokens: list[str]) -> list[str]:
    """
    Convert a token sequence into adjacent two-word phrases.
    """
    return [
        tokens[i] + " " + tokens[i + 1]
        for i in range(len(tokens) - 1)
    ]


def read_text_document(path: Path) -> str:
    """
    Read a plain text or PDF document and return text.
    """
    suffix = path.suffix.lower()

    if suffix == ".txt":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "Package 'pypdf' is missing. "
                "Install it with: python -m pip install pypdf"
            ) from exc

        reader = PdfReader(str(path))
        pages = []

        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

        return "\n".join(pages)

    raise ValueError(f"Unsupported document type: {path}")


def load_procedural_manual_phrase_sets(
    roberts_path: Path,
    riddick_path: Path,
    stemmer,
    stopwords: set[str],
) -> tuple[set[str], set[str]]:
    """
    Load Robert's Rules and Riddick Senate Procedure, clean them using the
    same text pipeline as speeches, and return bigram sets.
    """
    roberts_text = read_text_document(roberts_path)
    riddick_text = read_text_document(riddick_path)

    roberts_tokens = clean_text_to_tokens(
        text=roberts_text,
        stemmer=stemmer,
        stopwords=stopwords,
    )

    riddick_tokens = clean_text_to_tokens(
        text=riddick_text,
        stemmer=stemmer,
        stopwords=stopwords,
    )

    roberts_phrases = set(make_bigrams(roberts_tokens))
    riddick_phrases = set(make_bigrams(riddick_tokens))

    return roberts_phrases, riddick_phrases


def build_speech_level_phrase_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build bigram counts at the individual speech level.

    Required columns:
    - speech_id
    - speaker_session_id
    - speaker_bioguide
    - congress
    - party
    - text_clean

    Returns:
    speech_id | speaker_session_id | speaker_bioguide | congress | party | phrase | count
    """
    from collections import Counter

    records = []

    for row in df.itertuples(index=False):
        text_clean = getattr(row, "text_clean")
        tokens = str(text_clean).split()

        if len(tokens) < 2:
            continue

        bigram_counts = Counter(make_bigrams(tokens))

        base_record = {
            "speech_id": getattr(row, "speech_id"),
            "speaker_session_id": getattr(row, "speaker_session_id"),
            "speaker_bioguide": getattr(row, "speaker_bioguide"),
            "congress": getattr(row, "congress"),
            "party": getattr(row, "party"),
        }

        for phrase, count in bigram_counts.items():
            records.append({
                **base_record,
                "phrase": phrase,
                "count": count,
            })

    if len(records) == 0:
        raise ValueError("No speech-level bigrams were created.")

    speech_phrase_counts = pd.DataFrame(records)

    if speech_phrase_counts.duplicated(subset=["speech_id", "phrase"]).any():
        raise ValueError("Duplicate speech_id × phrase rows found.")

    return speech_phrase_counts


def build_bigram_counts_for_speaker_sessions(
    speech_phrase_counts: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate speech-level phrase counts to speaker-session phrase counts.
    """
    phrase_counts = (
        speech_phrase_counts
        .groupby(
            [
                "speaker_session_id",
                "speaker_bioguide",
                "congress",
                "party",
                "phrase",
            ],
            as_index=False,
        )
        .agg(count=("count", "sum"))
    )

    return phrase_counts


def classify_procedural_speeches(
    speech_phrase_counts: pd.DataFrame,
    roberts_phrases: set[str],
    riddick_phrases: set[str],
    threshold: float = 0.30,
) -> pd.DataFrame:
    """
    Classify speeches as highly Robert, highly Riddick, and procedural.

    Following Appendix B:
    - highly Robert: Robert phrases >= 30 percent of all bigrams
    - highly Riddick: Riddick phrases >= 30 percent of all bigrams
    - procedural: phrases in either manual >= 30 percent of all bigrams
    """
    df = speech_phrase_counts.copy()

    df["is_robert_phrase"] = df["phrase"].isin(roberts_phrases)
    df["is_riddick_phrase"] = df["phrase"].isin(riddick_phrases)
    df["is_manual_procedural_phrase"] = (
        df["is_robert_phrase"] | df["is_riddick_phrase"]
    )

    df["robert_count"] = df["count"].where(df["is_robert_phrase"], 0)
    df["riddick_count"] = df["count"].where(df["is_riddick_phrase"], 0)
    df["manual_procedural_count"] = df["count"].where(
        df["is_manual_procedural_phrase"],
        0,
    )

    speech_scores = (
        df.groupby(["speech_id", "congress"], as_index=False)
        .agg(
            total_bigrams=("count", "sum"),
            robert_bigrams=("robert_count", "sum"),
            riddick_bigrams=("riddick_count", "sum"),
            manual_procedural_bigrams=("manual_procedural_count", "sum"),
        )
    )

    speech_scores["robert_share"] = (
        speech_scores["robert_bigrams"]
        / speech_scores["total_bigrams"]
    )

    speech_scores["riddick_share"] = (
        speech_scores["riddick_bigrams"]
        / speech_scores["total_bigrams"]
    )

    speech_scores["manual_procedural_share"] = (
        speech_scores["manual_procedural_bigrams"]
        / speech_scores["total_bigrams"]
    )

    speech_scores["is_highly_robert_speech"] = (
        speech_scores["robert_share"] >= threshold
    )

    speech_scores["is_highly_riddick_speech"] = (
        speech_scores["riddick_share"] >= threshold
    )

    speech_scores["is_procedural_speech"] = (
        speech_scores["manual_procedural_share"] >= threshold
    )

    return speech_scores


def identify_cooccurrence_procedural_phrases(
    speech_phrase_counts: pd.DataFrame,
    speech_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify additional procedural phrases using the co-occurrence rules
    described in Online Appendix B.

    Returns one row per phrase with procedural-rule flags and diagnostics.
    """
    presence = (
        speech_phrase_counts[
            ["speech_id", "congress", "phrase", "count"]
        ]
        .copy()
    )

    total_counts = (
        presence
        .groupby("phrase", as_index=False)
        .agg(total_count=("count", "sum"))
    )

    presence = presence.drop(columns=["count"]).drop_duplicates()

    presence = presence.merge(
        speech_scores[
            [
                "speech_id",
                "robert_share",
                "riddick_share",
                "is_highly_robert_speech",
                "is_highly_riddick_speech",
                "is_procedural_speech",
            ]
        ],
        on="speech_id",
        how="left",
    )

    phrase_stats = (
        presence
        .groupby("phrase", as_index=False)
        .agg(
            n_speeches_containing=("speech_id", "nunique"),
            n_congresses=("congress", "nunique"),
            n_procedural_speeches=(
                "is_procedural_speech",
                "sum",
            ),
            n_highly_robert_speeches=(
                "is_highly_robert_speech",
                "sum",
            ),
            n_highly_riddick_speeches=(
                "is_highly_riddick_speech",
                "sum",
            ),
            avg_robert_share_in_speeches=(
                "robert_share",
                "mean",
            ),
            avg_riddick_share_in_speeches=(
                "riddick_share",
                "mean",
            ),
        )
    )

    procedural_congresses = (
        presence[presence["is_procedural_speech"]]
        .groupby("phrase", as_index=False)
        .agg(n_procedural_congresses=("congress", "nunique"))
    )

    highly_robert_congresses = (
        presence[presence["is_highly_robert_speech"]]
        .groupby("phrase", as_index=False)
        .agg(n_highly_robert_congresses=("congress", "nunique"))
    )

    highly_riddick_congresses = (
        presence[presence["is_highly_riddick_speech"]]
        .groupby("phrase", as_index=False)
        .agg(n_highly_riddick_congresses=("congress", "nunique"))
    )

    phrase_stats = phrase_stats.merge(
        total_counts,
        on="phrase",
        how="left",
    )

    phrase_stats = phrase_stats.merge(
        procedural_congresses,
        on="phrase",
        how="left",
    )

    phrase_stats = phrase_stats.merge(
        highly_robert_congresses,
        on="phrase",
        how="left",
    )

    phrase_stats = phrase_stats.merge(
        highly_riddick_congresses,
        on="phrase",
        how="left",
    )

    fill_zero_cols = [
        "n_procedural_congresses",
        "n_highly_robert_congresses",
        "n_highly_riddick_congresses",
    ]

    for col in fill_zero_cols:
        phrase_stats[col] = phrase_stats[col].fillna(0)

    phrase_stats["share_highly_robert_speeches"] = (
        phrase_stats["n_highly_robert_speeches"]
        / phrase_stats["n_speeches_containing"]
    )

    phrase_stats["share_highly_riddick_speeches"] = (
        phrase_stats["n_highly_riddick_speeches"]
        / phrase_stats["n_speeches_containing"]
    )

    # ------------------------------------------------------------------
    # Appendix B, first co-occurrence rule
    # ------------------------------------------------------------------

    base_procedural = (
        (phrase_stats["n_procedural_speeches"] >= 2)
        & (phrase_stats["n_procedural_congresses"] >= 2)
    )

    first_rule_robert_from_procedural = (
        base_procedural
        & (
            (
                (phrase_stats["n_highly_robert_speeches"] > 433)
                & (phrase_stats["share_highly_robert_speeches"] >= 0.0175)
            )
            | (
                (phrase_stats["n_highly_robert_speeches"] > 8)
                & (phrase_stats["share_highly_robert_speeches"] >= 0.075)
            )
            | (
                (phrase_stats["n_highly_robert_speeches"] > 4)
                & (phrase_stats["share_highly_robert_speeches"] > 0.30)
            )
        )
    )

    base_highly_robert = (
        (phrase_stats["n_highly_robert_speeches"] >= 2)
        & (phrase_stats["n_highly_robert_congresses"] >= 4)
    )

    first_rule_robert = (
        base_highly_robert
        & (
            (
                (phrase_stats["n_highly_robert_speeches"] > 167)
                & (phrase_stats["share_highly_robert_speeches"] >= 0.01)
            )
            | (
                (phrase_stats["n_highly_robert_speeches"] > 8)
                & (phrase_stats["share_highly_robert_speeches"] >= 0.05)
            )
            | (
                (phrase_stats["n_highly_robert_speeches"] > 4)
                & (phrase_stats["share_highly_robert_speeches"] >= 0.20)
            )
        )
    )

    base_highly_riddick = (
        (phrase_stats["n_highly_riddick_speeches"] >= 2)
        & (phrase_stats["n_highly_riddick_congresses"] >= 4)
    )

    first_rule_riddick = (
        base_highly_riddick
        & (
            (
                (phrase_stats["n_highly_riddick_speeches"] >= 250)
                & (phrase_stats["share_highly_riddick_speeches"] >= 0.0175)
            )
            | (
                (phrase_stats["n_highly_riddick_speeches"] >= 8)
                & (phrase_stats["share_highly_riddick_speeches"] >= 0.07)
            )
            | (
                (phrase_stats["n_highly_riddick_speeches"] >= 4)
                & (phrase_stats["share_highly_riddick_speeches"] >= 0.20)
            )
        )
    )

    phrase_stats["procedural_by_cooccurrence_rule_1"] = (
        first_rule_robert_from_procedural
        | first_rule_robert
        | first_rule_riddick
    )

    # ------------------------------------------------------------------
    # Appendix B, second co-occurrence rule
    # Applied only to phrases not identified by first rule.
    # ------------------------------------------------------------------

    second_rule_robert = (
        (phrase_stats["total_count"] > 42)
        & (phrase_stats["n_congresses"] >= 2)
        & (phrase_stats["avg_robert_share_in_speeches"] > 0.05)
    )

    second_rule_riddick_large = (
        (phrase_stats["total_count"] > 1667)
        & (phrase_stats["n_congresses"] >= 4)
        & (phrase_stats["avg_riddick_share_in_speeches"] > 0.075)
    )

    second_rule_riddick_small = (
        (phrase_stats["total_count"] > 42)
        & (phrase_stats["n_congresses"] >= 4)
        & (phrase_stats["avg_riddick_share_in_speeches"] > 0.096)
    )

    phrase_stats["procedural_by_cooccurrence_rule_2"] = (
        (~phrase_stats["procedural_by_cooccurrence_rule_1"])
        & (
            second_rule_robert
            | second_rule_riddick_large
            | second_rule_riddick_small
        )
    )

    phrase_stats["is_cooccurrence_procedural_phrase"] = (
        phrase_stats["procedural_by_cooccurrence_rule_1"]
        | phrase_stats["procedural_by_cooccurrence_rule_2"]
    )

    return phrase_stats


def build_vocabulary_stats(phrase_counts: pd.DataFrame) -> pd.DataFrame:
    """
    Build vocabulary-level statistics from speaker-session phrase counts.
    """
    vocab = (
        phrase_counts
        .groupby("phrase", as_index=False)
        .agg(
            total_count=("count", "sum"),
            n_speaker_sessions=("speaker_session_id", "nunique"),
            n_congresses=("congress", "nunique"),
        )
    )

    phrase_by_congress = (
        phrase_counts
        .groupby(["phrase", "congress"], as_index=False)
        .agg(count_in_congress=("count", "sum"))
    )

    max_count_in_congress = (
        phrase_by_congress
        .groupby("phrase", as_index=False)
        .agg(max_count_in_one_congress=("count_in_congress", "max"))
    )

    vocab = vocab.merge(
        max_count_in_congress,
        on="phrase",
        how="left",
    )

    return vocab


def get_appendix_table4_words() -> list[str]:
    """
    Words from Online Appendix Table 4.

    Gentzkow et al. remove any phrase containing the stem of at least
    one of these words.
    """
    return [
        "absent", "adjourn", "ask", "can", "chairman",
        "committee", "con", "democrat", "etc", "gentleladies",
        "gentlelady", "gentleman", "gentlemen", "gentlewoman", "gentlewomen",
        "hereabout", "hereafter", "hereat", "hereby", "herein",
        "hereinafter", "hereinbefore", "hereinto", "hereof", "hereon",
        "hereto", "heretofore", "hereunder", "hereunto", "hereupon",
        "herewith", "month", "mr", "mrs", "nai",
        "nay", "none", "now", "part", "per",
        "pro", "republican", "say", "senator", "shall",
        "sir", "speak", "speaker", "tell", "thank",
        "thereabout", "thereafter", "thereagainst", "thereat", "therebefore",
        "therebeforn", "thereby", "therefor", "therefore", "therefrom",
        "therein", "thereinafter", "thereof", "thereon", "thereto",
        "theretofore", "thereunder", "thereunto", "thereupon", "therewith",
        "therewithal", "today", "whereabouts", "whereafter", "whereas",
        "whereat", "whereby", "wherefore", "wherefrom", "wherein",
        "whereinto", "whereof", "whereon", "whereto", "whereunder",
        "whereupon", "wherever", "wherewith", "wherewithal", "will",
        "yea", "yes", "yield",
    ]


def get_us_state_and_month_names() -> list[str]:
    """
    US states, District of Columbia, and month names.
    """
    states = [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California",
        "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
        "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
        "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
        "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
        "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey",
        "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio",
        "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",
        "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
        "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming",
        "District of Columbia",
    ]

    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]

    return states + months


def _phrase_contains_pattern(phrase: str, patterns: set[str]) -> bool:
    """
    Check if a stemmed bigram phrase contains a full single- or multi-token pattern.
    """
    phrase = str(phrase)
    phrase_padded = f" {phrase} "

    for pattern in patterns:
        if f" {pattern} " in phrase_padded:
            return True

    return False


def build_appendix_b_exclusion_resources(
    df: pd.DataFrame,
    stemmer,
    stopwords: set[str],
    politicians_dir: Path | None = None,
    ) -> dict:
    """
    Build resources used by Appendix-B phrase filters:
    - member surnames
    - executive surnames from executive.json
    - state/month names
    - Table 4 procedural words
    """
    surname_patterns = set()

    # ------------------------------------------------------------
    # 1. Member surnames from congressional member data
    # ------------------------------------------------------------
    if "last_name" in df.columns:
        last_names = (
            df["last_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        for name in last_names:
            pattern = " ".join(clean_text_to_tokens(
                    text=name,
                    stemmer=stemmer,
                    stopwords=stopwords,
                ))
            if pattern:
                surname_patterns.add(pattern)

    else:
        print(f"Warning: No last names in Dataframe")

    # ------------------------------------------------------------
    # 2. Executive surnames from executive.json
    # ------------------------------------------------------------
    if politicians_dir is not None:
        executive_file = politicians_dir / "executive.json"

        if executive_file.exists():
            executives = read_json(executive_file)

            for person in executives:
                last_name = (
                    person
                    .get("name", {})
                    .get("last")
                )

                if last_name is None:
                    continue

                pattern = " ".join(
                    clean_text_to_tokens(
                        text=last_name,
                        stemmer=stemmer,
                        stopwords=stopwords,
                    )
                )

                if pattern:
                    surname_patterns.add(pattern)

        else:
            print(f"Warning: executive.json not found: {executive_file}")

    # ------------------------------------------------------------
    # Table 4 procedural words
    # ------------------------------------------------------------
    table4_stems = set()

    for word in get_appendix_table4_words():
        cleaned_word = " ".join(
            clean_text_to_tokens(
                text=word,
                stemmer=stemmer,
                stopwords=stopwords,
            )
        )

        if cleaned_word:
            table4_stems.add(cleaned_word)

    # ------------------------------------------------------------
    # State and month names
    # ------------------------------------------------------------

    state_month_patterns = set()

    for name in get_us_state_and_month_names():
        pattern = " ".join(clean_text_to_tokens(
                text=name,
                stemmer=stemmer,
                stopwords=stopwords,
            ))
        if pattern:
            state_month_patterns.add(pattern)

    return {
        "surname_patterns": surname_patterns,
        "table4_stems": table4_stems,
        "state_month_patterns": state_month_patterns,
    }


def add_appendix_b_filter_flags_to_vocabulary(
    vocabulary: pd.DataFrame,
    roberts_phrases: set[str],
    riddick_phrases: set[str],
    cooccurrence_phrase_stats: pd.DataFrame,
    exclusion_resources: dict,
) -> pd.DataFrame:
    """
    Add Appendix-B filter flags to the vocabulary.
    """
    vocab = vocabulary.copy()

    vocab["is_robert_phrase"] = vocab["phrase"].isin(roberts_phrases)
    vocab["is_riddick_phrase"] = vocab["phrase"].isin(riddick_phrases)

    vocab["is_manual_procedural_phrase"] = (
        vocab["is_robert_phrase"] | vocab["is_riddick_phrase"]
    )

    coocc_cols = [
        "phrase",
        "procedural_by_cooccurrence_rule_1",
        "procedural_by_cooccurrence_rule_2",
        "is_cooccurrence_procedural_phrase",
        "n_speeches_containing",
        "n_procedural_speeches",
        "n_highly_robert_speeches",
        "n_highly_riddick_speeches",
        "avg_robert_share_in_speeches",
        "avg_riddick_share_in_speeches",
    ]

    vocab = vocab.merge(
        cooccurrence_phrase_stats[coocc_cols],
        on="phrase",
        how="left",
    )

    bool_cols = [
        "procedural_by_cooccurrence_rule_1",
        "procedural_by_cooccurrence_rule_2",
        "is_cooccurrence_procedural_phrase",
    ]

    for col in bool_cols:
        vocab[col] = vocab[col].fillna(False).astype(bool)

    numeric_cols = [
        "n_speeches_containing",
        "n_procedural_speeches",
        "n_highly_robert_speeches",
        "n_highly_riddick_speeches",
        "avg_robert_share_in_speeches",
        "avg_riddick_share_in_speeches",
    ]

    for col in numeric_cols:
        vocab[col] = vocab[col].fillna(0)

    surname_patterns = exclusion_resources["surname_patterns"]
    table4_stems = exclusion_resources["table4_stems"]
    state_month_patterns = exclusion_resources["state_month_patterns"]

    def has_number_or_symbol(phrase: str) -> bool:
        phrase = str(phrase)
        return bool(re.search(r"\d", phrase)) or bool(
            re.search(r"[^a-z\s]", phrase)
        )

    def has_one_letter_word(phrase: str) -> bool:
        return any(len(token) == 1 for token in str(phrase).split())

    def has_table4_word(phrase: str) -> bool:
        tokens = str(phrase).split()
        return any(token in table4_stems for token in tokens)

    def has_member_surname(phrase: str) -> bool:
        return _phrase_contains_pattern(phrase, surname_patterns)

    def has_state_or_month(phrase: str) -> bool:
        return _phrase_contains_pattern(phrase, state_month_patterns)

    vocab["excluded_number_or_symbol"] = vocab["phrase"].apply(
        has_number_or_symbol
    )

    vocab["excluded_short_phrase"] = (
        vocab["phrase"].astype(str).str.len() < 5
    )

    vocab["excluded_one_letter_word"] = vocab["phrase"].apply(
        has_one_letter_word
    )

    vocab["excluded_table4_word"] = vocab["phrase"].apply(
        has_table4_word
    )

    vocab["excluded_member_surname"] = vocab["phrase"].apply(
        has_member_surname
    )

    vocab["excluded_state_or_month"] = vocab["phrase"].apply(
        has_state_or_month
    )

    vocab["excluded_procedural_manual_or_cooccurrence"] = (
        vocab["is_manual_procedural_phrase"]
        | vocab["is_cooccurrence_procedural_phrase"]
    )

    vocab["excluded_by_appendix_b"] = (
        vocab["excluded_procedural_manual_or_cooccurrence"]
        | vocab["excluded_number_or_symbol"]
        | vocab["excluded_short_phrase"]
        | vocab["excluded_one_letter_word"]
        | vocab["excluded_table4_word"]
        | vocab["excluded_member_surname"]
        | vocab["excluded_state_or_month"]
    )

    return vocab


def apply_phrase_filters(
    vocab: pd.DataFrame,
    min_total_count: int = 1,
    min_speaker_sessions: int = 1,
    min_count_in_one_congress: int = 1,
) -> pd.DataFrame:
    """
    Apply frequency filters and combine them with Appendix-B exclusions.

    Gentzkow-style baseline frequency filters:
    - total_count >= 100
    - n_speaker_sessions >= 10
    - max_count_in_one_congress >= 10
    """
    vocab = vocab.copy()

    vocab["passes_frequency_filters"] = (
        (vocab["total_count"] >= min_total_count)
        & (vocab["n_speaker_sessions"] >= min_speaker_sessions)
        & (vocab["max_count_in_one_congress"] >= min_count_in_one_congress)
    )

    if "excluded_by_appendix_b" not in vocab.columns:
        raise KeyError("Column 'excluded_by_appendix_b' is missing. Run Appendix-B filtering first.")

    vocab["keep_phrase"] = (
        vocab["passes_frequency_filters"]
        & (~vocab["excluded_by_appendix_b"])
    )

    return vocab