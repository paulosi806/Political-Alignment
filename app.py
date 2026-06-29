import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from utils_analysis import (
    add_census_region,
    add_majority_party_indicator,
    estimate_static_covariate_partisanship_model,
    get_top_partisan_phrases,
    get_top_partisan_phrases_by_direction,
)

st.set_page_config(
    page_title="Congressional Partisan Language",
    page_icon="🏛️",
    layout="wide",
)

CONGRESS_YEARS = {
    114: "2015–16",
    115: "2017–18",
    116: "2019–20",
    117: "2021–22",
    118: "2023–24",
    119: "2025–26",
}


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading speaker sessions…")
def load_speaker_sessions():
    return pd.read_parquet(HERE / "speaker_sessions.parquet")


@st.cache_data(show_spinner="Loading phrase counts…")
def load_phrase_counts():
    return pd.read_parquet(HERE / "phrase_counts_long.parquet")


@st.cache_data(show_spinner=False)
def load_config():
    with open(HERE / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Speaker data preparation ───────────────────────────────────────────────────

def prepare_speaker_data(
    speaker_sessions: pd.DataFrame,
    phrase_counts: pd.DataFrame,
    majority_party_by_congress_chamber: dict,
) -> pd.DataFrame:
    speaker_exposure = (
        phrase_counts
        .groupby("speaker_session_id", as_index=False)
        .agg(exposure=("count", "sum"))
    )

    meta_cols = [
        "speaker_session_id", "speaker_bioguide", "congress",
        "party", "state", "chamber_member", "gender",
    ]

    speaker_data = (
        speaker_sessions[meta_cols]
        .drop_duplicates()
        .copy()
    )

    speaker_data = add_census_region(speaker_data, state_col="state")
    speaker_data = add_majority_party_indicator(
        speaker_data,
        majority_party_by_congress_chamber=majority_party_by_congress_chamber,
        congress_col="congress",
        chamber_col="chamber_member",
        party_col="party",
    )
    speaker_data = speaker_data.merge(speaker_exposure, on="speaker_session_id", how="inner")
    speaker_data["party"] = speaker_data["party"].astype("string").str.strip()
    speaker_data = speaker_data[
        speaker_data["party"].isin(["Democrat", "Republican"])
        & (speaker_data["exposure"] > 0)
    ].copy()
    speaker_data["republican"] = (speaker_data["party"] == "Republican").astype(int)

    return speaker_data


# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("🏛️ Congressional Partisan Language")
st.caption(
    "Penalized Poisson model following Gentzkow et al. (2019) · "
    "114th–119th Congress (2015–2026)"
)

speaker_sessions = load_speaker_sessions()
phrase_counts_raw = load_phrase_counts()
config = load_config()
poisson_cfg = config.get("penalized_poisson", {})
majority_cfg = poisson_cfg.get("majority_party_by_congress_chamber", {})

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Model parameters")

    max_phrases = st.number_input(
        "Max phrases per Congress",
        min_value=50,
        max_value=5000,
        value=200,
        step=50,
        help="Limit phrases to the N most frequent per Congress. Lower = faster run.",
    )

    lambda_steps = st.slider(
        "Lambda path steps",
        min_value=5,
        max_value=100,
        value=20,
        help="Number of regularization steps. Lower = faster, less precise.",
    )

    n_jobs = st.slider(
        "Parallel jobs",
        min_value=1,
        max_value=8,
        value=2,
        help="Number of parallel workers for phrase estimation.",
    )

    top_n = st.slider(
        "Top phrases to show",
        min_value=5,
        max_value=50,
        value=15,
    )

    top_n_dir = st.slider(
        "Top phrases per direction",
        min_value=5,
        max_value=25,
        value=10,
    )

    st.divider()
    st.caption(
        "💡 Start with default settings for a quick test run (~1–5 min). "
        "Full estimation (max_phrases=null) can take much longer."
    )

# ── Overview ───────────────────────────────────────────────────────────────────

with st.expander("📊 Data overview", expanded=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("Unique legislators", f"{speaker_sessions['speaker_bioguide'].nunique():,}")
    c2.metric("Speaker-sessions", f"{len(speaker_sessions):,}")
    c3.metric("Unique bigrams", f"{phrase_counts_raw['phrase'].nunique():,}")

    party_congress = (
        speaker_sessions
        .groupby(["congress", "party"])
        .size()
        .unstack(fill_value=0)
        .rename(index=lambda c: f"{c} ({CONGRESS_YEARS.get(c, '')})")
    )

    st.subheader("Speaker-sessions by Congress and party")
    st.bar_chart(party_congress, color=["#3b82f6", "#ef4444"])

# ── Run model ──────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Run the model")

run_col, info_col = st.columns([1, 3])
with run_col:
    run_clicked = st.button("▶ Estimate partisanship", type="primary", use_container_width=True)
with info_col:
    st.caption(
        f"Will estimate the penalized Poisson model on up to {max_phrases:,} phrases "
        f"per Congress with {lambda_steps} lambda steps across {n_jobs} worker(s)."
    )

if run_clicked:
    with st.status("Running estimation…", expanded=True) as status:
        try:
            import traceback
            t0 = time.time()

            st.write("Preparing speaker data…")
            speaker_data = prepare_speaker_data(
                speaker_sessions, phrase_counts_raw, majority_cfg
            )

            phrase_counts_filtered = phrase_counts_raw.merge(
                speaker_data[["speaker_session_id"]],
                on="speaker_session_id",
                how="inner",
            )

            st.write(f"Speaker-sessions: {len(speaker_data):,}")
            st.write(f"Phrase rows: {len(phrase_counts_filtered):,}")
            st.write("Fitting phrase-specific Poisson models…")

            result = estimate_static_covariate_partisanship_model(
                speaker_data=speaker_data,
                phrase_counts=phrase_counts_filtered,
                lambda_path_steps=lambda_steps,
                lambda_path_min_ratio=float(poisson_cfg.get("lambda_path_min_ratio", 1e-5)),
                min_penalty_alpha=float(poisson_cfg.get("min_penalty_alpha", 1e-5)),
                maxiter=int(poisson_cfg.get("maxiter", 1000)),
                max_phrases=int(max_phrases),
                return_phrase_parameters=False,
                progress_label="Main model",
                n_jobs=n_jobs,
                parallel_backend="threading",
            )

            elapsed = time.time() - t0

            if result["success"]:
                st.session_state["model_result"] = result
                st.session_state["top_n"] = top_n
                st.session_state["top_n_dir"] = top_n_dir
                status.update(
                    label=f"Done in {elapsed:.0f}s · {result['n_phrases']:,} phrases estimated",
                    state="complete",
                )
            else:
                status.update(label=f"Failed: {result['reason']}", state="error")
                st.error(f"Estimation failed: {result['reason']}")

        except Exception:
            tb = traceback.format_exc()
            status.update(label="Crashed — see error below", state="error")
            st.error("The model raised an exception:")
            st.code(tb, language="python")

# ── Results ────────────────────────────────────────────────────────────────────

if "model_result" in st.session_state:
    res = st.session_state["model_result"]
    top_n_show = st.session_state.get("top_n", top_n)
    top_n_dir_show = st.session_state.get("top_n_dir", top_n_dir)

    partisanship_df = res["partisanship"]
    phrase_probs = res["phrase_probabilities"]

    st.divider()
    st.subheader("Results")

    # ── Average partisanship chart ─────────────────────────────────────────────
    st.markdown("#### Average partisanship by Congress")
    st.caption(
        "Values above 0.5 indicate stronger partisan distinguishability. "
        "The higher the value, the more a classifier can correctly identify "
        "party from speech patterns."
    )

    chart_df = (
        partisanship_df
        .sort_values("congress")
        .set_index("congress")[["average_partisanship"]]
        .rename(index=lambda c: f"{c} ({CONGRESS_YEARS.get(int(c), '')})")
    )
    st.line_chart(chart_df, color=["#6366f1"])

    with st.expander("Partisanship table"):
        display_cols = [
            c for c in [
                "congress", "average_partisanship",
                "n_speaker_sessions", "n_democrat_speaker_sessions",
                "n_republican_speaker_sessions", "n_phrases",
                "best_model_convergence_percent",
            ]
            if c in partisanship_df.columns
        ]
        st.dataframe(
            partisanship_df[display_cols].sort_values("congress"),
            use_container_width=True,
            hide_index=True,
        )

    # ── Top phrases ────────────────────────────────────────────────────────────
    st.markdown("#### Top partisan phrases by Congress")

    congresses = sorted(phrase_probs["congress"].dropna().unique().astype(int))
    selected_congress = st.selectbox(
        "Congress",
        congresses,
        format_func=lambda c: f"{c}th ({CONGRESS_YEARS.get(c, '')})",
    )

    phrase_probs_t = phrase_probs[phrase_probs["congress"] == selected_congress].copy()

    tab_overall, tab_direction = st.tabs(["Top overall", "By direction"])

    with tab_overall:
        top_overall = get_top_partisan_phrases(phrase_probs_t, n=top_n_show)

        def style_partisanship(val):
            color = "#ef4444" if val > 0 else "#3b82f6"
            return f"color: {color}; font-weight: bold"

        show_cols = [
            c for c in [
                "phrase", "phrase_partisanship", "abs_phrase_partisanship",
                "predicted_per_100k_republican", "predicted_per_100k_democrat",
                "posterior_republican_mean",
            ]
            if c in top_overall.columns
        ]

        st.dataframe(
            top_overall[show_cols].style.applymap(
                style_partisanship, subset=["phrase_partisanship"]
            ),
            use_container_width=True,
            hide_index=True,
        )

    with tab_direction:
        top_dir = get_top_partisan_phrases_by_direction(phrase_probs_t, n=top_n_dir_show)

        rep_phrases = top_dir[top_dir["direction"] == "Republican"]
        dem_phrases = top_dir[top_dir["direction"] == "Democratic"]

        dir_col1, dir_col2 = st.columns(2)

        dir_show_cols = [
            c for c in [
                "phrase", "phrase_partisanship",
                "predicted_per_100k_republican", "predicted_per_100k_democrat",
            ]
            if c in top_dir.columns
        ]

        with dir_col1:
            st.markdown("**🔴 Republican phrases**")
            st.dataframe(
                rep_phrases[dir_show_cols],
                use_container_width=True,
                hide_index=True,
            )

        with dir_col2:
            st.markdown("**🔵 Democratic phrases**")
            st.dataframe(
                dem_phrases[dir_show_cols],
                use_container_width=True,
                hide_index=True,
            )

    st.caption(
        f"Estimated {res['n_phrases']:,} phrases · "
        f"Convergence: {res['best_model_convergence_percent']:.1f}%"
    )
