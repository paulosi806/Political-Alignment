"""
Streamlit app — congressional partisan-language model.

Memory budget (Streamlit Cloud free tier: 1 GB):
  ~150 MB  Python + Streamlit overhead
  ~600 MB  peak PyArrow decode for 1 congress (7× the ~70 MB DataFrame)
  ~100 MB  model fit headroom (design matrices are dense but small)
  ──────────────────────────────────────────────────────
  Max safe: 2 congresses if reads are serialised; 1 is comfortable.

Design decisions that keep us inside the budget:
- phrase_counts is NEVER cached — read once per run, deleted immediately
  after filtering.  st.cache_data on a 70 MB frame plus a .copy() filter
  would keep ≥2 live copies simultaneously.
- st.form gates the run so widget interactions (sliders, selectbox) do
  not trigger reruns while a fit is in flight or between runs.
- Preflight refuses >2 congresses before any parquet I/O.
- Column projection: read only the 4 columns the model needs.
- gc.collect() + del at each major boundary.
"""

import gc
import sys
import time
import traceback
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
    114: "2015–16", 115: "2017–18", 116: "2019–20",
    117: "2021–22", 118: "2023–24", 119: "2025–26",
}

MAX_CONGRESSES = 2   # hard cap: 3+ congresses reliably OOM on 1 GB
MAX_PHRASES    = 500 # phrases cap; at 500 x 10 lambda steps model fits in ~3 min


# ── Static/small data — cached for the process lifetime ──────────────────────

@st.cache_resource
def _load_config() -> dict:
    with open(HERE / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner="Loading speaker sessions…")
def _load_speaker_sessions() -> pd.DataFrame:
    return pd.read_parquet(HERE / "speaker_sessions.parquet")


# ── Preflight helpers ─────────────────────────────────────────────────────────

def _missing_congress_chamber_combos(
    congresses: list[int],
    speaker_sessions: pd.DataFrame,
    majority_cfg: dict,
) -> list[tuple[int, str]]:
    """Return (congress, chamber) pairs that are absent from majority_cfg."""
    ss = speaker_sessions[speaker_sessions["congress"].isin(congresses)]
    chambers = ss["chamber_member"].dropna().unique().tolist()
    missing = []
    for c in congresses:
        ck = str(c)
        for ch in chambers:
            if ck not in majority_cfg or ch not in majority_cfg[ck]:
                missing.append((c, ch))
    return missing


def _estimate_mb(congresses: list[int], max_phrases: int) -> float:
    """Very rough RSS estimate for a run: 600 MB per congress + 100 MB base."""
    return 150 + len(congresses) * 620


# ── Phrase count loader — NOT cached, read fresh per run ─────────────────────

def _read_phrase_counts(congresses: tuple[int, ...]) -> pd.DataFrame:
    """
    Read only the 4 columns the model needs for the selected congresses.
    Never cached — caller is responsible for deleting after use.
    """
    import pyarrow.parquet as pq
    table = pq.read_table(
        HERE / "phrase_counts_long.parquet",
        filters=[("congress", "in", list(congresses))],
        columns=["speaker_session_id", "congress", "phrase", "count"],
    )
    return table.to_pandas()


# ── Speaker data preparation ──────────────────────────────────────────────────

def _prepare_speaker_data(
    speaker_sessions: pd.DataFrame,
    exposure: pd.DataFrame,
    majority_cfg: dict,
    congresses: tuple[int, ...],
) -> pd.DataFrame:
    """
    Build speaker-session rows with covariates and exposure.
    Takes a pre-computed exposure frame so phrase_counts never enters caching.
    """
    meta_cols = [
        "speaker_session_id", "speaker_bioguide", "congress",
        "party", "state", "chamber_member", "gender",
    ]
    ss = (
        speaker_sessions[speaker_sessions["congress"].isin(congresses)]
        [meta_cols]
        .drop_duplicates()
        .copy()
    )
    ss = add_census_region(ss, state_col="state")
    ss = add_majority_party_indicator(
        ss,
        majority_party_by_congress_chamber=majority_cfg,
        congress_col="congress",
        chamber_col="chamber_member",
        party_col="party",
    )
    ss = ss.merge(exposure, on="speaker_session_id", how="inner")
    ss["party"] = ss["party"].astype("string").str.strip()
    ss = ss[
        ss["party"].isin(["Democrat", "Republican"]) & (ss["exposure"] > 0)
    ].copy()
    ss["republican"] = (ss["party"] == "Republican").astype(int)
    return ss


# ── Load shared resources ─────────────────────────────────────────────────────

config         = _load_config()
poisson_cfg    = config.get("penalized_poisson", {})
majority_cfg   = poisson_cfg.get("majority_party_by_congress_chamber", {})
speaker_sessions = _load_speaker_sessions()
all_congresses = sorted(speaker_sessions["congress"].unique().tolist())

# ── Page header ───────────────────────────────────────────────────────────────

st.title("🏛️ Congressional Partisan Language")
st.caption(
    "Penalized Poisson model following Gentzkow et al. (2019) · "
    "114th–119th Congress (2015–2026)"
)

# ── Data overview (no phrase_counts needed) ───────────────────────────────────

with st.expander("📊 Data overview", expanded=True):
    c1, c2 = st.columns(2)
    c1.metric("Unique legislators",
              f"{speaker_sessions['speaker_bioguide'].nunique():,}")
    c2.metric("Speaker-sessions", f"{len(speaker_sessions):,}")

    party_congress = (
        speaker_sessions
        .groupby(["congress", "party"])
        .size()
        .unstack(fill_value=0)
        .rename(index=lambda c: f"{c} ({CONGRESS_YEARS.get(c, '')})")
    )
    st.subheader("Speaker-sessions by Congress and party")
    st.bar_chart(party_congress, color=["#3b82f6", "#ef4444"])

st.divider()

# ── Run form — all parameters + button in one form so no accidental reruns ───

st.subheader("Run the model")

with st.form("run_form"):
    col_left, col_right = st.columns(2)

    with col_left:
        selected_congresses = st.multiselect(
            f"Congress(es) — max {MAX_CONGRESSES}",
            options=all_congresses,
            default=[119],
            format_func=lambda c: f"{c} ({CONGRESS_YEARS.get(c, '')})",
            help=f"Max {MAX_CONGRESSES} at once due to memory limits on Streamlit Cloud.",
        )
        max_phrases = st.number_input(
            "Max phrases per Congress",
            min_value=25, max_value=MAX_PHRASES, value=100, step=25,
            help=f"Top N most-frequent phrases per Congress. Hard cap: {MAX_PHRASES}.",
        )
        lambda_steps = st.slider(
            "Lambda path steps", min_value=5, max_value=50, value=10,
            help="Regularisation path length. Lower = faster.",
        )

    with col_right:
        top_n = st.slider("Top phrases to show", min_value=5, max_value=50, value=15)
        top_n_dir = st.slider(
            "Top phrases per direction", min_value=5, max_value=25, value=10
        )
        st.caption(
            f"**Memory note:** each Congress loads ~600 MB peak during parquet decode.\n"
            f"Max {MAX_CONGRESSES} congress(es) enforced. Single-congress runs are most reliable."
        )

    submitted = st.form_submit_button(
        "▶ Estimate partisanship", type="primary", use_container_width=True
    )

# ── Preflight + run ───────────────────────────────────────────────────────────

if submitted:
    # ── Preflight ─────────────────────────────────────────────────────────────
    errors = []

    if not selected_congresses:
        errors.append("Select at least one Congress.")

    if len(selected_congresses) > MAX_CONGRESSES:
        errors.append(
            f"Selected {len(selected_congresses)} congresses but max is "
            f"{MAX_CONGRESSES} (memory limit). Deselect some and try again."
        )

    max_phrases = min(int(max_phrases), MAX_PHRASES)

    if selected_congresses and len(selected_congresses) <= MAX_CONGRESSES:
        missing_combos = _missing_congress_chamber_combos(
            selected_congresses, speaker_sessions, majority_cfg
        )
        if missing_combos:
            for c, ch in missing_combos:
                errors.append(
                    f"Congress {c} × chamber '{ch}' has no majority-party entry "
                    f"in config.yaml. Add it under "
                    f"penalized_poisson.majority_party_by_congress_chamber."
                )

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    congresses_tuple = tuple(sorted(selected_congresses))
    congress_labels  = ", ".join(
        f"{c} ({CONGRESS_YEARS.get(c, '')})" for c in congresses_tuple
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    with st.status(
        f"Running — Congress {congress_labels} · {max_phrases} phrases · "
        f"{lambda_steps} lambda steps",
        expanded=True,
    ) as status:
        try:
            t0 = time.time()

            # 1. Read phrase counts — column projection, NOT cached
            st.write("📂 Reading phrase counts…")
            phrase_counts = _read_phrase_counts(congresses_tuple)
            st.write(
                f"   {len(phrase_counts):,} rows loaded "
                f"({phrase_counts.memory_usage(deep=True).sum()/1024**2:.0f} MB)"
            )

            # 2. Compute exposure before filtering (uses full congress slice)
            exposure = (
                phrase_counts
                .groupby("speaker_session_id", as_index=False)
                .agg(exposure=("count", "sum"))
            )

            # 3. Prepare speaker metadata + merge exposure
            st.write("🔧 Preparing speaker data…")
            speaker_data = _prepare_speaker_data(
                speaker_sessions, exposure, majority_cfg, congresses_tuple
            )
            del exposure; gc.collect()

            # 4. Filter phrase counts to usable speakers, then free the full slice
            valid_ids = set(speaker_data["speaker_session_id"])
            phrase_counts_filtered = phrase_counts[
                phrase_counts["speaker_session_id"].isin(valid_ids)
            ]
            # drop the original immediately — one copy alive from here on
            del phrase_counts, valid_ids; gc.collect()

            st.write(
                f"   Speaker-sessions: {len(speaker_data):,} · "
                f"phrase rows: {len(phrase_counts_filtered):,}"
            )

            # 5. Fit
            st.write("⚙️ Fitting phrase-specific Poisson models (this takes a while)…")
            result = estimate_static_covariate_partisanship_model(
                speaker_data=speaker_data,
                phrase_counts=phrase_counts_filtered,
                lambda_path_steps=int(lambda_steps),
                lambda_path_min_ratio=float(
                    poisson_cfg.get("lambda_path_min_ratio", 1e-5)
                ),
                min_penalty_alpha=float(
                    poisson_cfg.get("min_penalty_alpha", 1e-5)
                ),
                maxiter=int(poisson_cfg.get("maxiter", 1000)),
                max_phrases=max_phrases,
                return_phrase_parameters=False,
                progress_label="Main model",
                n_jobs=1,
                parallel_backend="threading",
            )

            # 6. Free model inputs immediately
            del phrase_counts_filtered, speaker_data; gc.collect()

            elapsed = time.time() - t0

            if result["success"]:
                st.session_state["model_result"] = result
                st.session_state["top_n"]        = top_n
                st.session_state["top_n_dir"]    = top_n_dir
                status.update(
                    label=(
                        f"✅ Done in {elapsed:.0f}s · "
                        f"{result['n_phrases']:,} phrases · "
                        f"convergence {result['best_model_convergence_percent']:.1f}%"
                    ),
                    state="complete",
                )
            else:
                status.update(
                    label=f"Model returned failure: {result['reason']}", state="error"
                )
                st.error(f"Estimation failed: {result['reason']}")

        except MemoryError:
            gc.collect()
            status.update(label="Out of memory", state="error")
            st.error(
                "The server ran out of memory. Try fewer congresses or "
                "reduce max phrases and lambda steps, then run again."
            )
        except Exception:
            tb = traceback.format_exc()
            status.update(label="Crashed — see traceback below", state="error")
            st.error("The model raised an exception:")
            st.code(tb, language="python")

# ── Results (persisted in session_state across reruns) ───────────────────────

if "model_result" in st.session_state:
    res         = st.session_state["model_result"]
    top_n_show  = st.session_state.get("top_n", 15)
    top_n_dir_show = st.session_state.get("top_n_dir", 10)

    partisanship_df = res["partisanship"]
    phrase_probs    = res["phrase_probabilities"]

    st.divider()
    st.subheader("Results")

    # Average partisanship chart
    st.markdown("#### Average partisanship by Congress")
    st.caption(
        "Values above 0.5 = stronger partisan distinguishability. "
        "Higher → a classifier can more reliably identify party from speech."
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
            use_container_width=True, hide_index=True,
        )

    # Top phrases
    st.markdown("#### Top partisan phrases")
    result_congresses = sorted(phrase_probs["congress"].dropna().unique().astype(int))
    sel_congress = st.selectbox(
        "Congress",
        result_congresses,
        format_func=lambda c: f"{c} ({CONGRESS_YEARS.get(c, '')})",
    )
    phrase_probs_t = phrase_probs[phrase_probs["congress"] == sel_congress].copy()

    tab_overall, tab_dir = st.tabs(["Top overall", "By direction"])

    with tab_overall:
        top_overall = get_top_partisan_phrases(phrase_probs_t, n=top_n_show)

        def _colour(val):
            c = "#ef4444" if val > 0 else "#3b82f6"
            return f"color: {c}; font-weight: bold"

        show_cols = [
            c for c in [
                "phrase", "phrase_partisanship", "abs_phrase_partisanship",
                "predicted_per_100k_republican", "predicted_per_100k_democrat",
                "posterior_republican_mean",
            ] if c in top_overall.columns
        ]
        st.dataframe(
            top_overall[show_cols].style.applymap(
                _colour, subset=["phrase_partisanship"]
            ),
            use_container_width=True, hide_index=True,
        )

    with tab_dir:
        top_d = get_top_partisan_phrases_by_direction(phrase_probs_t, n=top_n_dir_show)
        rep_ph = top_d[top_d["direction"] == "Republican"]
        dem_ph = top_d[top_d["direction"] == "Democratic"]

        dir_cols = [
            c for c in [
                "phrase", "phrase_partisanship",
                "predicted_per_100k_republican", "predicted_per_100k_democrat",
            ] if c in top_d.columns
        ]
        dc1, dc2 = st.columns(2)
        with dc1:
            st.markdown("**🔴 Republican phrases**")
            st.dataframe(rep_ph[dir_cols], use_container_width=True, hide_index=True)
        with dc2:
            st.markdown("**🔵 Democratic phrases**")
            st.dataframe(dem_ph[dir_cols], use_container_width=True, hide_index=True)

    st.caption(
        f"Estimated {res['n_phrases']:,} phrases · "
        f"Convergence {res['best_model_convergence_percent']:.1f}%"
    )
