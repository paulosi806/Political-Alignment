#!/usr/bin/env python3
"""Build script: reads parquet data and generates public/index.html."""

import json
import os
import pandas as pd

DATA_DIR = os.path.dirname(__file__)
OUT_DIR = os.path.join(DATA_DIR, "public")
os.makedirs(OUT_DIR, exist_ok=True)

ss = pd.read_parquet(os.path.join(DATA_DIR, "speaker_sessions.parquet"))
pc = pd.read_parquet(os.path.join(DATA_DIR, "phrase_counts_long.parquet"))

# ── Summary stats ──────────────────────────────────────────────────────────────
total_speakers = ss["speaker_bioguide"].nunique()
total_sessions = len(ss)
total_phrases = pc["phrase"].nunique()
congresses = sorted(ss["congress"].unique().tolist())

# ── Party breakdown by Congress ────────────────────────────────────────────────
party_congress = (
    ss.groupby(["congress", "party"]).size().unstack(fill_value=0).reset_index()
)
party_congress.columns.name = None
party_congress["congress"] = party_congress["congress"].astype(int)

# ── Token volume by Congress and party ─────────────────────────────────────────
tokens = (
    ss.groupby(["congress", "party"])["n_tokens_clean"]
    .sum()
    .unstack(fill_value=0)
    .reset_index()
)
tokens.columns.name = None
tokens["congress"] = tokens["congress"].astype(int)

# ── Partisan phrases ───────────────────────────────────────────────────────────
phrase_party = pc.groupby(["phrase", "party"])["count"].sum().unstack(fill_value=0)
phrase_party.columns = [c.lower() for c in phrase_party.columns]
totals = phrase_party.sum(axis=1)
phrase_party = phrase_party[totals >= 500].copy()
phrase_party["total"] = phrase_party.sum(axis=1)
phrase_party["rep_share"] = phrase_party.get("republican", 0) / phrase_party["total"]
phrase_party["dem_share"] = phrase_party.get("democrat", 0) / phrase_party["total"]

top_rep = (
    phrase_party.nlargest(15, "rep_share")[["rep_share", "total"]]
    .reset_index()
    .rename(columns={"phrase": "phrase", "rep_share": "share", "total": "count"})
)
top_dem = (
    phrase_party.nlargest(15, "dem_share")[["dem_share", "total"]]
    .reset_index()
    .rename(columns={"phrase": "phrase", "dem_share": "share", "total": "count"})
)

top_rep["share"] = top_rep["share"].round(3)
top_dem["share"] = top_dem["share"].round(3)


def congress_label(n):
    return f"{n}th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th") and f"{n}{({1:'st',2:'nd',3:'rd'}.get(n%10,'th'))}"


CONGRESS_YEARS = {114: "2015–16", 115: "2017–18", 116: "2019–20", 117: "2021–22", 118: "2023–24", 119: "2025–26"}

# ── JSON payloads ──────────────────────────────────────────────────────────────
chart_party = {
    "labels": [f"{c} ({CONGRESS_YEARS.get(c, '')})" for c in party_congress["congress"].tolist()],
    "democrat": party_congress.get("Democrat", pd.Series([0] * len(party_congress))).tolist(),
    "republican": party_congress.get("Republican", pd.Series([0] * len(party_congress))).tolist(),
}

chart_tokens = {
    "labels": [f"{c} ({CONGRESS_YEARS.get(c, '')})" for c in tokens["congress"].tolist()],
    "democrat": [round(v / 1_000_000, 2) for v in tokens.get("Democrat", pd.Series([0] * len(tokens))).tolist()],
    "republican": [round(v / 1_000_000, 2) for v in tokens.get("Republican", pd.Series([0] * len(tokens))).tolist()],
}

partisan_rep = top_rep[["phrase", "share", "count"]].to_dict(orient="records")
partisan_dem = top_dem[["phrase", "share", "count"]].to_dict(orient="records")

# ── HTML ───────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Political Alignment — Congressional Speech</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: 'Georgia', serif; }}
  .badge-r {{ background:#ef4444; color:#fff; }}
  .badge-d {{ background:#3b82f6; color:#fff; }}
  canvas {{ max-height: 300px; }}
</style>
</head>
<body class="bg-gray-50 text-gray-800">

<!-- Header -->
<header class="bg-white border-b border-gray-200 py-8 px-6 text-center">
  <h1 class="text-3xl font-bold text-gray-900">Congressional Partisan Language</h1>
  <p class="mt-2 text-gray-500 text-sm max-w-xl mx-auto">
    Phrase-level partisanship in U.S. congressional floor speeches, 114th–119th Congress (2015–2026).
    Based on a penalized Poisson model following Gentzkow et al. (2019).
  </p>
</header>

<!-- Stats cards -->
<section class="max-w-5xl mx-auto mt-8 px-4 grid grid-cols-1 sm:grid-cols-3 gap-4">
  <div class="bg-white rounded-xl shadow-sm p-6 text-center border border-gray-100">
    <p class="text-4xl font-bold text-gray-900">{total_speakers:,}</p>
    <p class="mt-1 text-sm text-gray-500">Unique legislators</p>
  </div>
  <div class="bg-white rounded-xl shadow-sm p-6 text-center border border-gray-100">
    <p class="text-4xl font-bold text-gray-900">{total_sessions:,}</p>
    <p class="mt-1 text-sm text-gray-500">Speaker-sessions</p>
  </div>
  <div class="bg-white rounded-xl shadow-sm p-6 text-center border border-gray-100">
    <p class="text-4xl font-bold text-gray-900">{total_phrases:,}</p>
    <p class="mt-1 text-sm text-gray-500">Unique bigrams</p>
  </div>
</section>

<!-- Charts -->
<section class="max-w-5xl mx-auto mt-8 px-4 grid grid-cols-1 md:grid-cols-2 gap-6">
  <div class="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
    <h2 class="text-base font-semibold mb-4">Speaker-sessions by Congress</h2>
    <canvas id="chartParty"></canvas>
  </div>
  <div class="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
    <h2 class="text-base font-semibold mb-4">Speech volume by Congress (M tokens)</h2>
    <canvas id="chartTokens"></canvas>
  </div>
</section>

<!-- Partisan phrases -->
<section class="max-w-5xl mx-auto mt-8 px-4 grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
  <div class="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
    <h2 class="text-base font-semibold mb-1">Most Republican phrases</h2>
    <p class="text-xs text-gray-400 mb-4">Bigrams used ≥500 times, ranked by Republican usage share</p>
    <table class="w-full text-sm">
      <thead><tr class="text-left text-gray-400 text-xs uppercase border-b">
        <th class="pb-2">Phrase</th><th class="pb-2 text-right">R-share</th><th class="pb-2 text-right">Uses</th>
      </tr></thead>
      <tbody id="repTable"></tbody>
    </table>
  </div>
  <div class="bg-white rounded-xl shadow-sm p-6 border border-gray-100">
    <h2 class="text-base font-semibold mb-1">Most Democratic phrases</h2>
    <p class="text-xs text-gray-400 mb-4">Bigrams used ≥500 times, ranked by Democratic usage share</p>
    <table class="w-full text-sm">
      <thead><tr class="text-left text-gray-400 text-xs uppercase border-b">
        <th class="pb-2">Phrase</th><th class="pb-2 text-right">D-share</th><th class="pb-2 text-right">Uses</th>
      </tr></thead>
      <tbody id="demTable"></tbody>
    </table>
  </div>
</section>

<footer class="text-center text-xs text-gray-400 pb-8">
  Data: Congressional Record (GovInfo) · Legislators: unitedstates/congress-legislators ·
  Methodology follows Gentzkow, Shapiro &amp; Taddy (2019)
</footer>

<script>
const PARTY_DATA = {json.dumps(chart_party)};
const TOKEN_DATA = {json.dumps(chart_tokens)};
const REP_PHRASES = {json.dumps(partisan_rep)};
const DEM_PHRASES = {json.dumps(partisan_dem)};

const RED = "rgba(239,68,68,0.85)";
const BLUE = "rgba(59,130,246,0.85)";
const RED_BORDER = "rgba(239,68,68,1)";
const BLUE_BORDER = "rgba(59,130,246,1)";

new Chart(document.getElementById("chartParty"), {{
  type: "bar",
  data: {{
    labels: PARTY_DATA.labels,
    datasets: [
      {{ label: "Democrat", data: PARTY_DATA.democrat, backgroundColor: BLUE, borderColor: BLUE_BORDER, borderWidth: 1 }},
      {{ label: "Republican", data: PARTY_DATA.republican, backgroundColor: RED, borderColor: RED_BORDER, borderWidth: 1 }},
    ]
  }},
  options: {{ plugins: {{ legend: {{ position: "bottom" }} }}, scales: {{ x: {{ stacked: false }}, y: {{ beginAtZero: true }} }} }}
}});

new Chart(document.getElementById("chartTokens"), {{
  type: "bar",
  data: {{
    labels: TOKEN_DATA.labels,
    datasets: [
      {{ label: "Democrat", data: TOKEN_DATA.democrat, backgroundColor: BLUE, borderColor: BLUE_BORDER, borderWidth: 1 }},
      {{ label: "Republican", data: TOKEN_DATA.republican, backgroundColor: RED, borderColor: RED_BORDER, borderWidth: 1 }},
    ]
  }},
  options: {{ plugins: {{ legend: {{ position: "bottom" }} }}, scales: {{ x: {{ stacked: false }}, y: {{ beginAtZero: true, title: {{ display: true, text: "Millions of tokens" }} }} }} }}
}});

function pct(v) {{ return (v * 100).toFixed(1) + "%"; }}
function fmt(n) {{ return n.toLocaleString(); }}

const repTbody = document.getElementById("repTable");
REP_PHRASES.forEach(r => {{
  repTbody.innerHTML += `<tr class="border-b border-gray-50 hover:bg-gray-50">
    <td class="py-2 pr-2 font-mono text-xs">${{r.phrase}}</td>
    <td class="py-2 text-right"><span class="badge-r text-xs px-1.5 py-0.5 rounded">${{pct(r.share)}}</span></td>
    <td class="py-2 text-right text-gray-400">${{fmt(r.count)}}</td>
  </tr>`;
}});

const demTbody = document.getElementById("demTable");
DEM_PHRASES.forEach(r => {{
  demTbody.innerHTML += `<tr class="border-b border-gray-50 hover:bg-gray-50">
    <td class="py-2 pr-2 font-mono text-xs">${{r.phrase}}</td>
    <td class="py-2 text-right"><span class="badge-d text-xs px-1.5 py-0.5 rounded">${{pct(r.share)}}</span></td>
    <td class="py-2 text-right text-gray-400">${{fmt(r.count)}}</td>
  </tr>`;
}});
</script>
</body>
</html>
"""

out_path = os.path.join(OUT_DIR, "index.html")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Built {out_path}")
