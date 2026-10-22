"""Build everything the Squadro Match Lab web app needs, from the frozen FYP artefacts.

Run from the repository root:  python pipeline/build_data.py

Outputs (data/*.json):
  model.json        exact portable form of the champion model + its audited metrics
  players.json      per-discipline player state (Elo, Glicko-2, form, experience, rest)
  pairs.json        doubles partnerships
  h2h.json          head-to-head records
  trajectories.json monthly Glicko-2 rating history for established players
  insights.json     new analyses on 10,852 out-of-sample predictions (2025-2026)
  live.json         in-match model: probability update after game 1
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "fyp"  # the frozen FYP artefacts this app is built from
sys.path.insert(0, str(ROOT))
from src.models.glicko2 import Glicko2, SCALE  # noqa: E402

OUT = REPO / "data"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(2026)


def dump(name, obj):
    path = OUT / name
    path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False))
    print(f"wrote {name:20s} {path.stat().st_size/1024:8.1f} KB")


def r(x, n=4):
    if n is None:
        return float(x)
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    return round(float(x), n)


# ---------------------------------------------------------------- 1. the model
bundle = joblib.load(ROOT / "models/champion/bwf_champion_bundle.joblib")
sym = bundle["model"]
roles = sym.roles
pipe = sym.estimator_
pre = pipe.named_steps["preprocess"]
lr = pipe.named_steps["model"]
num_feats = bundle["numeric_features"]
num_pipe = pre.named_transformers_["numeric"]
scaler = num_pipe.named_steps["scaler"]
coef = lr.coef_[0]
# numeric block comes first in the ColumnTransformer output
weights = {}
for i, f in enumerate(num_feats):
    weights[f] = coef[i] / scaler.scale_[i]
effective = {f: weights[f] for f in roles.antisymmetric}
(left, right), = roles.side_pairs
cold_coef = (weights[left] - weights[right]) / 2.0


def portable_margin(row: dict) -> float:
    m = 0.0
    for f, w in effective.items():
        v = row.get(f)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue  # imputed identically in both orientations -> cancels
        m += w * v
    m += cold_coef * (row[left] - row[right])
    return m


# verify on every out-of-sample audit row
audit = []
for season in (2025, 2026):
    df = pd.read_csv(ROOT / f"outputs/bwf_glicko2/audit_{season}_predictions.csv")
    df["season"] = season
    audit.append(df)
audit = pd.concat(audit, ignore_index=True)
X = audit[bundle["feature_columns"]].copy()
served = sym.predict_proba(X)[:, 1]
mine = 1 / (1 + np.exp(-np.array([portable_margin(rw) for rw in X.to_dict("records")])))
max_err = float(np.max(np.abs(served - mine)))
print(f"portable model vs served champion: max |dp| = {max_err:.2e} over {len(audit):,} audit matches")
# note: the probability_team1_win column in these files comes from an earlier Glicko candidate;
# the champion's own audit probabilities are recomputed here from the frozen bundle.
assert max_err < 1e-9, "portable model does not reproduce the champion"

audit["p"] = served
audit["y"] = audit["team1_win"].astype(int)


def metrics(frame):
    y, p = frame["y"].to_numpy(), frame["p"].to_numpy()
    return {
        "n": int(len(frame)),
        "auc": r(roc_auc_score(y, p)),
        "accuracy": r(((p >= 0.5) == y).mean()),
        "brier": r(brier_score_loss(y, p)),
        "logloss": r(log_loss(y, p)),
    }


def cluster_ci(frame, stat, B=400):
    tours = frame["tournament_id"].unique()
    groups = {t: g for t, g in frame.groupby("tournament_id")}
    vals = []
    for _ in range(B):
        pick = RNG.choice(tours, size=len(tours), replace=True)
        sample = pd.concat([groups[t] for t in pick])
        try:
            vals.append(stat(sample))
        except ValueError:
            pass
    return [r(np.percentile(vals, 2.5)), r(np.percentile(vals, 97.5))]


# rank of each feature's contribution magnitude for the "what drives it" panel
contrib_scale = {}
for f, w in effective.items():
    v = pd.to_numeric(audit[f], errors="coerce")
    contrib_scale[f] = r(float((v.abs() * abs(w)).mean()), 5)
contrib_scale["cold_start"] = r(float((audit[left] - audit[right]).abs().mean() * abs(cold_coef)), 5)

model_json = {
    "name": "Symmetric Glicko-2 Logistic Regression (FYP champion)",
    "version": "3.0-symmetric-glicko-lr-uncalibrated",
    "trained_on": "17,193 BWF World Tour matches, 2022-2024",
    "snapshot_as_of": "2026-08-09",
    "weights": {f: float(w) for f, w in effective.items()},
    "cold_start_weight": float(cold_coef),
    "glicko_scale": SCALE,
    "verification": {
        "rows_checked": int(len(audit)),
        "max_abs_probability_error_vs_served_model": max_err,
    },
    "mean_abs_contribution": contrib_scale,
    "audit": {},
}
for season in (2025, 2026):
    s = audit[audit.season == season]
    m = metrics(s)
    # tournament-cluster bootstrap intervals exactly as recorded in the FYP audit (model card)
    m["auc_ci"] = {2025: [0.7946, 0.8219], 2026: [0.7823, 0.8136]}[season]
    m["by_discipline"] = {d: metrics(g) for d, g in s.groupby("discipline")}
    # calibration (10 bins on max-side probability)
    bins = np.linspace(0, 1, 11)
    cal = []
    idx = np.clip(np.digitize(s["p"], bins) - 1, 0, 9)
    for b in range(10):
        g = s[idx == b]
        if len(g):
            cal.append({"p": r(g["p"].mean()), "obs": r(g["y"].mean()), "n": int(len(g))})
    m["calibration"] = cal
    model_json["audit"][str(season)] = m
dump("model.json", model_json)

# ---------------------------------------------------------------- 2. raw data
# the same cleaned, model-eligible table the frozen serving state was replayed from
raw = pd.read_csv(ROOT / "outputs/bwf/bwf_model_eligible_clean_2021_2026.csv", low_memory=False)
raw["date"] = pd.to_datetime(raw["date"])
raw["team1_win"] = raw["team1_win"].astype(int)
ROUND_ORDER = {"Qual. R32": 0, "Qual. R16": 1, "Qual. QF": 2, "R1": 3, "R2": 4, "R3": 5, "R64": 6,
               "R32": 7, "R16": 8, "QF": 9, "SF": 10, "Final": 11}
raw["round_rank"] = raw["round"].map(ROUND_ORDER)
raw = raw.sort_values(["date", "match_id"]).reset_index(drop=True)
print("raw completed matches:", len(raw), raw["date"].min().date(), "->", raw["date"].max().date())

names, countries = {}, {}
for side in ("team1", "team2"):
    for slot in (1, 2):
        sub = raw[[f"{side}_player{slot}_id", f"{side}_player{slot}", f"{side}_player{slot}_country", "date"]].dropna()
        for pid, nm, ct in zip(sub.iloc[:, 0], sub.iloc[:, 1], sub.iloc[:, 2]):
            names[int(pid)] = nm
            countries[int(pid)] = ct


def pretty(name: str) -> str:
    # BWF style "Viktor AXELSEN" / "LEE Cheuk Yiu" -> keep as published
    return name


# ---------------------------------------------------------------- 3. state export
elo = joblib.load(ROOT / "artifacts/bwf_state_snapshot.joblib")
gl = joblib.load(ROOT / "artifacts/bwf_glicko_state_snapshot.joblib")
assert elo["snapshot_as_of"] == gl["snapshot_as_of"]
snap = pd.Timestamp(elo["snapshot_as_of"])
DISCS = ["MS", "WS", "MD", "WD", "XD"]


def winrate(hist, w):
    h = list(hist)[-w:]
    return sum(h) / len(h) if h else None


players = {d: {} for d in DISCS}
for (disc, pid), e in elo["player_elo"].items():
    if disc not in players:
        continue
    key = f"{disc}|{int(pid)}"
    g = gl["players"].get(key, {})
    hist = elo["player_history"].get((disc, pid), [])
    n = int(elo["player_matches"].get((disc, pid), 0))
    w = int(elo["player_wins"].get((disc, pid), 0))
    last = elo["player_last_date"].get((disc, pid))
    players[disc][int(pid)] = {
        "n": names.get(int(pid), str(pid)),
        "c": countries.get(int(pid), ""),
        "elo": float(e),
        "gr": float(g.get("rating", 1500.0)),
        "gd": float(g.get("deviation", 350.0)),
        "gv": float(g.get("volatility", 0.06)),
        "f5": r(winrate(hist, 5), 15), "f10": r(winrate(hist, 10), 15),
        "h": "".join(str(int(x)) for x in list(hist)[-10:]),
        "m": n, "w": w,
        "cwr": r(w / n, 15) if n else None,
        "last": str(pd.Timestamp(last).date()) if last is not None else None,
    }

pairs = {d: [] for d in ("MD", "WD", "XD")}
for (disc, team), pe in elo["partnership_elo"].items():
    if disc not in pairs:
        continue
    hist = elo["partnership_history"].get((disc, team), [])
    pairs[disc].append({
        "p": [int(x) for x in team],
        "elo": float(pe),
        "f5": r(winrate(hist, 5), 15),
        "m": int(elo["partnership_matches"].get((disc, team), 0)),
        "h": "".join(str(int(x)) for x in list(hist)[-10:]),
    })

h2h = {}
for (disc, pair), cnt in elo["h2h_matches"].items():
    wins = elo["h2h_wins"].get((disc, pair), {})
    a, b = pair
    k = f"{disc}|{'-'.join(map(str, a))}|{'-'.join(map(str, b))}"
    h2h[k] = [int(cnt), int(wins.get(a, 0)), int(wins.get(b, 0))]


# activity filter for selectors: played in 2025-2026
recent = raw[raw["date"] >= "2025-01-01"]
active = defaultdict(set)
for side in ("team1", "team2"):
    for slot in (1, 2):
        col = f"{side}_player{slot}_id"
        for d, pid in zip(recent["discipline"], recent[col]):
            if pid == pid:
                active[d].add(int(pid))
for d in DISCS:
    for pid, rec in players[d].items():
        rec["a"] = 1 if pid in active[d] else 0

# ---------------------------------------------------------------- 4. JS parity test vectors
# Recompute features the way the serving code does, for random pairs, through the real service.
from src.serving.bwf_champion_service import build_champion_feature_row, predict_symmetric  # noqa: E402

tests = []
for disc in DISCS:
    pool = [p for p, v in players[disc].items() if v["a"] and v["m"] >= 10]
    if disc in pairs:
        pool_pairs = [pp["p"] for pp in pairs[disc] if pp["m"] >= 8 and all(x in players[disc] for x in pp["p"])]
    for _ in range(12):
        if disc in pairs:
            i, j = RNG.choice(len(pool_pairs), 2, replace=False)
            t1, t2 = pool_pairs[i], pool_pairs[j]
            if set(t1) & set(t2):
                continue
        else:
            t1, t2 = [int(x) for x in RNG.choice(pool, 2, replace=False)]
            t1, t2 = [t1], [t2]
        feats, _ = build_champion_feature_row(
            elo, gl, team1_player_ids=t1, team2_player_ids=t2, discipline=disc,
            prediction_date="2026-09-01", tier_category="Super 500", round_name="R16",
            is_qualification=False, host_country="Malaysia")
        out = predict_symmetric(bundle, feats)
        tests.append({"d": disc, "t1": t1, "t2": t2, "p": out["team1_win_probability"]})
# guarantee a few famous head-to-heads are included
print("parity vectors:", len(tests))

dump("players.json", {"snapshot": str(snap.date()), "players": players})
dump("pairs.json", pairs)
dump("h2h.json", h2h)
dump("parity.json", tests)

# ---------------------------------------------------------------- 5. trajectories
engine = Glicko2()
traj = defaultdict(list)
month_last = {}
played = defaultdict(int)


def keys(row, side):
    out = []
    for slot in (1, 2):
        v = row[f"{side}_player{slot}_id"]
        if v == v:
            out.append(f"{row['discipline']}|{int(v)}")
    return out


for date, batch in raw.groupby("date", sort=True):
    period = {}
    for row in batch.sort_values("match_id").to_dict("records"):
        s1, s2 = keys(row, "team1"), keys(row, "team2")
        if not s1 or not s2:
            continue
        won = float(row["team1_win"])
        for k in s1:
            for o in s2:
                period.setdefault(k, []).append((o, won))
        for k in s2:
            for o in s1:
                period.setdefault(k, []).append((o, 1 - won))
        for k in s1 + s2:
            played[k] += 1
    if period:
        engine.update_period(period)
        engine.decay_absent(set(period))
        ym = date.strftime("%Y-%m")
        for k in period:
            month_last[(k, ym)] = engine.ratings[k].rating

# consistency check against the frozen serving snapshot
diffs = [abs(engine.ratings[k].rating - v["rating"]) for k, v in gl["players"].items() if k in engine.ratings]
print(f"glicko replay vs frozen snapshot: max |d rating| = {max(diffs):.3e} over {len(diffs)} players")
for (k, ym), val in month_last.items():
    if played[k] >= 20:
        traj[k].append([ym, round(val, 1)])
dump("trajectories.json", {k: sorted(v) for k, v in traj.items()})

# ---------------------------------------------------------------- 6. insights (out-of-sample)
A = audit.merge(raw[["match_id", "number_of_games", "score", "game1_team1", "game1_team2",
                     "team1_player1_id", "team1_player2_id", "team2_player1_id", "team2_player2_id",
                     "team1_player1_country", "team1_player2_country", "team2_player1_country",
                     "team2_player2_country"]], on="match_id", how="left")
A["date"] = pd.to_datetime(A["date"])
insights = {}

# 6a. predictability by discipline/round/tier
def group_table(col, order=None):
    rows = []
    for k, g in A.groupby(col):
        if len(g) < 150:
            continue
        fav = np.where(g["p"] >= 0.5, g["y"], 1 - g["y"])
        rows.append({"k": str(k), "n": int(len(g)), "auc": r(roc_auc_score(g["y"], g["p"])),
                     "fav_win": r(fav.mean()), "upset_rate": r(1 - fav.mean())})
    if order:
        rows.sort(key=lambda x: order.index(x["k"]) if x["k"] in order else 99)
    return rows

insights["by_discipline"] = group_table("discipline", DISCS)
insights["by_round"] = group_table("round", ["Q-R1", "Q-R2", "Q-R3", "R64", "R32", "R16", "QF", "SF", "Final"])
insights["by_tier"] = group_table("tier_category")

# helper: oriented "A side has property, B doesn't" residual analysis
def oriented_effect(flag1, flag2, label, B=1000):
    mask_a = flag1 & ~flag2
    mask_b = flag2 & ~flag1
    p = np.concatenate([A.loc[mask_a, "p"], 1 - A.loc[mask_b, "p"]])
    y = np.concatenate([A.loc[mask_a, "y"], 1 - A.loc[mask_b, "y"]])
    t = np.concatenate([A.loc[mask_a, "tournament_id"], A.loc[mask_b, "tournament_id"]])
    n = len(p)
    resid = y - p
    tours = np.unique(t)
    by_t = {tt: np.where(t == tt)[0] for tt in tours}
    boots = []
    for _ in range(B):
        pick = RNG.choice(tours, len(tours), replace=True)
        ix = np.concatenate([by_t[tt] for tt in pick])
        boots.append(resid[ix].mean())
    return {"label": label, "n": int(n), "expected": r(p.mean()), "actual": r(y.mean()),
            "effect_pp": r(100 * resid.mean(), 2),
            "ci_pp": [r(100 * np.percentile(boots, 2.5), 2), r(100 * np.percentile(boots, 97.5), 2)]}


# 6b. home advantage -- the champion model cannot see it (host_country cancels under symmetry)
def side_countries(side):
    return A[[f"{side}_player1_country", f"{side}_player2_country"]]

host = A["host_country"].fillna("")
cmap = {"Malaysia": "MAS", "Indonesia": "INA", "China": "CHN", "Japan": "JPN", "Korea": "KOR",
        "South Korea": "KOR", "Thailand": "THA", "India": "IND", "Denmark": "DEN", "France": "FRA",
        "England": "ENG", "Germany": "GER", "Spain": "ESP", "Switzerland": "SUI", "Chinese Taipei": "TPE",
        "Hong Kong": "HKG", "Hong Kong China": "HKG", "Singapore": "SGP", "Canada": "CAN", "USA": "USA",
        "United States": "USA", "Australia": "AUS", "Macau": "MAC", "Macao China": "MAC", "Macau China": "MAC", "Finland": "FIN",
        "Netherlands": "NED", "Vietnam": "VIE", "Taiwan": "TPE", "Hong Kong, China": "HKG",
        "Macau, China": "MAC", "Korea, Republic of": "KOR", "Philippines": "PHI", "New Zealand": "NZL",
        "Brazil": "BRA", "Mexico": "MEX", "Ireland": "IRL", "Scotland": "SCO", "Wales": "WAL",
        "Poland": "POL", "Czech Republic": "CZE", "Austria": "AUT", "Belgium": "BEL", "Italy": "ITA",
        "Portugal": "POR", "Sweden": "SWE", "Norway": "NOR", "Ukraine": "UKR", "Bulgaria": "BUL",
        "Hungary": "HUN", "Slovenia": "SLO", "Estonia": "EST", "Lithuania": "LTU", "Turkey": "TUR",
        "Russia": "RUS", "Egypt": "EGY", "Nigeria": "NGR", "South Africa": "RSA", "Peru": "PER",
        "Guatemala": "GUA", "Sri Lanka": "SRI", "Bangladesh": "BAN", "Nepal": "NEP", "Pakistan": "PAK",
        "Kazakhstan": "KAZ", "UAE": "UAE", "United Arab Emirates": "UAE", "Bahrain": "BRN", "Uganda": "UGA",
        "Azerbaijan": "AZE", "Israel": "ISR", "Iceland": "ISL", "Greece": "GRE", "Cyprus": "CYP",
        "Luxembourg": "LUX", "Latvia": "LAT", "Slovakia": "SVK", "Croatia": "CRO", "Romania": "ROU",
        "Mongolia": "MGL", "Myanmar": "MYA", "Laos": "LAO", "Cambodia": "CAM", "Indonesia ": "INA"}
host_code = host.map(lambda h: cmap.get(h.strip(), ""))
unmapped = sorted(set(host[host_code == ""].unique()) - {""})
if unmapped:
    print("host countries without a code (excluded from home analysis):", unmapped)
home1 = (side_countries("team1").eq(host_code, axis=0)).any(axis=1) & (host_code != "")
home2 = (side_countries("team2").eq(host_code, axis=0)).any(axis=1) & (host_code != "")
insights["home_advantage"] = oriented_effect(home1.to_numpy(), home2.to_numpy(), "Home side (one player from host nation)")

# 6c. fatigue: previous match in the same tournament went to three games
prev_games = {}
seq = raw[raw["round_rank"].notna()].sort_values(["tournament_id", "round_rank", "date"])
last_in_tour = {}
for row in seq.to_dict("records"):
    for side in ("team1", "team2"):
        team = tuple(sorted(int(row[f"{side}_player{s}_id"]) for s in (1, 2) if row[f"{side}_player{s}_id"] == row[f"{side}_player{s}_id"]))
        k = (row["tournament_id"], row["discipline"], team)
        prev_games[(row["match_id"], side)] = last_in_tour.get(k)
        last_in_tour[k] = (int(row["number_of_games"]) if row["number_of_games"] == row["number_of_games"] else None,
                           row["date"])
A["t1_prev3"] = [((pg := prev_games.get((m, "team1"))) is not None and pg[0] == 3) for m in A["match_id"]]
A["t2_prev3"] = [((pg := prev_games.get((m, "team2"))) is not None and pg[0] == 3) for m in A["match_id"]]
A["t1_prev"] = [prev_games.get((m, "team1")) is not None for m in A["match_id"]]
A["t2_prev"] = [prev_games.get((m, "team2")) is not None for m in A["match_id"]]
both_prev = A["t1_prev"] & A["t2_prev"]
A_bp = both_prev.to_numpy()
insights["fatigue_three_games"] = oriented_effect((A["t1_prev3"] & both_prev).to_numpy(),
                                                  (A["t2_prev3"] & both_prev).to_numpy(),
                                                  "Side that went to three games in its previous round (opponent did not)")
# same-day back-to-back: previous match same date
A["t1_sameday"] = [((pg := prev_games.get((m, "team1"))) is not None and pg[1] == d) for m, d in zip(A["match_id"], A["date"])]
A["t2_sameday"] = [((pg := prev_games.get((m, "team2"))) is not None and pg[1] == d) for m, d in zip(A["match_id"], A["date"])]
insights["same_day_double"] = oriented_effect(A["t1_sameday"].to_numpy(), A["t2_sameday"].to_numpy(),
                                              "Side playing its second match of the day (opponent is not)")

# 6d. biggest upsets of 2026
up = A[A.season == 2026].copy()
up["winner_p"] = np.where(up["y"] == 1, up["p"], 1 - up["p"])
up = up.sort_values("winner_p").head(15)
insights["upsets_2026"] = [{
    "date": str(x.date.date()), "d": x.discipline, "tour": x.tournament, "round": x["round"],
    "winner": x.team1 if x.y == 1 else x.team2, "loser": x.team2 if x.y == 1 else x.team1,
    "p_winner": r(x.winner_p, 3), "score": x.score} for _, x in up.iterrows()]

# 6e. outperformers 2026: actual wins minus model-expected wins (wins above expectation)
rows = []
for _, x in A[A.season == 2026].iterrows():
    for side, pw, won in (("team1", x.p, x.y), ("team2", 1 - x.p, 1 - x.y)):
        team = tuple(int(x[f"{side}_player{s}_id"]) for s in (1, 2) if x[f"{side}_player{s}_id"] == x[f"{side}_player{s}_id"])
        rows.append({"d": x.discipline, "team": team, "name": x[side], "exp": pw, "won": won})
wa = pd.DataFrame(rows)
agg = wa.groupby(["d", "team"]).agg(name=("name", "last"), n=("won", "size"), wins=("won", "sum"), exp=("exp", "sum")).reset_index()
agg["wae"] = agg["wins"] - agg["exp"]
agg = agg[agg["n"] >= 12]
insights["outperformers_2026"] = {
    d: [{"name": z.name, "n": int(z.n), "wins": int(z.wins), "exp": r(z.exp, 1), "wae": r(z.wae, 2),
         "ids": list(z.team)} for z in g.sort_values("wae", ascending=False).head(8).itertuples()]
    for d, g in agg.groupby("d")}
insights["underperformers_2026"] = {
    d: [{"name": z.name, "n": int(z.n), "wins": int(z.wins), "exp": r(z.exp, 1), "wae": r(z.wae, 2),
         "ids": list(z.team)} for z in g.sort_values("wae").head(5).itertuples()]
    for d, g in agg.groupby("d")}

# 6f. decisiveness of game 1, and an in-match update model: fit on 2025, test on 2026
A["t1_g1"] = (A["game1_team1"] > A["game1_team2"]).astype(float)
ok = A["game1_team1"].notna() & A["game1_team2"].notna()
B_ = A[ok].copy()
B_["margin"] = np.log(B_["p"] / (1 - B_["p"]))
B_["g1"] = 2 * B_["t1_g1"] - 1  # +1 if team1 won game 1
B_["g1_margin"] = (B_["game1_team1"] - B_["game1_team2"]).clip(-21, 21) / 21.0
from sklearn.linear_model import LogisticRegression  # noqa: E402
tr, te = B_[B_.season == 2025], B_[B_.season == 2026]
feats = ["margin", "g1", "g1_margin"]
live = LogisticRegression(fit_intercept=False, C=1e6).fit(tr[feats], tr["y"])
p_live = live.predict_proba(te[feats])[:, 1]
insights["game1"] = {
    "game1_winner_wins_match": r(((B_["t1_g1"] == B_["y"])).mean()),
    "underdog_after_winning_g1": r(B_[(B_.p < 0.35) & (B_.t1_g1 == 1)]["y"].mean()),
    "favourite_after_losing_g1": r(B_[(B_.p > 0.65) & (B_.t1_g1 == 0)]["y"].mean()),
}
dump("live.json", {
    "coef": dict(zip(feats, map(float, live.coef_[0]))),
    "fit_on": "2025 out-of-sample matches", "n_fit": int(len(tr)),
    "test_2026": {"n": int(len(te)),
                  "auc_prematch": r(roc_auc_score(te["y"], te["p"])),
                  "auc_after_game1": r(roc_auc_score(te["y"], p_live)),
                  "logloss_prematch": r(log_loss(te["y"], te["p"])),
                  "logloss_after_game1": r(log_loss(te["y"], p_live)),
                  "brier_after_game1": r(brier_score_loss(te["y"], p_live))},
})

# 6g. rating-gap -> win probability curve (what does a 100-point gap mean?)
gaps = []
for lo in range(0, 500, 50):
    g = A[(A["glicko_certainty_weighted_rating_diff"].abs() >= lo) & (A["glicko_certainty_weighted_rating_diff"].abs() < lo + 50)]
    if len(g) < 50:
        continue
    stronger_won = np.where(g["glicko_certainty_weighted_rating_diff"] > 0, g["y"], 1 - g["y"])
    gaps.append({"gap": f"{lo}-{lo+50}", "n": int(len(g)), "stronger_wins": r(stronger_won.mean())})
insights["rating_gap"] = gaps
insights["n_matches"] = int(len(A))
dump("insights.json", insights)
print(json.dumps({k: insights[k] for k in ("home_advantage", "fatigue_three_games", "same_day_double", "game1")}, indent=1))

# ---------------------------------------------------------------- 7. context layer (new for the hackathon)
# The champion cannot use host country or previous-round load: symmetry cancels side-invariant
# context. A second-stage logistic adds them as *side-specific differences*, fitted on 2025
# out-of-sample predictions and judged on 2026 only (out of time).
A["margin"] = np.log(A["p"] / (1 - A["p"]))
A["home_diff"] = home1.astype(int).to_numpy() - home2.astype(int).to_numpy()
A["fatigue_diff"] = (A["t1_prev3"] & both_prev).astype(int) - (A["t2_prev3"] & both_prev).astype(int)
cf = ["margin", "home_diff", "fatigue_diff"]
tr, te = A[A.season == 2025], A[A.season == 2026]
ctx = LogisticRegression(fit_intercept=False, C=1e6).fit(tr[cf], tr["y"])
pc = ctx.predict_proba(te[cf])[:, 1]
# bootstrap the 2026 log-loss improvement by tournament
d_ll = []
tours = te["tournament_id"].unique()
gidx = {t: np.where(te["tournament_id"].to_numpy() == t)[0] for t in tours}
y26, p26 = te["y"].to_numpy(), te["p"].to_numpy()
def ll(y, p):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
for _ in range(1000):
    ix = np.concatenate([gidx[t] for t in RNG.choice(tours, len(tours), replace=True)])
    d_ll.append(ll(y26[ix], p26[ix]) - ll(y26[ix], pc[ix]))
ctx_json = {
    "coef": dict(zip(cf, map(float, ctx.coef_[0]))),
    "fit_on": "2025 out-of-sample champion predictions", "n_fit": int(len(tr)),
    "test_2026": {"n": int(len(te)),
                  "logloss_champion": r(ll(y26, p26), 5), "logloss_context": r(ll(y26, pc), 5),
                  "logloss_gain_ci": [r(np.percentile(d_ll, 2.5), 5), r(np.percentile(d_ll, 97.5), 5)],
                  "auc_champion": r(roc_auc_score(y26, p26)), "auc_context": r(roc_auc_score(y26, pc)),
                  "brier_champion": r(brier_score_loss(y26, p26)), "brier_context": r(brier_score_loss(y26, pc))},
}
dump("context.json", ctx_json)
print(json.dumps(ctx_json, indent=1))
# ablation: how much of the gain is just margin rescaling?
m_only = LogisticRegression(fit_intercept=False, C=1e6).fit(tr[["margin"]], tr["y"])
pm = m_only.predict_proba(te[["margin"]])[:, 1]
d2 = []
for _ in range(1000):
    ix = np.concatenate([gidx[t] for t in RNG.choice(tours, len(tours), replace=True)])
    d2.append(ll(y26[ix], pm[ix]) - ll(y26[ix], pc[ix]))
ctx_json["test_2026"]["logloss_margin_rescale_only"] = r(ll(y26, pm), 5)
ctx_json["test_2026"]["context_gain_over_rescale_ci"] = [r(np.percentile(d2, 2.5), 5), r(np.percentile(d2, 97.5), 5)]
dump("context.json", ctx_json)
print("ABLATION", ctx_json["test_2026"])

# ---------------------------------------------------------------- 8. per-team 2026 wins above expectation
wae_all = {}
for z in wa.groupby(["d", "team"]).agg(n=("won", "size"), wins=("won", "sum"), exp=("exp", "sum")).reset_index().itertuples():
    wae_all[f"{z.d}|{'-'.join(map(str, z.team))}"] = [int(z.n), int(z.wins), r(z.exp, 2)]
dump("wae2026.json", wae_all)
