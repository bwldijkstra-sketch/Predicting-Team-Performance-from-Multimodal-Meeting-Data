"""
thin_slice.py
-------------
RQ3: Thin-slice analysis — replicates the full modelling pipeline on
features extracted from only the first 10 minutes (600 seconds) of
each AMI meeting.

Steps:
    1. Re-extract DA and sentiment features from truncated transcripts
    2. Re-extract structural features from truncated transcripts
    3. Merge with target variables
    4. Run nested cross-validation (same setup as full-session models)
    5. Compare thin-slice vs full-session performance

Outputs (saved to RESULTS_DIR/thinslice/):
    thinslice_features.csv          — feature matrix for 10-min window
    thinslice_model_comparison.csv  — R² and MAE per model/feature set
    thinslice_vs_fullsession.csv    — direct comparison table
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# Import repository configurations
import config

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration & Directory Setup
# =============================================================================

WINDOW_SECONDS = 600   # 10 minutes

THINSLICE_DIR = os.path.join(config.OUT_DIR, "results", "thinslice")
RESULTS_DIR   = os.path.join(config.OUT_DIR, "results")
os.makedirs(THINSLICE_DIR, exist_ok=True)

# Use unified DA mapping from config to maintain multi-script consistency
DA_TYPE_MAP = config.DA_TYPE_MAP 


# =============================================================================
# Helper Functions (Timestamp-Aware Extraction)
# =============================================================================

def parse_words_file_timed(words_path, max_time=None):
    """
    Parse words XML and return:
        word_map  : nite_id -> word text
        time_map  : nite_id -> starttime (float seconds)
    Optionally filter to words with starttime < max_time.
    """
    if not os.path.exists(words_path):
        return {}, {}
    tree = ET.parse(words_path)
    root = tree.getroot()
    word_map = {}
    time_map = {}
    for elem in root.iter():
        nite_id = elem.get("{http://nite.sourceforge.net/}id")
        if nite_id is None:
            continue
        try:
            starttime = float(elem.get("starttime", -1))
        except (ValueError, TypeError):
            starttime = -1

        if max_time is not None and starttime >= max_time:
            continue   # outside window

        if elem.tag == "w":
            word_map[nite_id] = (elem.text or "").strip()
        else:
            word_map[nite_id] = ""
        time_map[nite_id] = starttime

    return word_map, time_map


def extract_word_ids_from_child(child_href):
    if "#" not in child_href:
        return []
    fragment = child_href.split("#", 1)[1]
    range_match = re.match(r"id\(([^)]+)\)\.\.id\(([^)]+)\)", fragment)
    if range_match:
        start_id = range_match.group(1)
        end_id   = range_match.group(2)
        prefix   = re.sub(r"\d+$", "", start_id)
        start_n  = int(re.search(r"(\d+)$", start_id).group(1))
        end_n    = int(re.search(r"(\d+)$", end_id).group(1))
        return [f"{prefix}{n}" for n in range(start_n, end_n + 1)]
    single_match = re.match(r"id\(([^)]+)\)", fragment)
    if single_match:
        return [single_match.group(1)]
    return []


def get_da_type(dact_elem):
    ns_nite = "http://nite.sourceforge.net/"
    for child in dact_elem:
        tag = child.tag.replace(f"{{{ns_nite}}}", "")
        if tag == "pointer":
            role = child.get("role", "")
            if role == "da-aspect":
                href = child.get("href", "")
                m = re.search(r"id\(([^)]+)\)", href)
                if m:
                    raw = m.group(1)
                    return DA_TYPE_MAP.get(raw, raw)
    return "unknown"


def extract_meeting_thinslice(meeting_id, ami_root, max_time=600, speakers=["A", "B", "C", "D"]):
    """
    Extract utterances from the first max_time seconds of a meeting.
    An utterance is included if at least one of its words falls within the time window.
    """
    da_dir    = os.path.join(ami_root, "dialogueActs")
    words_dir = os.path.join(ami_root, "words")
    ns_nite   = "http://nite.sourceforge.net/"
    utterances = []

    for speaker in speakers:
        da_file    = os.path.join(da_dir,    f"{meeting_id}.{speaker}.dialog-act.xml")
        words_file = os.path.join(words_dir, f"{meeting_id}.{speaker}.words.xml")

        if not os.path.exists(da_file):
            continue

        # Load all words with timestamps, filtered to window
        word_map, time_map = parse_words_file_timed(words_file, max_time=max_time)

        tree = ET.parse(da_file)
        root = tree.getroot()

        for dact in root.iter("dact"):
            da_type = get_da_type(dact)
            tokens  = []
            in_window = False

            for child_elem in dact:
                tag = child_elem.tag.replace(f"{{{ns_nite}}}", "")
                if tag == "child":
                    href     = child_elem.get("href", "")
                    word_ids = extract_word_ids_from_child(href)
                    for wid in word_ids:
                        if wid in word_map:   # only words within window
                            tokens.append(word_map.get(wid, ""))
                            in_window = True

            if not in_window:
                continue   # entire utterance outside window

            text = " ".join(t for t in tokens if t).strip()
            utterances.append({
                "meeting_id": meeting_id,
                "speaker":    speaker,
                "da_type":    da_type,
                "text":       text,
                "n_words":    len(text.split()) if text else 0,
            })

    return utterances


def compute_da_distribution(utterances_df):
    counts = (
        utterances_df
        .groupby(["meeting_id", "da_type"])
        .size()
        .unstack(fill_value=0)
    )
    proportions = counts.div(counts.sum(axis=1), axis=0)
    proportions.columns = [f"da_prop_{c}" for c in proportions.columns]
    return proportions.reset_index()


def score_utterance(text, analyzer):
    scores = analyzer.polarity_scores(str(text))
    return pd.Series({
        "vader_pos":       scores["pos"],
        "vader_neg":       scores["neg"],
        "vader_neu":       scores["neu"],
        "vader_compound": scores["compound"],
    })


def extract_structural_thinslice(meeting_id, ami_root, max_time=600, speakers=["A", "B", "C", "D"]):
    """
    Compute structural features from the first max_time seconds only.
    Uses word-level timestamps to reconstruct speaking segments.
    """
    words_dir = os.path.join(ami_root, "words")
    segments  = []   # list of (speaker, starttime, endtime)

    for speaker in speakers:
        words_file = os.path.join(words_dir, f"{meeting_id}.{speaker}.words.xml")
        if not os.path.exists(words_file):
            continue

        tree = ET.parse(words_file)
        root = tree.getroot()

        current_start = None
        current_end   = None

        for elem in root.iter("w"):
            try:
                st = float(elem.get("starttime", -1))
                et = float(elem.get("endtime",   -1))
            except (ValueError, TypeError):
                continue

            if st < 0 or st >= max_time:
                continue

            et = min(et, max_time)

            if current_start is None:
                current_start = st
                current_end   = et
            elif st - current_end < 0.3:   # merge if gap < 300ms
                current_end = max(current_end, et)
            else:
                segments.append((speaker, current_start, current_end))
                current_start = st
                current_end   = et

        if current_start is not None:
            segments.append((speaker, current_start, current_end))

    if len(segments) == 0:
        return None

    seg_df = pd.DataFrame(segments, columns=["speaker", "start", "end"])
    seg_df["duration"] = seg_df["end"] - seg_df["start"]
    seg_df = seg_df[seg_df["duration"] > 0]

    total_duration = max_time
    speak_time = seg_df.groupby("speaker")["duration"].sum()

    # Gini coefficient calculation
    times = speak_time.values
    if len(times) > 1 and times.sum() > 0:
        times_sorted = np.sort(times)
        n = len(times_sorted)
        gini = (2 * np.sum((np.arange(1, n+1)) * times_sorted) / (n * times_sorted.sum()) - (n+1)/n)
    else:
        gini = 0.0

    turn_durations = seg_df["duration"].values

    # Overlap calculation
    seg_sorted = seg_df.sort_values("start").reset_index(drop=True)
    overlap_time = 0.0
    for i in range(1, len(seg_sorted)):
        prev_end = seg_sorted.loc[:i-1, "end"].max()
        curr_start = seg_sorted.loc[i, "start"]
        if curr_start < prev_end:
            overlap_time += prev_end - curr_start

    total_speech = seg_df["duration"].sum()
    silence_time = total_duration - total_speech
    silence_ratio = max(silence_time, 0) / total_duration

    return {
        "meeting_id":              meeting_id,
        "speaking_time_mean":    speak_time.mean(),
        "speaking_time_std":     speak_time.std() if len(speak_time) > 1 else 0,
        "gini_speaking_time":    gini,
        "max_speaker_share":     speak_time.max() / speak_time.sum() if speak_time.sum() > 0 else 0,
        "num_turns":             len(seg_df),
        "turns_per_minute":      len(seg_df) / (total_duration / 60),
        "mean_turn_duration":    turn_durations.mean(),
        "std_turn_duration":     turn_durations.std() if len(turn_durations) > 1 else 0,
        "overlap_ratio":         overlap_time / total_duration,
        "num_interruptions_proxy": int(overlap_time / total_duration * len(seg_df)),
        "silence_ratio":         silence_ratio,
        "avg_pause_duration":    silence_time / max(len(seg_df), 1),
    }


def get_models():
    return {
        "ENR": (
            Pipeline([("scaler", StandardScaler()),
                      ("model",  ElasticNet(max_iter=10000, random_state=42))]),
            {"model__alpha":    [0.001, 0.01, 0.1, 1.0, 10.0],
             "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
        ),
        "SVR": (
            Pipeline([("scaler", StandardScaler()),
                      ("model",  SVR(kernel="rbf"))]),
            {"model__C":       [0.1, 1.0, 10.0, 100.0],
             "model__gamma":   ["scale", "auto", 0.01, 0.1],
             "model__epsilon": [0.01, 0.1, 0.5]},
        ),
        "RFR": (
            Pipeline([("scaler", StandardScaler()),
                      ("model",  RandomForestRegressor(random_state=42, n_jobs=-1))]),
            {"model__n_estimators":     [100, 200],
             "model__max_depth":        [None, 5, 10],
             "model__min_samples_leaf": [1, 2, 5]},
        ),
        "XGB": (
            Pipeline([("scaler", StandardScaler()),
                      ("model",  XGBRegressor(random_state=42, n_jobs=-1, verbosity=0))]),
            {"model__n_estimators":  [100, 200],
             "model__max_depth":     [3, 5, 7],
             "model__learning_rate": [0.01, 0.1, 0.2],
             "model__subsample":     [0.8, 1.0]},
        ),
    }


# =============================================================================
# Main Pipeline Run Execution
# =============================================================================

def main():
    da_dir = os.path.join(config.AMI_ROOT, "dialogueActs")
    files  = os.listdir(da_dir)
    meeting_ids = sorted(set(f.split(".")[0] for f in files if f.endswith(".dialog-act.xml")))
    print(f"Found {len(meeting_ids)} meetings")

    # --- 1. Extract DA and Sentiment Features ---
    all_utterances = []
    failed = []
    for mid in meeting_ids:
        try:
            utts = extract_meeting_thinslice(mid, config.AMI_ROOT, max_time=WINDOW_SECONDS)
            if len(utts) == 0:
                failed.append((mid, "no utterances in window"))
                continue
            all_utterances.extend(utts)
        except Exception as e:
            failed.append((mid, str(e)))

    utterances_df = pd.DataFrame(all_utterances)
    print(f"Total utterances in window: {len(utterances_df)}")
    print(f"Meetings with data: {utterances_df['meeting_id'].nunique()}")
    print(f"Failed/empty: {len(failed)} — {[m for m,_ in failed]}")

    da_features = compute_da_distribution(utterances_df)
    da_features = da_features.drop(columns=["da_prop_unknown"], errors="ignore")
    print(f"DA features shape: {da_features.shape}")

    analyzer = SentimentIntensityAnalyzer()
    utts_clean = utterances_df[utterances_df["text"].notna() & (utterances_df["text"].str.strip() != "")].copy()

    print("Scoring sentiment...")
    sentiment_scores = utts_clean["text"].apply(lambda t: score_utterance(t, analyzer))
    utts_clean = pd.concat([utts_clean, sentiment_scores], axis=1)

    sentiment_features = (
        utts_clean
        .groupby("meeting_id")
        .agg(
            vader_pos_mean      = ("vader_pos",      "mean"),
            vader_pos_std       = ("vader_pos",      "std"),
            vader_neg_mean      = ("vader_neg",      "mean"),
            vader_neg_std       = ("vader_neg",      "std"),
            vader_neu_mean      = ("vader_neu",      "mean"),
            vader_neu_std       = ("vader_neu",      "std"),
            vader_compound_mean = ("vader_compound", "mean"),
            vader_compound_std  = ("vader_compound", "std"),
        )
        .reset_index()
    )
    print(f"Sentiment features shape: {sentiment_features.shape}")

    # --- 2. Extract Structural Features ---
    print("Extracting structural features from 10-min window...")
    structural_rows = []
    for mid in meeting_ids:
        row = extract_structural_thinslice(mid, config.AMI_ROOT, max_time=WINDOW_SECONDS)
        if row is not None:
            structural_rows.append(row)

    structural_ts = pd.DataFrame(structural_rows)
    print(f"Structural thin-slice shape: {structural_ts.shape}")

    # --- 3. Merge Features & Target Labels ---
    y = pd.read_csv(config.Y_CSV, sep=None, engine="python")
    target_cols = ["overall_performance", "satisfaction", "cohesiveness", "leadership", "information_processing"]

    thinslice_features = (
        da_features
        .merge(sentiment_features, on="meeting_id", how="inner")
        .merge(structural_ts,      on="meeting_id", how="inner")
        .merge(y[["meeting_id"] + target_cols], on="meeting_id", how="inner")
    )

    print(f"\nThin-slice feature matrix shape: {thinslice_features.shape}")
    print(f"Missing values: {thinslice_features.isnull().sum().sum()}")

    thinslice_features.to_csv(os.path.join(THINSLICE_DIR, "thinslice_features.csv"), index=False)
    print("Saved thinslice_features.csv")

    # --- 4. Nested Cross-Validation Modelling Pipeline ---
    STRUCTURAL_COLS = [
        "speaking_time_mean", "speaking_time_std", "gini_speaking_time",
        "max_speaker_share", "num_turns", "turns_per_minute",
        "mean_turn_duration", "std_turn_duration", "overlap_ratio",
        "num_interruptions_proxy", "silence_ratio", "avg_pause_duration",
    ]
    DA_COLS        = [c for c in thinslice_features.columns if c.startswith("da_prop_")]
    SENTIMENT_COLS = [
        "vader_pos_mean", "vader_pos_std", "vader_neg_mean", "vader_neg_std",
        "vader_neu_mean", "vader_neu_std", "vader_compound_mean", "vader_compound_std",
    ]
    MULTIMODAL_COLS = STRUCTURAL_COLS + DA_COLS + SENTIMENT_COLS

    FEATURE_SETS = {
        "structural": STRUCTURAL_COLS,
        "da":         DA_COLS,
        "sentiment":  SENTIMENT_COLS,
        "multimodal": MULTIMODAL_COLS,
    }

    RANDOM_STATE = 42
    N_OUTER      = 10
    N_INNER      = 5

    all_results   = []
    total_runs   = len(target_cols) * len(FEATURE_SETS) * 4
    run          = 0

    for target in target_cols:
        y_raw    = thinslice_features[target].values
        y_binned = pd.qcut(y_raw, q=3, labels=False, duplicates="drop")
        outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=RANDOM_STATE)

        for fset_name, fset_cols in FEATURE_SETS.items():
            fset_cols_available = [c for c in fset_cols if c in thinslice_features.columns]
            X_raw = thinslice_features[fset_cols_available].values

            for model_name, (pipeline, param_grid) in get_models().items():
                run += 1
                print(f"[{run}/{total_runs}] {target} | {fset_name} | {model_name}")

                fold_r2  = []
                fold_mae = []

                for train_idx, test_idx in outer_cv.split(X_raw, y_binned):
                    X_train, X_test = X_raw[train_idx], X_raw[test_idx]
                    y_train, y_test = y_raw[train_idx], y_raw[test_idx]
                    y_train_binned  = y_binned[train_idx]

                    inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=RANDOM_STATE)
                    grid_search = GridSearchCV(
                        estimator  = pipeline,
                        param_grid = param_grid,
                        cv         = inner_cv.split(X_train, y_train_binned),
                        scoring    = "r2",
                        n_jobs     = -1,
                        refit      = True,
                    )
                    grid_search.fit(X_train, y_train)
                    y_pred = grid_search.predict(X_test)

                    fold_r2.append(r2_score(y_test, y_pred))
                    fold_mae.append(mean_absolute_error(y_test, y_pred))

                all_results.append({
                    "target":      target,
                    "feature_set": fset_name,
                    "model":       model_name,
                    "mean_r2":     round(np.mean(fold_r2),   4),
                    "std_r2":      round(np.std(fold_r2),    4),
                    "mean_mae":    round(np.mean(fold_mae),  4),
                    "std_mae":     round(np.std(fold_mae),   4),
                })
                print(f"   R²={np.mean(fold_r2):.3f} ± {np.std(fold_r2):.3f}  "
                      f"MAE={np.mean(fold_mae):.3f} ± {np.std(fold_mae):.3f}")

    print("\nAll runs complete.")
    ts_results = pd.DataFrame(all_results)
    ts_results.to_csv(os.path.join(THINSLICE_DIR, "thinslice_model_comparison.csv"), index=False)
    print("Saved thinslice_model_comparison.csv")

    # --- 5. Compare Thin-Slice vs Full-Session ---
    full_results = pd.read_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), sep=None, engine="python")
    full_results = full_results.iloc[:, :7].copy()
    full_results.columns = ["target", "feature_set", "model", "mean_r2", "std_r2", "mean_mae", "std_mae"]
    
    for col in ["mean_r2", "std_r2", "mean_mae", "std_mae"]:
        full_results[col] = full_results[col].astype(str).str.replace(",", ".").astype(float)

    comparison_rows = []
    for target in target_cols:
        fs = full_results[(full_results["target"] == target) & (full_results["feature_set"] == "multimodal")]
        fs_best = fs.loc[fs["mean_r2"].idxmax()]

        ts = ts_results[(ts_results["target"] == target) & (ts_results["feature_set"] == "multimodal")]
        ts_best = ts.loc[ts["mean_r2"].idxmax()]

        comparison_rows.append({
            "target":             target,
            "full_model":         fs_best["model"],
            "full_r2":            round(fs_best["mean_r2"], 4),
            "full_mae":           round(fs_best["mean_mae"], 4),
            "thinslice_model":    ts_best["model"],
            "thinslice_r2":       round(ts_best["mean_r2"], 4),
            "thinslice_mae":      round(ts_best["mean_mae"], 4),
            "r2_retention":       round(ts_best["mean_r2"] / fs_best["mean_r2"], 3) if fs_best["mean_r2"] > 0 else "N/A",
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(os.path.join(THINSLICE_DIR, "thinslice_vs_fullsession.csv"), index=False)

    print("\n=== THIN-SLICE vs FULL-SESSION (best multimodal per target) ===")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()