"""
extract_semantic_features.py
=============================
Extracts semantic features from the AMI Meeting Corpus XML files.

Two feature categories are produced:
    1. Dialogue-act (DA) distributions  — proportion of each DA type per meeting
    2. VADER sentiment features          — mean and SD of utterance-level
                                           positive, negative, neutral, and
                                           compound polarity scores per meeting

Output files (saved to OUT_DIR):
    utterances.csv         — utterance-level text with DA labels (intermediate)
    da_features.csv        — meeting-level DA proportion features
    sentiment_features.csv — meeting-level VADER sentiment features

Requirements:
    pip install vaderSentiment pandas

AMI corpus folder structure expected:
    <AMI_ROOT>/
        dialogueActs/    -> <meeting>.<speaker>.dialog-act.xml
        words/           -> <meeting>.<speaker>.words.xml

Usage:
    1. Set AMI_ROOT and OUT_DIR in the CONFIGURATION block below
    2. Run: python extract_semantic_features.py

Author: [Your Name]
Date:   [Date]
"""

import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# =============================================================================
# CONFIGURATION — update these two paths before running
# =============================================================================

AMI_ROOT = r"C:\path\to\your\AMI"    # root folder of the AMI corpus
OUT_DIR  = r"C:\path\to\your\output" # folder where CSV files will be saved


# =============================================================================
# DA TYPE MAPPING
# Derived from AMI ontology file (ontologies/da-types.xml)
# =============================================================================

DA_TYPE_MAP = {
    "ami_da_1":  "backchannel",
    "ami_da_2":  "stall",
    "ami_da_3":  "fragment",
    "ami_da_4":  "inform",
    "ami_da_5":  "elicit_inform",
    "ami_da_6":  "suggest",
    "ami_da_7":  "offer",
    "ami_da_8":  "elicit_suggest",
    "ami_da_9":  "assess",
    "ami_da_11": "elicit_assessment",
    "ami_da_12": "comment_understanding",
    "ami_da_13": "elicit_comment_understanding",
    "ami_da_14": "be_positive",
    "ami_da_15": "be_negative",
    "ami_da_16": "other",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_words_file(words_path):
    """
    Parse a words XML file and return a dict mapping nite:id -> word text.
    Non-word tokens are mapped to empty string.
    """
    if not os.path.exists(words_path):
        return {}
    tree = ET.parse(words_path)
    root = tree.getroot()
    word_map = {}
    for elem in root.iter():
        nite_id = elem.get("{http://nite.sourceforge.net/}id")
        if nite_id is None:
            continue
        word_map[nite_id] = (elem.text or "").strip() if elem.tag == "w" else ""
    return word_map


def extract_word_ids_from_child(child_href):
    """
    Parse a nite:child href and return a list of word nite:ids.
    Handles both range and single-word syntax.
    """
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
    """
    Extract the DA type label from a <dact> element via nite:pointer.
    Returns a readable label via DA_TYPE_MAP.
    """
    ns_nite = "http://nite.sourceforge.net/"
    for child in dact_elem:
        tag = child.tag.replace(f"{{{ns_nite}}}", "")
        if tag == "pointer":
            if child.get("role", "") == "da-aspect":
                href = child.get("href", "")
                m = re.search(r"id\(([^)]+)\)", href)
                if m:
                    return DA_TYPE_MAP.get(m.group(1), m.group(1))
    return "unknown"


def extract_meeting_utterances(meeting_id, ami_root,
                                speakers=["A", "B", "C", "D"]):
    """
    Extract all utterances from a single meeting.
    Returns a list of dicts: meeting_id, speaker, da_type, text, n_words.
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

        word_map = parse_words_file(words_file)
        tree     = ET.parse(da_file)
        root     = tree.getroot()

        for dact in root.iter("dact"):
            da_type = get_da_type(dact)
            tokens  = []
            for child_elem in dact:
                tag = child_elem.tag.replace(f"{{{ns_nite}}}", "")
                if tag == "child":
                    href     = child_elem.get("href", "")
                    word_ids = extract_word_ids_from_child(href)
                    for wid in word_ids:
                        tokens.append(word_map.get(wid, ""))
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
    """
    Compute per-meeting proportions of each DA type.
    Returns a DataFrame with one row per meeting.
    """
    counts = (
        utterances_df
        .groupby(["meeting_id", "da_type"])
        .size()
        .unstack(fill_value=0)
    )
    proportions = counts.div(counts.sum(axis=1), axis=0)
    proportions.columns = [f"da_prop_{c}" for c in proportions.columns]
    return proportions.reset_index()


def compute_sentiment_features(utterances_df):
    """
    Compute per-meeting VADER sentiment features (mean and SD of pos, neg,
    neu, compound) aggregated from utterance-level scores.
    Returns a DataFrame with one row per meeting and 8 columns.
    """
    analyzer = SentimentIntensityAnalyzer()

    def score(text):
        s = analyzer.polarity_scores(str(text))
        return pd.Series({
            "vader_pos":      s["pos"],
            "vader_neg":      s["neg"],
            "vader_neu":      s["neu"],
            "vader_compound": s["compound"],
        })

    utts = utterances_df[
        utterances_df["text"].notna() &
        (utterances_df["text"].str.strip() != "")
    ].copy()

    print("  Scoring utterances with VADER...")
    scores = utts["text"].apply(score)
    utts   = pd.concat([utts, scores], axis=1)

    return (
        utts
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


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Discover meeting IDs
    da_dir = os.path.join(AMI_ROOT, "dialogueActs")
    files  = os.listdir(da_dir)
    meeting_ids = sorted(set(
        f.split(".")[0] for f in files if f.endswith(".dialog-act.xml")
    ))
    print(f"Found {len(meeting_ids)} meetings")

    # Extract utterances
    all_utterances = []
    failed         = []
    for mid in meeting_ids:
        try:
            utts = extract_meeting_utterances(mid, AMI_ROOT)
            all_utterances.extend(utts)
        except Exception as e:
            failed.append((mid, str(e)))

    print(f"Extraction complete: {len(meeting_ids)-len(failed)} meetings, "
          f"{len(all_utterances)} utterances, {len(failed)} failed")
    if failed:
        print(f"Failed: {[m for m, _ in failed]}")

    utterances_df = pd.DataFrame(all_utterances)

    # DA distributions
    print("\nComputing DA distributions...")
    da_features = compute_da_distribution(utterances_df)
    da_features = da_features.drop(columns=["da_prop_unknown"], errors="ignore")
    print(f"  Shape: {da_features.shape}")

    # VADER sentiment
    print("\nComputing VADER sentiment features...")
    sentiment_features = compute_sentiment_features(utterances_df)
    print(f"  Shape: {sentiment_features.shape}")

    # Save
    utterances_df.to_csv(os.path.join(OUT_DIR, "utterances.csv"),         index=False)
    da_features.to_csv(os.path.join(OUT_DIR, "da_features.csv"),           index=False)
    sentiment_features.to_csv(os.path.join(OUT_DIR, "sentiment_features.csv"), index=False)

    print(f"\nSaved to {OUT_DIR}:")
    print(f"  utterances.csv         ({len(utterances_df)} rows)")
    print(f"  da_features.csv        ({len(da_features)} rows)")
    print(f"  sentiment_features.csv ({len(sentiment_features)} rows)")


if __name__ == "__main__":
    main()