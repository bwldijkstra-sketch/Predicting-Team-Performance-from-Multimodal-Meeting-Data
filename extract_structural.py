"""
features/extract_structural.py
================================
Extracts structural participation features from the AMI Meeting Corpus
for each meeting. Features are aggregated at the meeting level across
all four speakers.

Twelve features are computed per meeting:
  1.  total_speaking_time        — sum of all speaker durations (seconds)
  2.  speaking_time_mean         — mean per-speaker speaking time
  3.  speaking_time_std          — SD of per-speaker speaking time
  4.  gini_speaking_time         — Gini coefficient of speaking time
                                   (0 = perfectly equal, 1 = fully dominated)
  5.  max_speaker_share          — proportion of speech held by most active speaker
  6.  silence_ratio              — proportion of meeting time with no speech
  7.  turn_duration_mean         — mean duration of individual turns (seconds)
  8.  turn_duration_std          — SD of turn duration
  9.  num_turns                  — total number of turns across all speakers
  10. turn_taking_frequency      — turns per minute
  11. simultaneous_speech_ratio  — proportion of time with overlapping speech
  12. speaker_change_rate        — speaker changes per minute

All timing information is extracted from the word-level transcript XML files
provided by the AMI corpus (word start/end timestamps).

USAGE
-----
    python features/extract_structural.py

OUTPUT
------
    outputs/ami_structural_features.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from xml.etree import ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import WORDS_DIR, Y_PERFORMANCE_CSV, STRUCTURAL_FEATURES_CSV


# ── Helpers ───────────────────────────────────────────────────────────────────

def gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient of an array of non-negative values."""
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    values = np.sort(values)
    n      = len(values)
    idx    = np.arange(1, n + 1)
    return (2 * (idx * values).sum()) / (n * values.sum()) - (n + 1) / n


def parse_words_file(filepath: str) -> list[tuple[float, float]]:
    """
    Parse a .words.xml file and return a list of (start_time, end_time)
    tuples for each word. Words without timing are skipped.
    """
    if not os.path.exists(filepath):
        return []

    tree      = ET.parse(filepath)
    root      = tree.getroot()
    intervals = []

    for w in root.findall('w'):
        start = w.attrib.get('starttime')
        end   = w.attrib.get('endtime')
        if start and end:
            try:
                intervals.append((float(start), float(end)))
            except ValueError:
                continue

    return intervals


def intervals_to_turns(word_intervals: list[tuple[float, float]],
                       gap_threshold: float = 0.5) -> list[tuple[float, float]]:
    """
    Merge consecutive word intervals into turns.
    A new turn begins when the gap between words exceeds gap_threshold seconds.
    """
    if not word_intervals:
        return []

    word_intervals = sorted(word_intervals)
    turns          = []
    turn_start, turn_end = word_intervals[0]

    for start, end in word_intervals[1:]:
        if start - turn_end <= gap_threshold:
            turn_end = max(turn_end, end)
        else:
            turns.append((turn_start, turn_end))
            turn_start, turn_end = start, end

    turns.append((turn_start, turn_end))
    return turns


# ── Meeting-level feature extraction ─────────────────────────────────────────

def extract_meeting_features(meeting_id: str) -> dict | None:
    """Extract all structural features for one meeting."""
    speakers = ['A', 'B', 'C', 'D']

    speaker_words = {}
    speaker_turns = {}

    for spk in speakers:
        words_file = os.path.join(WORDS_DIR, f'{meeting_id}.{spk}.words.xml')
        intervals  = parse_words_file(words_file)
        if intervals:
            speaker_words[spk] = intervals
            speaker_turns[spk] = intervals_to_turns(intervals)

    if not speaker_turns:
        return None

    # ── Speaking time per speaker ─────────────────────────────────────────────
    speaking_times = {}
    for spk, turns in speaker_turns.items():
        speaking_times[spk] = sum(end - start for start, end in turns)

    times_arr = np.array(list(speaking_times.values()))
    total_speaking_time = times_arr.sum()

    # ── Meeting duration (first word to last word across all speakers) ────────
    all_words = [interval for wlist in speaker_words.values() for interval in wlist]
    if not all_words:
        return None

    meeting_start    = min(s for s, _ in all_words)
    meeting_end      = max(e for _, e in all_words)
    meeting_duration = meeting_end - meeting_start
    if meeting_duration <= 0:
        return None

    # ── Silence ratio ─────────────────────────────────────────────────────────
    # Build a timeline of speech activity across all speakers
    resolution    = 0.1   # seconds
    n_bins        = int(np.ceil(meeting_duration / resolution))
    speech_active = np.zeros(n_bins, dtype=bool)
    overlap_count = np.zeros(n_bins, dtype=int)

    for spk, turns in speaker_turns.items():
        for start, end in turns:
            s_bin = int((start - meeting_start) / resolution)
            e_bin = int((end   - meeting_start) / resolution) + 1
            s_bin = max(0, min(s_bin, n_bins - 1))
            e_bin = max(0, min(e_bin, n_bins))
            speech_active[s_bin:e_bin]  = True
            overlap_count[s_bin:e_bin] += 1

    silence_ratio              = 1.0 - speech_active.mean()
    simultaneous_speech_ratio  = (overlap_count >= 2).mean()

    # ── Turn statistics ───────────────────────────────────────────────────────
    all_turns = sorted(
        [(start, end, spk)
         for spk, turns in speaker_turns.items()
         for start, end in turns]
    )
    all_turn_durations = [end - start for start, end, _ in all_turns]
    num_turns          = len(all_turns)
    turn_duration_mean = np.mean(all_turn_durations) if all_turn_durations else 0.0
    turn_duration_std  = np.std(all_turn_durations)  if all_turn_durations else 0.0

    # Speaker change rate
    speaker_changes = sum(
        1 for i in range(1, len(all_turns))
        if all_turns[i][2] != all_turns[i - 1][2]
    )
    turn_taking_frequency = (num_turns / meeting_duration) * 60
    speaker_change_rate   = (speaker_changes / meeting_duration) * 60

    return {
        'meeting_id':               meeting_id,
        'total_speaking_time':      round(total_speaking_time, 3),
        'speaking_time_mean':       round(times_arr.mean(), 3),
        'speaking_time_std':        round(times_arr.std(),  3),
        'gini_speaking_time':       round(gini(times_arr),  4),
        'max_speaker_share':        round(times_arr.max() / total_speaking_time, 4)
                                    if total_speaking_time > 0 else np.nan,
        'silence_ratio':            round(silence_ratio, 4),
        'turn_duration_mean':       round(turn_duration_mean, 3),
        'turn_duration_std':        round(turn_duration_std,  3),
        'num_turns':                num_turns,
        'turn_taking_frequency':    round(turn_taking_frequency, 3),
        'simultaneous_speech_ratio': round(simultaneous_speech_ratio, 4),
        'speaker_change_rate':      round(speaker_change_rate, 3),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    y_df        = pd.read_csv(Y_PERFORMANCE_CSV)
    meeting_ids = y_df['meeting_id'].tolist()
    print(f"Extracting structural features for {len(meeting_ids)} meetings...\n")

    records, failed = [], []
    for mid in meeting_ids:
        result = extract_meeting_features(mid)
        if result:
            records.append(result)
            print(f"  OK  {mid}")
        else:
            failed.append(mid)
            print(f"  FAIL {mid}")

    struct_df = pd.DataFrame(records)
    struct_df.to_csv(STRUCTURAL_FEATURES_CSV, index=False)

    print(f"\nExtracted : {len(struct_df)} meetings")
    print(f"Failed    : {len(failed)} — {failed}")
    print(f"Features  : {struct_df.shape[1] - 1}")
    print(f"Saved to  : {STRUCTURAL_FEATURES_CSV}")
    print(struct_df.describe().round(3))


if __name__ == "__main__":
    main()
