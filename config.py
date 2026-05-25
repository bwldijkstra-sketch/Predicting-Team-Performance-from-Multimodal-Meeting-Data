"""
config.py
=========
Central configuration for all data paths.
Edit this file once before running any script.
"""

import os

# ── Root path to your downloaded AMI corpus ───────────────────────────────────
# Download from: http://corpus.amiproject.org/
AMI_ROOT = r"path/to/ami_corpus"          # <- change this

# ── Sub-directories (do not change unless your layout differs) ────────────────
DIALOGUE_ACTS_DIR = os.path.join(AMI_ROOT, "dialogueActs")
WORDS_DIR         = os.path.join(AMI_ROOT, "words")
DECISION_DIR      = os.path.join(AMI_ROOT, "decision", "manual")
MEETINGS_XML      = os.path.join(AMI_ROOT, "meetings.xml")
PARTICIPANTS_XML  = os.path.join(AMI_ROOT, "participants.xml")

# ── Output directory for extracted feature CSVs ───────────────────────────────
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Feature file paths ────────────────────────────────────────────────────────
STRUCTURAL_FEATURES_CSV = os.path.join(OUTPUT_DIR, "ami_structural_features.csv")
SEMANTIC_FEATURES_CSV   = os.path.join(OUTPUT_DIR, "ami_semantic_features.csv")
Y_PERFORMANCE_CSV       = os.path.join(OUTPUT_DIR, "y_performance_final.csv")
Y_DECISIONS_CSV         = os.path.join(OUTPUT_DIR, "y_variable_decisions.csv")
