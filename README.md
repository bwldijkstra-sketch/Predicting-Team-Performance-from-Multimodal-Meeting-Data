# Predicting-Team-Performance-from-Multimodal-Meeting-Data
# Predicting Team Performance from Multimodal Meeting Data

Master's thesis — Data Science & Society, Tilburg University, 2025
Author: [Bram Dijkstra]

## Overview
This repository contains all analysis code for the thesis 
"[Predicting team performance from multimodal data]". The framework predicts perceived team 
performance from structural, dialogue-act, and sentiment features 
extracted from the AMI Meeting Corpus, using nested cross-validation 
across four model classes (Elastic Net, SVR, Random Forest, XGBoost).

## Requirements
Python 3.x. Install dependencies with:
    pip install -r requirements.txt

## Data
This project uses the AMI Meeting Corpus, which is publicly available 
at: http://corpus.amiproject.org/
The corpus is not included in this repository. Place the downloaded 
corpus in the data/ directory before running the scripts. See 
data/README_data.md for the expected folder structure.

## Reproducing the analysis
Run scripts in this order:
1. features/extract_structural.py     — extracts structural features
2. features/extract_dialogue_acts.py  — extracts DA distributions
3. features/extract_sentiment.py      — extracts VADER sentiment
4. models/nested_cv.py                — runs full model comparison
5. analysis/shap_analysis.py          — SHAP feature importance
6. analysis/thin_slice.py             — thin-slice replication
7. analysis/fairness_audit.py         — gender composition audit

## Citation
If you use this code, please cite:
[Bram Dijkstra] ([2026]). [Predicting team performance from multimodal data]. 
Master's Thesis, Tilburg University.
