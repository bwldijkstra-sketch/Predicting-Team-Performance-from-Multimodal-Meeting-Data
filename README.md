# Predicting Team Performance from Multimodal Meeting Data

**Master's Thesis — Data Science & Society, Tilburg University**  
Author: Bram Dijkstra

---

## Overview

This repository contains all analysis code for the thesis *"[Your Thesis Title]"*.  
The framework predicts perceived team performance from behavioral interaction data extracted from the AMI Meeting Corpus, using a nested cross-validation procedure across four model classes and four feature modalities.

**Five composite performance outcomes are predicted:**
- Overall performance
- Satisfaction
- Cohesiveness
- Leadership
- Information processing

**Three feature modalities are compared:**
- Structural (participation dynamics: speaking time, turn-taking, silence ratio, Gini coefficient)
- Semantic — Dialogue-act distributions (15 DA types from AMI annotations)
- Semantic — VADER sentiment (positive, negative, neutral, compound polarity)

**Four models are compared:**
- Elastic Net Regression (ENR)
- Support Vector Regression (SVR)
- Random Forest Regression (RFR)
- XGBoost Regression (XGB)

---

## Repository Structure

```
├── config.py                     ← Set your AMI corpus path here (edit once)
├── requirements.txt
│
├── features/
│   ├── extract_y.py              ← Extract decision counts from AMI XML
│   └── extract_semantic.py       ← Extract DA distributions and VADER sentiment
│
├── models/
│   └── nested_cv.py              ← Full nested CV pipeline (RQ1, RQ2)
│
├── analysis/
│   ├── eda.py                    ← Exploratory data analysis and target distributions
│   ├── shap_analysis.py          ← SHAP feature importance and fold stability (RQ4)
│   └── fairness_audit.py         ← Gender composition fairness audit (RQ5)
│
├── data/
│   └── README_data.md            ← How to obtain and set up the AMI corpus
│
└── outputs/                      ← Created automatically; all results saved here
```

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Download the AMI corpus**

The AMI Meeting Corpus is publicly available at:  
http://corpus.amiproject.org/

See `data/README_data.md` for the expected folder structure.

**3. Set your corpus path**

Open `config.py` and set `AMI_ROOT` to the path of your downloaded corpus:

```python
AMI_ROOT = "/path/to/your/ami_corpus"
```

---

## Running the Analysis

Run scripts in this order:

```bash
# 1. Extract target variables (decision counts)
python features/extract_y.py

# 2. Extract semantic features (DA distributions + sentiment)
python features/extract_semantic.py

# 3. Exploratory data analysis
python analysis/eda.py

# 4. Nested cross-validation model comparison (RQ1, RQ2)
python models/nested_cv.py

# 5. SHAP feature importance and stability (RQ4)
python analysis/shap_analysis.py

# 6. Fairness audit (RQ5)
python analysis/fairness_audit.py
```

All outputs (CSV files and figures) are saved to the `outputs/` directory.

> **Note:** Structural feature extraction code and thin-slice analysis code  
> are available on request. The structural features CSV  
> (`ami_structural_features.csv`) is a direct input to `models/nested_cv.py`.

---

## Data Availability

The AMI Meeting Corpus is not included in this repository. It is distributed  
for academic research purposes and can be obtained free of charge from  
http://corpus.amiproject.org/ or directly from the original authors.

---

## Libraries Used

| Library | Purpose | Citation |
|---|---|---|
| scikit-learn | ENR, SVR, RFR, cross-validation | Pedregosa et al. (2011) |
| xgboost | XGBoost model | Chen & Guestrin (2016) |
| shap | Post-hoc SHAP explanation | Lundberg & Lee (2017) |
| nltk | Text processing | Bird et al. (2009) |
| vaderSentiment | Utterance-level sentiment | Hutto & Gilbert (2014) |
| pandas, numpy, scipy, matplotlib | Data handling and statistics | — |

---

## Citation

If you use this code, please cite:

> Dijkstra, B. (2025). *[Thesis Title]*. Master's Thesis, Tilburg University.
