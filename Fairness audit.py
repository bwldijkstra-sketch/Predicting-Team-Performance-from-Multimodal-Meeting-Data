"""
rq5_pipeline_v2.py
------------------
RQ5 Fairness Audit — complete pipeline using participant IDs from meetings.xml.
Extracts demographics implicitly encoded within the AMI participant ID structural metadata string:
  [MF][IET][EDO][0-9][0-9][0-9]
  - First letter: M = Male, F = Female
  - Third letter: E = English native, D = Dutch native, O = Other

Required inputs (defined via config.py):
  - config.AMI_ROOT   (AMI corpus path containing corpusResources/meetings.xml)
  - config.OUT_DIR    (Directory containing full_features.csv)
  - config.Y_CSV      (Target performance variables csv)

Outputs (saved to RESULTS_DIR/fairness_v2/):
  - ami_demographics_full_v2.csv — Extracted raw base demographics dataframe
  - rq5_fairness_boxplots.png     — Prediction error diagnostic plots
  - rq5_fairness_stats.csv        — Downstream bias indicators (Cohen's d, MW-U)
  - rq5_oof_residuals.csv         — Extracted out-of-fold residuals
"""

import os
import re
import glob
import warnings
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

# Import repository configurations
import config

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration & Path Mapping
# =============================================================================

AMI_CORPUS_PATH = config.AMI_ROOT
FEATURES_CSV    = os.path.join(config.OUT_DIR, "full_features.csv")
TARGETS_CSV     = config.Y_CSV

# Routing outputs clean to a dedicated subdirectory within the results folder
OUTPUT_DIR      = os.path.join(config.OUT_DIR, "results", "fairness_v2")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Helper Functions (Statistical Analysis)
# =============================================================================

def cohen_d(a, b):
    """Calculate Cohen's d effect size metric for variance comparison."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def nested_cv_residuals(X, y, random_state=42):
    """
    Executes a 10-fold outer / 5-fold inner nested cross-validation loop.
    Returns out-of-fold (OOF) prediction residuals.
    """
    y_binned = pd.qcut(y, q=3, labels=False, duplicates='drop')
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    oof_preds = np.full(len(y), np.nan)

    alphas    = [0.001, 0.01, 0.1, 1.0, 10.0]
    l1_ratios = [0.1, 0.5, 0.7, 0.9, 1.0]

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y_binned)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_te_s   = scaler.transform(X_te)

        y_tr_bin = pd.qcut(y_tr, q=3, labels=False, duplicates='drop')
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        best_score, best_alpha, best_l1 = -np.inf, 0.01, 0.5
        for alpha in alphas:
            for l1r in l1_ratios:
                fold_r2 = []
                for i_tr, i_val in inner_cv.split(X_tr_s, y_tr_bin):
                    m = ElasticNet(alpha=alpha, l1_ratio=l1r, max_iter=5000, random_state=random_state)
                    m.fit(X_tr_s[i_tr], y_tr[i_tr])
                    p      = m.predict(X_tr_s[i_val])
                    ss_res = np.sum((y_tr[i_val] - p) ** 2)
                    ss_tot = np.sum((y_tr[i_val] - np.mean(y_tr[i_val])) ** 2)
                    fold_r2.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
                score = np.mean(fold_r2)
                if score > best_score:
                    best_score, best_alpha, best_l1 = score, alpha, l1r

        model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1, max_iter=5000, random_state=random_state)
        model.fit(X_tr_s, y_tr)
        oof_preds[te_idx] = model.predict(X_te_s)
        print(f"    Fold {fold+1:2d}: alpha={best_alpha}, l1={best_l1}, inner R²={best_score:.3f}")

    return y - oof_preds


# =============================================================================
# Main Pipeline Run Execution
# =============================================================================

def main():
    # ── STEP 1: Extract demographics from participant IDs in meetings.xml ─────────
    print("=" * 60)
    print("STEP 1: Extracting demographics from participant IDs")
    print("=" * 60)

    meetings_xml = os.path.join(AMI_CORPUS_PATH, 'corpusResources', 'meetings.xml')
    if not os.path.exists(meetings_xml):
        candidates = glob.glob(os.path.join(AMI_CORPUS_PATH, '**', 'meetings.xml'), recursive=True)
        if candidates:
            meetings_xml = candidates[0]
        else:
            raise FileNotFoundError(f"Could not find meetings.xml under {AMI_CORPUS_PATH}")

    print(f"  Using: {meetings_xml}")
    tree = ET.parse(meetings_xml)
    root = tree.getroot()

    # Regex capturing pattern: [MF][IET][EDO]\d{3}
    PARTICIPANT_ID_RE = re.compile(r'\b([MF][IET][EDO]\d{3})\b', re.IGNORECASE)

    rows = []
    for meeting_elem in root.iter('meeting'):
        obs = meeting_elem.get('observation', meeting_elem.get('name', ''))
        if not obs or not re.match(r'^(ES|IS|TS)', obs):
            continue

        group = re.sub(r'[a-d]$', '', obs.strip())

        for speaker in meeting_elem.iter('speaker'):
            global_name = speaker.get('global_name', '')
            m = PARTICIPANT_ID_RE.match(global_name)
            if m:
                pid = m.group(1).upper()
                sex_letter  = pid[0]   # M or F
                lang_letter = pid[2]   # E, D, or O
                rows.append({
                    'meeting_group':      group,
                    'participant_id':     global_name,
                    'sex':                sex_letter,
                    'native_lang_code':   lang_letter,
                    'is_female':          1 if sex_letter == 'F' else 0,
                    'is_native_english':  1 if lang_letter == 'E' else 0,
                })

    if not rows:
        raise ValueError("No matching valid participant ID codes parsed from meetings.xml.")

    ind_df = pd.DataFrame(rows)
    print(f"\n  Parsed {len(ind_df)} participant records")
    print(f"  Sample records:")
    print(ind_df.head(8).to_string(index=False))

    # ── STEP 2: Aggregate to meeting-group level ──────────────────────────────────
    demo_df = (ind_df
        .groupby('meeting_group')
        .agg(
            n_participants       = ('sex',              'count'),
            prop_female          = ('is_female',        'mean'),
            prop_native_english = ('is_native_english','mean'),
        )
        .reset_index()
    )
    demo_df['gender_balanced'] = (demo_df['prop_female']         >= 0.5).astype(int)
    demo_df['native_dominant'] = (demo_df['prop_native_english'] >  0.5).astype(int)

    demo_path = os.path.join(OUTPUT_DIR, 'ami_demographics_full_v2.csv')
    demo_df.to_csv(demo_path, index=False)
    print(f"\n  Saved demographics ({len(demo_df)} groups) to: {demo_path}")

    # ── STEP 3: Load features and merge ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading features and merging demographics")
    print("=" * 60)

    feat_df = pd.read_csv(FEATURES_CSV)
    feat_df['meeting_group'] = feat_df['meeting_id'].str.extract(r'^([A-Z]+\d+)')

    TARGETS = ['overall_performance', 'satisfaction', 'cohesiveness', 'leadership', 'information_processing']
    FEATURE_COLS = [c for c in feat_df.columns if c not in ['meeting_id', 'meeting_group'] + TARGETS]

    merged = feat_df.merge(
        demo_df[['meeting_group', 'gender_balanced', 'prop_female', 'prop_native_english', 'native_dominant']],
        on='meeting_group', how='left'
    )

    n_missing = merged['gender_balanced'].isna().sum()
    print(f"  Total meetings:            {len(merged)}")
    print(f"  Matched with demographics: {merged['gender_balanced'].notna().sum()}")
    print(f"  Unmatched (dropped):       {n_missing}")

    if n_missing > 0:
        print(f"  Unmatched groups: {sorted(merged[merged['gender_balanced'].isna()]['meeting_group'].unique())}")

    merged = merged.dropna(subset=['gender_balanced']).reset_index(drop=True)
    print(f"  Proceeding with {len(merged)} meetings")
    print(f"  Gender-balanced   (>=50% F): {int(merged['gender_balanced'].sum())} meetings")
    print(f"  Gender-imbalanced (<50%  F): {int((merged['gender_balanced'] == 0).sum())} meetings")

    X            = merged[FEATURE_COLS].values
    gender_label = merged['gender_balanced'].values

    # ── STEP 4: Nested CV — out-of-fold residuals ─────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Nested cross-validation (ElasticNet, multimodal)")
    print("=" * 60)

    oof_results = {}
    for target in TARGETS:
        print(f"\n  --- {target} ---")
        y = merged[target].values
        oof_results[target] = nested_cv_residuals(X, y)
        print(f"  Mean residual: {np.mean(oof_results[target]):.4f}, SD: {np.std(oof_results[target]):.4f}")

    # ── STEP 5: Fairness statistics ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Fairness statistics")
    print("=" * 60)

    stat_rows = []
    for target in TARGETS:
        res   = oof_results[target]
        bal   = res[gender_label == 1]
        imbal = res[gender_label == 0]
        d     = cohen_d(bal, imbal)
        u, p  = stats.mannwhitneyu(bal, imbal, alternative='two-sided')
        
        print(f"\n  {target}:")
        print(f"    Balanced   (n={len(bal):3d}): mean={np.mean(bal):+.4f}, SD={np.std(bal):.4f}")
        print(f"    Imbalanced (n={len(imbal):3d}): mean={np.mean(imbal):+.4f}, SD={np.std(imbal):.4f}")
        print(f"    Cohen's d = {d:.3f}   Mann-Whitney p = {p:.3f}")
        
        stat_rows.append({
            'outcome':             target,
            'n_balanced':          len(bal),
            'n_imbalanced':        len(imbal),
            'mean_res_balanced':   round(np.mean(bal),   4),
            'mean_res_imbalanced': round(np.mean(imbal), 4),
            'sd_res_balanced':     round(np.std(bal),    4),
            'sd_res_imbalanced':   round(np.std(imbal),   4),
            'cohens_d':            round(d, 3),
            'mw_p':                round(p, 3),
        })

    stats_df = pd.DataFrame(stat_rows)
    stats_path = os.path.join(OUTPUT_DIR, 'rq5_fairness_stats.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"\n  Saved statistics to: {stats_path}")

    # ── STEP 6: Figure Generation ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5: Generating figure")
    print("=" * 60)

    TARGET_LABELS = {
        'overall_performance':    'Overall\nPerformance',
        'satisfaction':           'Satisfaction',
        'cohesiveness':           'Cohesiveness',
        'leadership':             'Leadership',
        'information_processing': 'Information\nProcessing',
    }
    COL_BAL   = '#4878CF'
    COL_IMBAL = '#D65F5F'

    fig, axes = plt.subplots(1, 5, figsize=(14, 5))
    fig.patch.set_facecolor('white')

    for ax, target in zip(axes, TARGETS):
        res   = oof_results[target]
        bal   = res[gender_label == 1]
        imbal = res[gender_label == 0]
        row   = stats_df[stats_df['outcome'] == target].iloc[0]
        d, p  = row['cohens_d'], row['mw_p']

        bp = ax.boxplot(
            [bal, imbal],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker='o', markersize=3.5, alpha=0.5, markerfacecolor='grey', markeredgecolor='grey'),
        )
        bp['boxes'][0].set_facecolor(COL_BAL);   bp['boxes'][0].set_alpha(0.75)
        bp['boxes'][1].set_facecolor(COL_IMBAL); bp['boxes'][1].set_alpha(0.75)

        ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.set_title(TARGET_LABELS[target], fontsize=9.5, fontweight='bold', pad=7)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'Balanced\n(n={len(bal)})', f'Imbalanced\n(n={len(imbal)})'], fontsize=8)
        ax.set_ylabel('Residual ($y - \\hat{y}$)', fontsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        sig_label = '*' if p < 0.05 else 'n.s.'
        ax.text(0.5, 0.97, f'$d$ = {d:.2f}, {sig_label}', transform=ax.transAxes, ha='center', va='top', fontsize=8, style='italic')

    patch1 = mpatches.Patch(color=COL_BAL,   alpha=0.75, label='Gender-balanced (≥50% female)')
    patch2 = mpatches.Patch(color=COL_IMBAL, alpha=0.75, label='Gender-imbalanced (<50% female)')
    fig.legend(handles=[patch1, patch2], loc='lower center', ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    
    fig.suptitle(
        'Prediction Residuals by Gender Composition of Meeting Group\n'
        '(Elastic Net Regression, Multimodal Features, 10-fold Nested CV)',
        fontsize=10.5, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'rq5_fairness_boxplots.png')
    plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved figure to: {fig_path}")

    # ── STEP 7: Save residuals ────────────────────────────────────────────────────
    residuals_df = merged[['meeting_id', 'meeting_group', 'gender_balanced', 'prop_female', 'prop_native_english'] + TARGETS].copy()
    for target in TARGETS:
        residuals_df[f'residual_{target}'] = oof_results[target]
        
    res_path = os.path.join(OUTPUT_DIR, 'rq5_oof_residuals.csv')
    residuals_df.to_csv(res_path, index=False)
    print(f"  Saved residuals to: {res_path}")

    print("\n" + "=" * 60)
    print("DONE. Pipeline files are generated and structured:")
    print(f"  1. {fig_path}")
    print(f"  2. {stats_path}")
    print(f"  3. {res_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()