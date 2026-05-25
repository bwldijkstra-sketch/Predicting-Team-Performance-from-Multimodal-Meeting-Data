from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

feature_cols = [c for c in struct.columns if c != 'meeting_id']
X = df[feature_cols].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': make_scorer(accuracy_score),
    'f1':       make_scorer(f1_score),
    'roc_auc':  make_scorer(roc_auc_score, needs_proba=True)
}

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, random_state=42,
        eval_metric='logloss', verbosity=0
    )
}

results = {}

for target in targets:
    y_bin = df[f'{target}_binary'].values
    results[target] = {}
    
    for model_name, model in models.items():
        scores = cross_validate(model, X, y_bin, cv=cv,
                                scoring=scoring, n_jobs=-1)
        results[target][model_name] = {
            'Accuracy': scores['test_accuracy'].mean(),
            'Accuracy SD': scores['test_accuracy'].std(),
            'F1':       scores['test_f1'].mean(),
            'F1 SD':    scores['test_f1'].std(),
            'AUC':      scores['test_roc_auc'].mean(),
            'AUC SD':   scores['test_roc_auc'].std(),
        }
    print(f"Done: {target}")

for target in targets:
    print(f"\n{'='*60}")
    print(f"Target: {target}")
    print(f"{'='*60}")
    rows = []
    for model_name, metrics in results[target].items():
        rows.append({
            'Model': model_name,
            'Accuracy': f"{metrics['Accuracy']:.3f} ± {metrics['Accuracy SD']:.3f}",
            'F1':       f"{metrics['F1']:.3f} ± {metrics['F1 SD']:.3f}",
            'AUC':      f"{metrics['AUC']:.3f} ± {metrics['AUC SD']:.3f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))
