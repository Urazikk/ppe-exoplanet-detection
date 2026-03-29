"""
=============================================================================
ENTRAINEMENT TSFRESH - Dataset Kaggle Kepler  v2
=============================================================================
Utilise exoTrain.csv et exoTest.csv du dataset Kaggle
"Kepler Labelled Time Series Data" - zéro téléchargement NASA.

Stratégie :
  - Train + Test Kaggle fusionnés → 5657 étoiles, 42 planètes
  - Sélection des top 100 features (évite overfitting sur 777)
  - scale_pos_weight = ratio négatifs/positifs (pénalise les planètes ratées)
  - Optimisation du seuil de décision (maximise F1 en CV)
  - 10-fold CV stratifiée avec SMOTE dans chaque fold
  - Split final 80/20 stratifié pour holdout test

Usage : python 03_kaggle_tsfresh_train.py
=============================================================================
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, precision_recall_curve)
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[WARN] imbalanced-learn non installé, SMOTE désactivé.")

try:
    from tsfresh import extract_features
    from tsfresh.feature_extraction import EfficientFCParameters
    from tsfresh.utilities.dataframe_functions import impute
    HAS_TSFRESH = True
except ImportError:
    HAS_TSFRESH = False
    print("[WARN] tsfresh non installé.")

# =============================================================================
# CHEMINS & CONFIG
# =============================================================================

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV  = os.path.join(BASE_DIR, "data", "kaggle", "exoTrain.csv")
TEST_CSV   = os.path.join(BASE_DIR, "data", "kaggle", "exoTest.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
CACHE_DIR  = os.path.join(BASE_DIR, "data", "cache")

DOWNSAMPLE_FACTOR = 1     # pas de sous-échantillonnage → 3197 pts complets
TOP_N_FEATURES    = 100   # garder les 100 features les plus importantes

# Chemins alternatifs
ALT_TRAIN = "/sessions/admiring-gifted-goodall/mnt/uploads/exoTrain.csv"
ALT_TEST  = "/sessions/admiring-gifted-goodall/mnt/uploads/exoTest.csv"


def find_csv(primary, fallback):
    if os.path.exists(primary):
        return primary
    if os.path.exists(fallback):
        print(f"[INFO] Fichier trouvé à : {fallback}")
        return fallback
    raise FileNotFoundError(f"Fichier introuvable : {primary}")


# =============================================================================
# 1. CHARGEMENT + FUSION
# =============================================================================

def load_data():
    train_path = find_csv(TRAIN_CSV, ALT_TRAIN)
    test_path  = find_csv(TEST_CSV,  ALT_TEST)

    print("[1/7] Chargement et fusion des données...")
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    df_all = pd.concat([df_train, df_test], ignore_index=True)
    y_all  = (df_all['LABEL'] == 2).astype(int)
    X_all  = df_all.drop(columns=['LABEL'])

    n_pos, n_neg = y_all.sum(), (y_all == 0).sum()
    print(f"     {len(y_all)} étoiles — {n_pos} planètes / {n_neg} non-planètes (1:{n_neg//n_pos})")
    return X_all, y_all


# =============================================================================
# 2. CONVERSION FORMAT TSFRESH
# =============================================================================

def to_tsfresh_format(X_raw, downsample=DOWNSAMPLE_FACTOR):
    if downsample and downsample > 1:
        X_raw = X_raw.iloc[:, ::downsample]
    n_stars, n_points = X_raw.shape
    ids  = np.repeat(np.arange(n_stars), n_points)
    t    = np.tile(np.arange(n_points), n_stars)
    flux = X_raw.values.flatten()
    return pd.DataFrame({"id": ids, "time": t, "flux": flux})


# =============================================================================
# 3. EXTRACTION TSFRESH (cache parquet)
# =============================================================================

def extract_tsfresh_features(X_raw, label, n_jobs=4):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"features_{label}_efficient_ds{DOWNSAMPLE_FACTOR}.parquet")

    if os.path.exists(cache_path):
        print(f"     [CACHE] {os.path.basename(cache_path)}...")
        features = pd.read_parquet(cache_path)
        print(f"     [CACHE] {features.shape[1]} features, {features.shape[0]} étoiles")
        return features

    if label == "full":
        ct = os.path.join(CACHE_DIR, f"features_train_efficient_ds{DOWNSAMPLE_FACTOR}.parquet")
        ce = os.path.join(CACHE_DIR, f"features_test_efficient_ds{DOWNSAMPLE_FACTOR}.parquet")
        if os.path.exists(ct) and os.path.exists(ce):
            print("     [CACHE] Recomposition train + test...")
            ft, fe = pd.read_parquet(ct), pd.read_parquet(ce)
            cols = ft.columns.intersection(fe.columns)
            features = pd.concat([ft[cols], fe[cols]], ignore_index=True)
            impute(features)
            features.to_parquet(cache_path, index=True)
            print(f"     [CACHE] → {features.shape[1]} features, {features.shape[0]} étoiles")
            return features

    ts_df = to_tsfresh_format(X_raw)
    print(f"     Extraction TSFRESH ({n_jobs} workers)...")
    t0 = time.time()
    features = extract_features(
        ts_df, column_id="id", column_sort="time", column_value="flux",
        default_fc_parameters=EfficientFCParameters(),
        n_jobs=n_jobs, disable_progressbar=False,
    )
    print(f"     Terminé en {time.time()-t0:.1f}s — {features.shape[1]} features")
    impute(features)
    features.to_parquet(cache_path, index=True)
    return features


# =============================================================================
# 4. SÉLECTION DES TOP FEATURES
# =============================================================================

def select_top_features(X, y, n_top=TOP_N_FEATURES):
    print(f"\n[3/7] Sélection des top {n_top} features (sur {X.shape[1]})...")

    ratio = (y == 0).sum() / max(y.sum(), 1)
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, scale_pos_weight=ratio,
        eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0,
    )
    clf.fit(X, y)

    importances = clf.feature_importances_
    ranking = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    selected = [name for name, _ in ranking[:n_top]]

    print(f"     Top 5 :")
    for name, imp in ranking[:5]:
        print(f"       {imp:.4f} | {name}")
    print(f"     → {n_top} features retenues")
    return selected


# =============================================================================
# 5. ÉVALUATION 10-FOLD CV + SEUIL OPTIMISÉ
# =============================================================================

def find_best_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[min(best_idx, len(thresholds) - 1)], f1_scores[best_idx]


def robust_cross_validate(X, y, use_smote=True):
    print("\n[4/7] Évaluation robuste — 10-fold CV stratifiée...")

    ratio = (y == 0).sum() / max(y.sum(), 1)
    print(f"     scale_pos_weight = {ratio:.0f}")
    print(f"     {y.sum()} planètes sur 10 folds (~{y.sum()//10}/fold)")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_metrics, fold_thresholds = [], []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if use_smote and HAS_SMOTE:
            smote = SMOTE(random_state=42)
            X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

        clf = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            gamma=0.1, scale_pos_weight=ratio, eval_metric='logloss',
            random_state=42, n_jobs=-1, verbosity=0,
        )
        clf.fit(X_tr, y_tr)
        y_proba = clf.predict_proba(X_val)[:, 1]

        best_thr, _ = find_best_threshold(y_val, y_proba)
        fold_thresholds.append(best_thr)
        y_pred = (y_proba >= best_thr).astype(int)

        fold_metrics.append({
            "accuracy":  accuracy_score(y_val, y_pred),
            "roc_auc":   roc_auc_score(y_val, y_proba),
            "f1":        f1_score(y_val, y_pred, zero_division=0),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall":    recall_score(y_val, y_pred, zero_division=0),
        })

    optimal_threshold = float(np.median(fold_thresholds))
    cv_metrics = {"optimal_threshold": optimal_threshold}

    print(f"\n     Seuil optimal (médiane) : {optimal_threshold:.4f}")
    print(f"\n     {'Métrique':<14} {'Moyenne':>8}  {'±Std':>8}  {'Min':>8}  {'Max':>8}")
    print(f"     {'-'*52}")

    for metric_name in ["accuracy", "roc_auc", "f1", "precision", "recall"]:
        vals = np.array([m[metric_name] for m in fold_metrics])
        display = metric_name.replace("roc_auc", "AUC-ROC").replace("_", " ").title()
        print(f"     {display:<14} {vals.mean():>8.4f}  {vals.std():>8.4f}  "
              f"{vals.min():>8.4f}  {vals.max():>8.4f}")
        cv_metrics[f"cv10_{metric_name}_mean"] = float(vals.mean())
        cv_metrics[f"cv10_{metric_name}_std"]  = float(vals.std())
        cv_metrics[f"cv10_{metric_name}_min"]  = float(vals.min())
        cv_metrics[f"cv10_{metric_name}_max"]  = float(vals.max())

    return cv_metrics


# =============================================================================
# 6. ENTRAÎNEMENT FINAL — split 80/20
# =============================================================================

def train_final(X, y, use_smote=True):
    print("\n[5/7] Entraînement final — split stratifié 80/20...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    ratio = (y_train == 0).sum() / max(y_train.sum(), 1)

    print(f"     Train : {len(y_train)} ({y_train.sum()} planètes)")
    print(f"     Test  : {len(y_test)} ({y_test.sum()} planètes)")

    clf = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
        gamma=0.1, scale_pos_weight=ratio, eval_metric='logloss',
        random_state=42, n_jobs=-1, verbosity=0,
    )

    if use_smote and HAS_SMOTE:
        smote = SMOTE(random_state=42)
        X_fit, y_fit = smote.fit_resample(X_train, y_train)
        print(f"     Après SMOTE : {len(y_fit)} samples")
    else:
        X_fit, y_fit = X_train, y_train

    clf.fit(X_fit, y_fit)
    return clf, X_train, X_test, y_train, y_test


# =============================================================================
# 7. ÉVALUATION HOLDOUT (seuil optimisé)
# =============================================================================

def evaluate(model, X_test, y_test, feature_names, cv_metrics):
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = cv_metrics.get("optimal_threshold", 0.5)

    y_pred_default = (y_proba >= 0.5).astype(int)
    y_pred_optimal = (y_proba >= threshold).astype(int)

    print(f"\n{'='*60}")
    print(f"  RÉSULTATS HOLDOUT ({len(y_test)} étoiles, {y_test.sum()} planètes)")
    print(f"{'='*60}")
    print(f"\n  {'Métrique':<14} {'Seuil=0.50':>12} {'Seuil={:.3f}'.format(threshold):>12}")
    print(f"  {'-'*40}")

    for name, scorer in [
        ("Accuracy",  accuracy_score),
        ("Precision", lambda yt, yp: precision_score(yt, yp, zero_division=0)),
        ("Recall",    lambda yt, yp: recall_score(yt, yp, zero_division=0)),
        ("F1",        lambda yt, yp: f1_score(yt, yp, zero_division=0)),
    ]:
        v_def = scorer(y_test, y_pred_default)
        v_opt = scorer(y_test, y_pred_optimal)
        arrow = " ⬆" if v_opt > v_def + 0.001 else (" ⬇" if v_opt < v_def - 0.001 else "  ")
        print(f"  {name:<14} {v_def:>12.4f} {v_opt:>12.4f}{arrow}")

    auc = roc_auc_score(y_test, y_proba)
    print(f"  {'AUC-ROC':<14} {auc:>12.4f} {auc:>12.4f}")

    y_pred = y_pred_optimal
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1v  = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n  Matrice de confusion (seuil={threshold:.3f}) :")
    print(f"  {cm}")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    # --- DIAGNOSTIC : probas des vraies planètes ---
    planet_mask = (y_test == 1)
    planet_probas = y_proba[planet_mask.values if hasattr(planet_mask, 'values') else planet_mask]
    print(f"  📊 DIAGNOSTIC — Probabilités des {len(planet_probas)} vraies planètes (holdout) :")
    for i, p in enumerate(sorted(planet_probas, reverse=True)):
        status = "✅ TROUVÉE" if p >= threshold else ("⚠️  ratée (> 0.5)" if p >= 0.5 else f"❌ ratée (proba trop basse)")
        print(f"    Planète {i+1} : proba = {p:.4f}  {status}")

    # --- SWEEP DE SEUILS ---
    print(f"\n  📈 SWEEP DE SEUILS — Impact sur Recall / Precision / F1 :")
    print(f"  {'Seuil':>8}  {'Recall':>8}  {'Precision':>10}  {'F1':>8}  {'FP':>4}  {'FN':>4}")
    print(f"  {'-'*48}")
    for thr in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        yp = (y_proba >= thr).astype(int)
        r  = recall_score(y_test, yp, zero_division=0)
        p  = precision_score(y_test, yp, zero_division=0)
        f  = f1_score(y_test, yp, zero_division=0)
        fp = int(((yp == 1) & (y_test == 0)).sum())
        fn = int(((yp == 0) & (y_test == 1)).sum())
        marker = " ◀" if abs(thr - threshold) < 0.02 else ""
        print(f"  {thr:>8.2f}  {r:>8.4f}  {p:>10.4f}  {f:>8.4f}  {fp:>4}  {fn:>4}{marker}")

    importances = model.feature_importances_
    top_features = sorted(
        [{"name": n, "importance": float(v)} for n, v in zip(feature_names, importances)],
        key=lambda x: x["importance"], reverse=True
    )
    print("\n  Top 10 features :")
    for f in top_features[:10]:
        print(f"    {f['importance']:.4f} | {f['name']}")

    return {
        "holdout_accuracy":   float(acc),
        "holdout_precision":  float(prec),
        "holdout_recall":     float(rec),
        "holdout_f1":         float(f1v),
        "holdout_auc_roc":    float(auc),
        "holdout_size":       int(len(y_test)),
        "holdout_n_planets":  int(y_test.sum()),
        "confusion_matrix":   cm,
        "optimal_threshold":  threshold,
        "n_features":         len(feature_names),
        "top_features":       top_features[:20],
        "source": "Kaggle Kepler (fusionné) + TSFRESH EfficientFCParameters (ds×4)",
        **{k: v for k, v in cv_metrics.items() if k != "optimal_threshold"},
    }


# =============================================================================
# SAUVEGARDE
# =============================================================================

def save_all(model, X_train, X_test, y_train, y_test, feature_names, metrics):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    df_tr = X_train.copy()
    df_tr['target_label'] = y_train.values if hasattr(y_train, 'values') else y_train
    df_tr.to_csv(os.path.join(OUTPUT_DIR, "training_dataset.csv"), index=False)

    df_te = X_test.copy()
    df_te['target_label'] = y_test.values if hasattr(y_test, 'values') else y_test
    df_te.to_csv(os.path.join(OUTPUT_DIR, "test_dataset.csv"), index=False)

    metadata = {
        "total_samples":     len(df_tr) + len(df_te),
        "train_samples":     len(df_tr),
        "test_samples":      len(df_te),
        "n_features":        len(feature_names),
        "optimal_threshold": metrics.get("optimal_threshold", 0.5),
        "source":    "Kaggle - Kepler Labelled Time Series Data (train+test fusionnés)",
        "method":    "TSFRESH Efficient + ds×4 + top100 features + scale_pos_weight + threshold opt + SMOTE-in-CV",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUTPUT_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    model.save_model(os.path.join(MODEL_DIR, "exoplanet_model.json"))

    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(MODEL_DIR, "selected_features.json"), "w") as f:
        json.dump(feature_names, f)

    print(f"\n[OK] Modèle     → {MODEL_DIR}/exoplanet_model.json")
    print(f"[OK] Métriques  → {MODEL_DIR}/model_metrics.json")
    print(f"[OK] Features   → {MODEL_DIR}/selected_features.json ({len(feature_names)})")
    print(f"[OK] Datasets   → {OUTPUT_DIR}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if not HAS_TSFRESH:
        print("[FATAL] tsfresh non installé. Lancez : pip install tsfresh")
        return

    print("=" * 60)
    print("  KAGGLE TSFRESH TRAINING - ExoPlanet AI  v2")
    print("  scale_pos_weight + feature selection + threshold opt")
    print("=" * 60)
    t0_global = time.time()

    # 1. Chargement + fusion
    X_all_raw, y_all = load_data()

    # 2. Extraction TSFRESH
    print(f"\n[2/7] Extraction TSFRESH ({len(y_all)} étoiles)...")
    X_all = extract_tsfresh_features(X_all_raw, "full", n_jobs=4)
    print(f"     {X_all.shape[1]} features brutes disponibles")

    X_all = X_all.reset_index(drop=True)
    y_all = y_all.reset_index(drop=True)

    # 3. Feature selection → top 100
    selected_features = select_top_features(X_all, y_all, n_top=TOP_N_FEATURES)
    X_all = X_all[selected_features]

    # 4. 10-fold CV robuste avec seuil optimisé
    cv_metrics = robust_cross_validate(X_all, y_all, use_smote=True)

    # 5. Entraînement final
    model, X_train, X_test, y_train, y_test = train_final(
        X_all, y_all, use_smote=True
    )

    # 6. Évaluation holdout
    print(f"\n[6/7] Évaluation holdout...")
    metrics = evaluate(model, X_test, y_test, selected_features, cv_metrics)

    # 7. Sauvegarde
    print(f"\n[7/7] Sauvegarde...")
    save_all(model, X_train, X_test, y_train, y_test, selected_features, metrics)

    elapsed = time.time() - t0_global
    print(f"\n{'='*60}")
    print(f"  Terminé en {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()