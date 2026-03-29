"""
=============================================================================
DATASET + ENTRAINEMENT - Basé catalogue NASA (sans téléchargement)
=============================================================================
Utilise directement les features physiques du catalogue Kepler KOI.
7326 étoiles disponibles, traitement en quelques secondes.

Usage : python 02_catalog_train.py
=============================================================================
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[WARN] imbalanced-learn non installé, SMOTE désactivé.")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "data", "catalog", "kepler_koi_catalog.csv"
)
CATALOG_PATH = os.path.normpath(CATALOG_PATH)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "data", "processed"
)
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "models"
)


# =============================================================================
# 1. CHARGEMENT ET FEATURE ENGINEERING
# =============================================================================

def load_and_engineer(catalog_path):
    """Charge le catalogue et construit les features physiques."""
    df = pd.read_csv(catalog_path)
    print(f"[OK] Catalogue chargé : {len(df)} entrées")

    # Label binaire
    df = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
    df['target_label'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)

    print(f"     CONFIRMED : {df['target_label'].sum()} | FALSE POSITIVE : {(df['target_label']==0).sum()}")

    # --- Features brutes du catalogue ---
    feature_cols = [
        'koi_period',       # Période orbitale (jours)
        'koi_depth',        # Profondeur du transit (ppm)
        'koi_duration',     # Durée du transit (heures)
        'koi_prad',         # Rayon de la planète (rayons terrestres)
        'koi_steff',        # Température effective de l'étoile (K)
        'koi_srad',         # Rayon de l'étoile (rayons solaires)
        'koi_kepmag',       # Magnitude Kepler
    ]

    # --- Features dérivées (ingénierie) ---
    eps = 1e-9

    # Rapport taille planète / étoile
    df['ratio_prad_srad'] = df['koi_prad'] / (df['koi_srad'] * 109.076 + eps)

    # Profondeur relative au rayon stellaire
    df['depth_per_srad'] = df['koi_depth'] / (df['koi_srad'] + eps)

    # Durée / Période (fraction du temps passé en transit)
    df['duty_cycle'] = df['koi_duration'] / (df['koi_period'] * 24 + eps)

    # Vitesse orbitale proxy (loi de Kepler simplifiée)
    df['orbital_speed_proxy'] = df['koi_srad'] / (df['koi_period'] ** (1/3) + eps)

    # Densité stellaire proxy
    df['stellar_density_proxy'] = df['koi_steff'] / (df['koi_srad'] ** 3 + eps)

    # SNR proxy (signal/bruit estimé)
    df['snr_proxy'] = df['koi_depth'] / (10 ** (df['koi_kepmag'] / 2.5 + eps))

    # Log des features à forte dynamique
    for col in ['koi_period', 'koi_depth', 'koi_prad', 'snr_proxy']:
        df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))

    derived_cols = [
        'ratio_prad_srad', 'depth_per_srad', 'duty_cycle',
        'orbital_speed_proxy', 'stellar_density_proxy', 'snr_proxy',
        'log_koi_period', 'log_koi_depth', 'log_koi_prad', 'log_snr_proxy'
    ]

    all_features = feature_cols + derived_cols

    X = df[all_features].copy()
    y = df['target_label'].copy()

    # Remplacement des NaN par la médiane
    X = X.fillna(X.median())

    # Suppression des infinis
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

    print(f"[OK] Features construites : {len(all_features)} features sur {len(X)} échantillons")
    return X, y, all_features


# =============================================================================
# 2. SMOTE + SPLIT
# =============================================================================

def prepare_data(X, y, test_ratio=0.2, use_smote=True):
    """Split stratifié + SMOTE sur le train."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )

    print(f"\n[Split] Train : {len(y_train)} | Test : {len(y_test)}")
    print(f"        Train -> {y_train.sum()} CONFIRMED / {(y_train==0).sum()} FP")

    if use_smote and HAS_SMOTE:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"[SMOTE] Après rééchantillonnage : {len(y_train)} samples ({y_train.sum()} / {(y_train==0).sum()})")
    else:
        print("[INFO] SMOTE non appliqué.")

    return X_train, X_test, y_train, y_test


# =============================================================================
# 3. ENTRAINEMENT XGBOOST
# =============================================================================

def train_xgboost(X_train, y_train):
    """Entraîne un XGBoost avec validation croisée."""
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print("\n[XGBoost] Validation croisée 5-fold...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    cv_f1  = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')

    print(f"  CV Accuracy : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"  CV AUC-ROC  : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"  CV F1       : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    print("\n[XGBoost] Entraînement final...")
    model.fit(X_train, y_train)

    return model, {
        "cv_accuracy_mean": float(cv_acc.mean()),
        "cv_accuracy_std": float(cv_acc.std()),
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "cv_f1_mean": float(cv_f1.mean()),
        "cv_f1_std": float(cv_f1.std()),
    }


# =============================================================================
# 4. EVALUATION
# =============================================================================

def evaluate(model, X_test, y_test, feature_names, cv_metrics):
    """Évalue sur le test set et sauvegarde les métriques."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n{'='*50}")
    print(f"  RESULTATS SUR TEST SET")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}  (objectif >= 0.90)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n  Matrice de confusion :")
    print(f"  {cm}")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

    # Top features
    importances = model.feature_importances_
    top_features = sorted(
        [{"name": n, "importance": float(v)} for n, v in zip(feature_names, importances)],
        key=lambda x: x["importance"], reverse=True
    )

    print("  Top 10 features :")
    for f in top_features[:10]:
        print(f"    {f['importance']:.4f} | {f['name']}")

    metrics = {
        "test_accuracy": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "test_auc_roc": float(auc),
        "confusion_matrix": cm,
        "n_features": len(feature_names),
        "top_features": top_features,
        **cv_metrics,
    }

    return metrics


# =============================================================================
# 5. SAUVEGARDE
# =============================================================================

def save_all(model, X_train, X_test, y_train, y_test, feature_names, metrics):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Datasets
    df_train = pd.DataFrame(X_train, columns=feature_names)
    df_train['target_label'] = y_train.values if hasattr(y_train, 'values') else y_train
    df_train.to_csv(os.path.join(OUTPUT_DIR, "training_dataset.csv"), index=False)

    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test['target_label'] = y_test.values if hasattr(y_test, 'values') else y_test
    df_test.to_csv(os.path.join(OUTPUT_DIR, "test_dataset.csv"), index=False)

    # Métadonnées
    metadata = {
        "total_samples": len(df_train) + len(df_test),
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "n_features": len(feature_names),
        "source": "NASA Kepler KOI Catalog - features physiques directes",
        "method": "Catalog features + feature engineering + SMOTE",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(OUTPUT_DIR, "dataset_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Modèle et métriques
    model.save_model(os.path.join(MODEL_DIR, "exoplanet_model.json"))

    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(MODEL_DIR, "selected_features.json"), "w") as f:
        json.dump(feature_names, f)

    print(f"\n[OK] Modèle sauvé dans {MODEL_DIR}/")
    print(f"[OK] Datasets sauvés dans {OUTPUT_DIR}/")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("  CATALOG-BASED TRAINING - ExoPlanet AI")
    print("=" * 60)
    t0 = time.time()

    # 1. Chargement + feature engineering
    print("\n[1/4] Chargement du catalogue et feature engineering...")
    X, y, feature_names = load_and_engineer(CATALOG_PATH)

    # 2. Split + SMOTE
    print("\n[2/4] Split train/test + SMOTE...")
    X_train, X_test, y_train, y_test = prepare_data(X, y, test_ratio=0.2, use_smote=True)

    # 3. Entraînement
    print("\n[3/4] Entraînement XGBoost...")
    model, cv_metrics = train_xgboost(X_train, y_train)

    # 4. Evaluation + sauvegarde
    print("\n[4/4] Evaluation et sauvegarde...")
    metrics = evaluate(model, X_test, y_test, feature_names, cv_metrics)
    save_all(model, X_train, X_test, y_train, y_test, feature_names, metrics)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Terminé en {elapsed:.1f} secondes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()