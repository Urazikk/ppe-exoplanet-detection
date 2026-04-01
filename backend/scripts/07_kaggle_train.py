"""
Script d'entraînement XGBoost MULTI-MISSIONS (Kepler KOI + TESS TOI).
Fusionne les deux datasets, harmonise les colonnes, et entraîne un
classifieur binaire robuste capable de prédire sur les deux missions.
Utilise astropy pour les coordonnées galactiques.
"""

import sys
import os
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Bibliothèque astropy pour les coordonnées galactiques
from astropy.coordinates import SkyCoord
import astropy.units as u

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, precision_score,
                             recall_score, confusion_matrix)
from xgboost import XGBClassifier

# ============================================================================
# CONFIGURATION
# ============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "catalog"
MODELS_DIR = ROOT_DIR / "models"

KEPLER_PATH = DATA_DIR / "exoplanet_binary_model_imputed.csv"
TESS_PATH   = DATA_DIR / "tess_toi_binary.csv"

MODEL_PATH    = MODELS_DIR / "exoplanet_model.json"
FEATURES_PATH = MODELS_DIR / "selected_features.json"
METRICS_PATH  = MODELS_DIR / "model_metrics.json"

RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def log(msg):
    logging.info(msg)


# ============================================================================
# FONCTIONS UTILITAIRES & ASTROPY
# ============================================================================

def add_astropy_features(df):
    """
    Convertit RA/Dec (ICRS) en coordonnées Galactiques (glon, glat)
    via astropy. Applicable aux deux missions (Kepler et TESS).
    """
    if "ra" in df.columns and "dec" in df.columns:
        log("Transformation RA/DEC -> Coordonnées Galactiques avec Astropy...")
        try:
            coords = SkyCoord(
                ra=df["ra"].values * u.degree,
                dec=df["dec"].values * u.degree,
                frame="icrs"
            )
            df["glon"] = coords.galactic.l.degree
            df["glat"] = coords.galactic.b.degree
            log("Features 'glon' et 'glat' générées avec succès.")
            df = df.drop(columns=["ra", "dec"], errors="ignore")
        except Exception as e:
            log(f"Erreur avec astropy : {e}")
    else:
        log("Aucune colonne RA/DEC trouvée. Transformation ignorée.")
    return df


def load_and_merge_datasets():
    """
    Charge les datasets Kepler et TESS, harmonise les colonnes,
    ajoute une feature 'is_tess' et les fusionne.
    """
    # --- Kepler ---
    if not KEPLER_PATH.exists():
        raise FileNotFoundError(f"Dataset Kepler introuvable : {KEPLER_PATH}")
    
    log(f"Chargement Kepler : {KEPLER_PATH.name}")
    df_kepler = pd.read_csv(KEPLER_PATH)
    df_kepler["is_tess"] = 0
    df_kepler["mission"] = "Kepler"
    log(f"  → {len(df_kepler)} entrées Kepler")
    
    # --- TESS ---
    if not TESS_PATH.exists():
        log(f"[!] Dataset TESS introuvable : {TESS_PATH}")
        log("    Lancez d'abord : python backend/scripts/download_tess_toi.py")
        log("    Entraînement avec Kepler seul...")
        return df_kepler
    
    log(f"Chargement TESS : {TESS_PATH.name}")
    df_tess = pd.read_csv(TESS_PATH)
    df_tess["is_tess"] = 1
    log(f"  → {len(df_tess)} entrées TESS")
    
    # --- Harmonisation des colonnes ---
    # Trouver les colonnes communes (numériques + target + is_tess)
    kepler_cols = set(df_kepler.columns)
    tess_cols = set(df_tess.columns)
    common_cols = sorted(kepler_cols & tess_cols)
    
    # S'assurer que les colonnes essentielles sont présentes
    essential = ["target_planet", "is_tess"]
    for col in essential:
        if col not in common_cols:
            common_cols.append(col)
    
    log(f"  → {len(common_cols)} colonnes communes entre Kepler et TESS")
    
    # Ne garder que les colonnes communes
    df_kepler_slim = df_kepler[[c for c in common_cols if c in df_kepler.columns]].copy()
    df_tess_slim = df_tess[[c for c in common_cols if c in df_tess.columns]].copy()
    
    # Concaténer
    df_merged = pd.concat([df_kepler_slim, df_tess_slim], ignore_index=True)
    log(f"Dataset fusionné : {len(df_merged)} entrées totales")
    log(f"  → Kepler : {(df_merged['is_tess'] == 0).sum()}")
    log(f"  → TESS   : {(df_merged['is_tess'] == 1).sum()}")
    
    return df_merged


def prepare_dataset(df):
    """
    Filtre les données, élimine les fuites d'informations (leakage)
    et sépare X (features) de y (cibles).
    """
    if "target_planet" not in df.columns:
        raise ValueError("La colonne cible 'target_planet' est manquante.")
        
    y = df["target_planet"].astype(int)

    # Colonnes à exclure (identifiants, labels, métadonnées textuelles)
    drop_cols = [
        "target_planet",
        "kepid", "kepoi_name", "kepler_name", "koi_disposition",
        "index", "target_label", "mission", "tid", "toi"
    ]
    
    # Ne garder que les colonnes numériques
    X = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in drop_cols if c in df.columns], 
        errors="ignore"
    )

    # Nettoyage
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    log(f"Dataset prêt : {len(X)} astres | {X.shape[1]} features")
    log(f"Répartition : {y.sum()} CONFIRMED (1) | {(y == 0).sum()} FALSE POSITIVE (0)")
    
    return X, y


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_training():
    log("=" * 70)
    log("  ENTRAÎNEMENT XGBOOST MULTI-MISSIONS (Kepler + TESS)")
    log("=" * 70)
    
    # 1. Charger et fusionner les datasets
    df = load_and_merge_datasets()
    
    # 2. Feature Engineering avec Astropy
    df = add_astropy_features(df)
    
    # 3. Préparation
    X, y = prepare_dataset(df)
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    
    features_list = list(X.columns)
    log(f"Features utilisées ({len(features_list)}) : {features_list[:10]}...")
    log(f"Test Set = {len(X_test)} échantillons")

    # 4. Modèle XGBoost (plus d'arbres pour le dataset multi-missions)
    pos_weight = (y_train == 0).sum() / max(1, y_train.sum())
    
    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=pos_weight, 
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    # 5. Validation croisée
    log("Validation croisée stratifiée (K=5)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_results = cross_validate(
        model, X_train, y_train, 
        cv=cv, scoring=['accuracy', 'f1', 'roc_auc'],
        n_jobs=-1
    )
    
    cv_acc_mean = np.mean(cv_results['test_accuracy'])
    cv_acc_std  = np.std(cv_results['test_accuracy'])
    cv_f1_mean  = np.mean(cv_results['test_f1'])
    cv_f1_std   = np.std(cv_results['test_f1'])
    cv_auc_mean = np.mean(cv_results['test_roc_auc'])
    log(f"CV -> Accuracy {cv_acc_mean:.3f} ± {cv_acc_std:.3f}")
    log(f"CV -> F1       {cv_f1_mean:.3f} ± {cv_f1_std:.3f}")
    log(f"CV -> AUC      {cv_auc_mean:.3f}")

    # 6. Entraînement final
    log("Entraînement final sur le train set complet...")
    model.fit(X_train, y_train)

    # 7. Évaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    
    log("=" * 50)
    log(f"  TEST SET RESULTS")
    log(f"  Accuracy  : {acc:.4f}")
    log(f"  Precision : {prec:.4f}")
    log(f"  Recall    : {rec:.4f}")
    log(f"  F1-Score  : {f1:.4f}")
    log(f"  ROC-AUC   : {auc:.4f}")
    log("=" * 50)
    log("\n" + classification_report(y_test, y_pred))

    # Feature Importances
    imp = model.feature_importances_
    imp_df = pd.DataFrame({"name": features_list, "importance": imp})\
             .sort_values(by="importance", ascending=False)
    log(f"Top 10 Features :\n{imp_df.head(10).to_string(index=False)}")

    # 8. Sauvegarde
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
    model.save_model(str(MODEL_PATH))
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(features_list, f, indent=4)
        
    metrics = {
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "test_auc_roc": float(auc),
        "cv_accuracy_mean": float(cv_acc_mean),
        "cv_accuracy_std": float(cv_acc_std),
        "cv_f1_mean": float(cv_f1_mean),
        "cv_f1_std": float(cv_f1_std),
        "n_features_selected": len(features_list),
        "n_features_total": len(features_list),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "confusion_matrix": cm,
        "top_features": imp_df.head(20).to_dict('records'),
        "datasets": {
            "kepler": int((X_train["is_tess"] == 0).sum() + (X_test["is_tess"] == 0).sum()) if "is_tess" in X.columns else len(X),
            "tess": int((X_train["is_tess"] == 1).sum() + (X_test["is_tess"] == 1).sum()) if "is_tess" in X.columns else 0,
        }
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    log(f"Modèle sauvegardé : {MODEL_PATH}")
    log(f"Features sauvegardées : {FEATURES_PATH}")
    log(f"Métriques sauvegardées : {METRICS_PATH}")
    log("=== Entraînement Multi-Missions terminé avec succès ===")


if __name__ == "__main__":
    run_training()

