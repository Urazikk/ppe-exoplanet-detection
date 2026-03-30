"""
=============================================================================
Script d'entraînement XGBoost "Hybride" (Option C)
Signal Processing (TSFRESH) + Physique Stellaire (NASA KOI)
=============================================================================
Ce modèle représente l'état de l'art du projet. Il fusionne :
1. Les 27 features de morphologie de signal (Extraites de la courbe de lumière)
2. Les caractéristiques physiques de l'étoile et de l'astrométrie (Astropy)

Résultat attendu : Accuracy > 85% tout en respectant l'échelle du backend.
"""

import os
import json
import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ============================================================================
# CONFIGURATION DES CHEMINS
# ============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

SIGNAL_PATH = ROOT_DIR / "data" / "processed" / "overnight_features.csv"
CATALOG_PATH = ROOT_DIR / "data" / "catalog" / "exoplanet_binary_model_imputed.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "exoplanet_model.json"
FEATURES_PATH = MODELS_DIR / "selected_features.json"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def add_astropy_features(df):
    """Convertit RA/DEC en coordonnées Galactiques via Astropy."""
    if "ra" in df.columns and "dec" in df.columns:
        try:
            coords = SkyCoord(
                ra=df["ra"].values * u.degree,
                dec=df["dec"].values * u.degree,
                frame="icrs"
            )
            df["glon"] = coords.galactic.l.degree
            df["glat"] = coords.galactic.b.degree
            df = df.drop(columns=["ra", "dec"], errors="ignore")
        except Exception as e:
            logging.error(f"Erreur astropy : {e}")
    return df


def run_hybrid_training():
    logging.info("=== Lancement de l'Entraînement Hybride (Option C) ===")
    
    if not SIGNAL_PATH.exists() or not CATALOG_PATH.exists():
        logging.error("Fichiers de données (Signal ou Catalogue) introuvables.")
        return

    # 1. CHARGEMENT DES DEUX DATASETS
    df_signal = pd.read_csv(SIGNAL_PATH)
    df_catalog = pd.read_csv(CATALOG_PATH)
    
    logging.info(f"Signal Dataset: {len(df_signal)} cibles étudiées.")
    logging.info(f"Catalog Dataset: {len(df_catalog)} KOIs disponibles.")

    # 2. PRÉPARATION DU CATALOGUE
    # On supprime les KOI dupliqués sur le même kepid (étoile multi-planétaire)
    # pour pouvoir faire un inner join 1:1 avec les signaux de `overnight_features`
    df_catalog = df_catalog.drop_duplicates(subset=["kepid"]).copy()
    
    # Conversion spatiale
    df_catalog = add_astropy_features(df_catalog)

    # 3. FUSION (MERGE)
    # Le inner join garantit un alignement parfait de l'échelle Kepler ID
    df = pd.merge(df_signal, df_catalog, on="kepid", how="inner")
    logging.info(f"Fusion Hybride accomplie : Matrice finale [ {len(df)} x {df.shape[1]} ]")

    # 4. SÉLECTION DES FEATURES
    # On garde : TSFRESH (flux__), Maths (sci_), Physique (koi_), Astrométrie (glon, glat)
    features_cols = [c for c in df.columns if 
                     c.startswith("flux__") or 
                     c.startswith("sci_") or 
                     c.startswith("koi_") or 
                     c in ["glon", "glat"]]
    
    # On exclut formellement les identifiants et les cibles pour prévenir la fraude (Leakage)
    drop_cols = ["koi_disposition", "koi_pdisposition", "kepler_name", "kepoi_name", "target_planet", "target_label"]
    features_cols = [c for c in features_cols if c not in drop_cols]
    
    X = df[features_cols].copy()
    y = df["target_label"].astype(int) # La cible de vérité validée
    
    # Nettoyage (Inf -> NaN -> 0)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    logging.info(f"Dimensions pour XGBoost : {len(X)} cibles, {X.shape[1]} features (Signal + Physique).")

    # 5. SPLIT 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # 6. CONSTUCTION DU MODÈLE XGBOOST
    # Puisque nous avons beaucoup de features, max_depth=6 aide à capturer l'intéraction
    pos_weight = (y_train == 0).sum() / max(1, y_train.sum())
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    t0 = time.time()
    
    # 7. VALIDATION CROISÉE
    logging.info("Exécution de la validation croisée CV (5-folds)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_results = cross_validate(
        model, X_train, y_train, 
        cv=cv, scoring=['accuracy', 'f1', 'roc_auc'],
        n_jobs=-1
    )
    
    cv_acc = np.mean(cv_results['test_accuracy'])
    cv_acc_std = np.std(cv_results['test_accuracy'])
    cv_f1  = np.mean(cv_results['test_f1'])
    cv_f1_std = np.std(cv_results['test_f1'])
    cv_auc = np.mean(cv_results['test_roc_auc'])
    logging.info(f"CV Hybride -> Accuracy: {cv_acc:.3f} | F1: {cv_f1:.3f} | AUC: {cv_auc:.3f}")

    # 8. ENTRAÎNEMENT GLOBAL SUR LE TRAIN SET
    logging.info("Entraînement sur la totalité du Superset d'apprentissage...")
    model.fit(X_train, y_train)

    # 9. TEST SUR DONNÉES INÉDITES
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    logging.info(f"=== RÉSULTATS SUR LE TEST SET (Option C - Hybride) ===")
    logging.info(f"Accuracy : {acc:.4f}")
    logging.info(f"Precision : {prec:.4f}")
    logging.info(f"Recall : {rec:.4f}")
    logging.info(f"F1-Score : {f1:.4f}")
    
    # Exporter le TOP Features pour constater le mélange Signal/Catalog
    features_list = list(X.columns)
    imp = model.feature_importances_
    imp_df = pd.DataFrame({"name": features_list, "importance": imp}).sort_values(by="importance", ascending=False)
    
    logging.info(f"Top 5 des descripteurs (Preuve de l'Hybridation) :\n{imp_df.head(5).to_string(index=False)}")

    # 10. SAUVEGARDE ET REMPLACEMENT DU MODÈLE ACTUEL
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True)
        
    model.save_model(MODEL_PATH)
    
    # App.py recevra un dict panachable (Signal + NASA)
    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(features_list, f, indent=4)
        
    metrics = {
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_accuracy": float(acc),
        "test_f1": float(f1),
        "test_auc_roc": float(auc),
        "cv_accuracy_mean": float(cv_acc),
        "cv_accuracy_std": float(cv_acc_std),
        "cv_f1_mean": float(cv_f1),
        "cv_f1_std": float(cv_f1_std),
        "n_features_selected": len(features_list),
        "n_features_total": len(features_list),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "confusion_matrix": cm,
        "top_features": imp_df.head(20).to_dict('records')
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"Entraînement Terminé. Modèle Hybride verrouillé en {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    run_hybrid_training()
