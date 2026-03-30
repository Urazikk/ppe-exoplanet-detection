"""
Script d'entraînement XGBoost sur le dataset Kaggle 'cumulative'.
Utilise le fichier pré-imputé: exoplanet_binary_model_imputed.csv
Tire parti de la bibliothèque astropy pour générer des features locales
galactiques dynamiques à partir de l'astrométrie ICRS (RA/Dec).
"""

import sys
import os
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

# Demande utilisateur : utilisation de la bibliothèque astropy
from astropy.coordinates import SkyCoord
import astropy.units as u

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier

# ============================================================================
# CONFIGURATION
# ============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = ROOT_DIR / "backend"
DATA_DIR = BACKEND_DIR / "data" / "catalog"
MODELS_DIR = ROOT_DIR / "models"

DATA_PATH = DATA_DIR / "exoplanet_binary_model_imputed.csv"
MODEL_PATH = MODELS_DIR / "exoplanet_model.json"
FEATURES_PATH = MODELS_DIR / "selected_features.json"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

def log(msg):
    logging.info(msg)


# ============================================================================
# FONCTIONS UTILITAIRES & ASTROPY
# ============================================================================

def add_astropy_features(df):
    """
    Utilise la bibliothèque `astropy` pour convertir les coordonnées
    équatoriales ICRS (RA, Dec) en coordonnées Galactiques (glon, glat).
    Ces variables spatiales peuvent aider le classifieur à traiter des
    biais observationnels dans la ligne de vue de Kepler.
    """
    # Verifier si ra et dec existent
    if "ra" in df.columns and "dec" in df.columns:
        log("Transformation RA/DEC -> Coordonnées Galactiques avec Astropy...")
        try:
            coords = SkyCoord(
                ra=df["ra"].values * u.degree,
                dec=df["dec"].values * u.degree,
                frame="icrs"
            )
            df["glon"] = coords.galactic.l.degree  # Longitude galactique
            df["glat"] = coords.galactic.b.degree  # Latitude galactique
            log("Features 'glon' et 'glat' générées par astropy avec succès.")
            
            # On drop les originaux pour forcer le modèle à utiliser les nouvelles coords
            # (Optionnel, mais souvent préférable pour réduire la colinéarité)
            df = df.drop(columns=["ra", "dec"], errors="ignore")
        except Exception as e:
            log(f"Erreur avec astropy : {e}")
    else:
        log("Aucune colonne RA/DEC trouvée. Transformation astropy ignorée.")
    return df


def prepare_dataset(df):
    """
    Filtre les données, élimine les fuites d'informations (leakage)
    et différencie X (features) de y (cibles).
    """
    if "target_planet" not in df.columns:
        raise ValueError("Erreur critique: La colonne cible 'target_planet' est manquante dans le dataset.")
        
    y = df["target_planet"].astype(int)

    # Identifiants et variables liés à la décision qui fuient sur le test
    drop_cols = [
        "target_planet",
        "kepid", "kepoi_name", "kepler_name", "koi_disposition", 
        "index", "target_label"
    ]
    
    # Ne garder que les colonnes numériques, puis enlever les fuites
    X = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in drop_cols if c in df.columns], 
        errors="ignore"
    )

    # Remplacement des infinis par NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # On comble les derniers nans potentiels résiduels avec la médiane
    X = X.fillna(X.median())
    
    log(f"Dataset prêt. Taille : {len(X)} astres | {X.shape[1]} features")
    log(f"Répartition cible : {y.sum()} CONFIRMED (1) | {(y == 0).sum()} FALSE POSITIVE (0)")
    
    return X, y


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_training():
    log("=== Lancement de l'entraînement XGBoost sur dataset Kaggle (Astropy features) ===")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Le fichier n'existe pas : {DATA_PATH}")
        
    log(f"Lecture du dataset : {DATA_PATH.name}")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Feature Engineering avec Astropy
    df = add_astropy_features(df)
    
    # 2. Séparation 
    X, y = prepare_dataset(df)
    
    # Séparation finale d'évaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    
    features_list = list(X.columns)
    
    log(f"Test Set = {len(X_test)} échantillons.")

    # 3. Paramètres du modèle optimisés pour les données complexes spatiales
    # On calcule un scale_pos_weight pour gérer le déséquilibre potentiel des classes
    pos_weight = (y_train == 0).sum() / max(1, y_train.sum())
    
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight, 
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    # Validation croisée 5 plis pour se rassurer de la robustesse
    log("Exécution de la validation croisée stratifiée (K=5)...")
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
    log(f"Résultats CV (Training) -> Accuracy {cv_acc_mean:.3f} | F1 {cv_f1_mean:.3f} | AUC {cv_auc_mean:.3f}")

    # 4. Entraînement final
    log("Entraînement complet sur le train set...")
    model.fit(X_train, y_train)

    # 5. Évaluation 
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    log("=== Résultats du Test Set ===")
    log(f"Accuracy : {acc:.4f}")
    log(f"F1-Score : {f1:.4f}")
    log(f"ROC-AUC  : {auc:.4f}")
    log("\n" + classification_report(y_test, y_pred))

    # Features Importances
    imp = model.feature_importances_
    imp_df = pd.DataFrame({"name": features_list, "importance": imp}).sort_values(by="importance", ascending=False)
    log(f"Top 5 Features Importantes :\n{imp_df.head(5).to_string(index=False)}")

    # 6. Sauvegarde
    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True)
        
    model.save_model(MODEL_PATH)
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
        "top_features": imp_df.head(20).to_dict('records')
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    log(f"Modèle sauvegardé dans {MODEL_PATH}")
    log(f"Features exportées dans {FEATURES_PATH}")
    log("=== Entraînement Kaggle + Astropy terminé avec succès ===")

if __name__ == "__main__":
    run_training()
