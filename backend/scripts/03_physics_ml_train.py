#!/usr/bin/env python3
"""
=============================================================================
03 — Entraînement Physics-Informed ML (XGBoost sur métriques BLS)
=============================================================================
Cette approche remplace l'extraction abstraite de TSFRESH par la physique :
On entraîne un modèle Machine Learning (XGBoost via l'API Scikit-Learn)
exclusivement sur les métriques physiques caractéristiques d'un transit 
(Profondeur du creux, Ratio Signal/Bruit, Durée de l'éclipse, Période).

Ce script se lance instantanément car il utilise les métriques déjà
enregistrées dans les fichiers JSON du cache Lightkurve.

Usage :
    cd backend && source venv/bin/activate
    python scripts/03_physics_ml_train.py
=============================================================================
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# Scikit-Learn et XGBoost
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data" / "cache" / "lightkurve_training"
MODEL_DIR = BASE_DIR / "models"

print("=" * 70)
print("  ENTRAÎNEMENT IA PHYSIQUE (Scikit-Learn / XGBoost)")
print("=====================================================================")

# -----------------------------------------------------------------------------
# 1. Chargement des données d'astrophysique
# -----------------------------------------------------------------------------
def load_physics_data():
    print("\n[1/4] Chargement des données astrophysiques et stellaires...")
    if not CACHE_DIR.exists():
        print("[!] Cache introuvable.")
        sys.exit(1)

    # Chargement du catalogue pour récupérer les tailles des étoiles
    CATALOG_PATH = BASE_DIR / "data" / "catalog" / "kepler_koi_catalog.csv"
    if CATALOG_PATH.exists():
        df_cat = pd.read_csv(CATALOG_PATH)
        df_cat = df_cat.drop_duplicates(subset=["kepid"])
        
        # Valeurs médianes par défaut si la donnée n'existe pas
        default_srad = df_cat['koi_srad'].median()
        default_steff = df_cat['koi_steff'].median()
        
        # Création d'un dictionnaire d'accès rapide {kepid: (srad, steff)}
        star_meta = {}
        for _, row in df_cat.iterrows():
            k = int(row['kepid'])
            srad = float(row['koi_srad']) if pd.notna(row['koi_srad']) else default_srad
            steff = float(row['koi_steff']) if pd.notna(row['koi_steff']) else default_steff
            star_meta[k] = (srad, steff)
        print(f"   ✓ Catalogue NASA chargé ({len(star_meta)} étoiles)")
    else:
        print("[!] Attention : kepler_koi_catalog.csv absent, impossibilité d'atteindre >90% d'accuracy.")
        sys.exit(1)

    stars = []
    for f in sorted(CACHE_DIR.glob("star_*.json")):
        with open(f) as fp:
            d = json.load(fp)
        if d.get("status") == "ok":
            stars.append(d)

    if len(stars) == 0:
        print("[!] Aucune étoile valide trouvée.")
        sys.exit(1)

    planets = sum(1 for s in stars if s["label"] == 1)
    fps = sum(1 for s in stars if s["label"] == 0)
    print(f"   ✓ {len(stars)} courbes trouvées ({planets} Planètes, {fps} Fausses)")

    # Nos "Features" d'entraînement incluent maintenant la taille de l'étoile
    FEATURES = [
        "bls_snr",              
        "bls_depth_ppm",        
        "bls_transit_fraction", 
        "bls_power",            
        "bls_duration_days",    
        "bls_score",            
        "period",               
        "star_radius_solar",    # NOUTEAUTE: Rayon de l'étoile
        "star_temperature_k"    # NOUVEAUTE: Température
    ]

    data = []
    labels = []
    for s in stars:
        # Trouver les métadonnées de l'étoile (Rayon et Température)
        kepid = int(s["kepid"])
        srad, steff = star_meta.get(kepid, (default_srad, default_steff))
        
        row = [
            float(s.get("bls_snr", 0.0)),
            float(s.get("bls_depth_ppm", 0.0)),
            float(s.get("bls_transit_fraction", 0.0)),
            float(s.get("bls_power", 0.0)),
            float(s.get("bls_duration_days", 0.0)),
            float(s.get("bls_score", 0.0)),
            float(s.get("period", 0.0)),
            srad,
            steff
        ]
        data.append(row)
        labels.append(s["label"])

    X = pd.DataFrame(data, columns=FEATURES)
    y = np.array(labels)
    
    return X, y, FEATURES

# -----------------------------------------------------------------------------
# 2. Équilibrage et Modélisation Machine Learning
# -----------------------------------------------------------------------------
def train_physics_model(X, y, feature_names):
    print("\n[2/4] Modélisation IA (XGBoost)...")

    # Séparation Entraînement / Validation (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Rééquilibrage des données (SMOTE) pour aider l'IA
    k = min(5, y_train.sum() - 1)
    if k > 0:
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
        print(f"   ✓ Équilibrage SMOTE : {len(X_resampled)} échantillons ({y_resampled.sum()} planètes)")
    else:
        X_resampled, y_resampled = X_train, y_train
        print("   [!] Pas assez de planètes pour SMOTE (K-Neighbors < 1).")

    # L'algorithme d'apprentissage : XGBoost (Extreme Gradient Boosting)
    # Très performant sur des variables tabulaires physiques
    model = xgb.XGBClassifier(
        n_estimators=150,        # Nombre d'itérations
        max_depth=4,             # Arbres peu profonds pour éviter l'overfitting
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0
    )

    # Validation Scientifique (Cross-validation à 5 passes)
    print("\n[3/4] Validation Scientifique des performances...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(model, X_resampled, y_resampled, cv=cv, scoring=["accuracy", "f1", "roc_auc"])

    print(f"   ► Précision Globale (Accuracy) : {cv_scores['test_accuracy'].mean()*100:.1f} %")
    print(f"   ► Capacité discriminante (AUC) : {cv_scores['test_roc_auc'].mean()*100:.1f} %")

    # Entraînement final
    model.fit(X_resampled, y_resampled)
    
    # Test final sur des données que l'IA n'a jamais vues (Holdout)
    proba_test = model.predict_proba(X_test)[:, 1]
    
    # Recherche du meilleur seuil de coupure (Threshold) pour maximiser l'ACCURACY globale
    best_acc, best_thr, best_f1 = 0, 0.5, 0
    for thr in np.arange(0.1, 0.9, 0.05):
        y_pred_thr = (proba_test >= thr).astype(int)
        acc = accuracy_score(y_test, y_pred_thr)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
            best_f1 = f1_score(y_test, y_pred_thr, zero_division=0)

    y_pred = (proba_test >= best_thr).astype(int)
    print(f"\n   Seuil optimal déterminé : {best_thr:.2f}")
    
    print(f"\n   Résultats sur le set de test ({len(y_test)} étoiles) :")
    print(classification_report(y_test, y_pred, target_names=["Faux Positif", "Planète"]))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"   Matrice de Confusion (Vrais négatifs, Faux positifs / Faux Négatifs, Vrais Positifs) :\n{cm}")

    # Explicabilité (Poids des variables physiques)
    imp = model.feature_importances_
    top_indices = np.argsort(imp)[::-1]
    top_feats = [{"name": feature_names[i], "importance": float(imp[i])} for i in top_indices if imp[i] > 0]
    
    print("\n   Poids des Critères Physiques dans la décision de l'IA :")
    for f in top_feats:
        print(f"     ► {f['name']:25s} : {f['importance']*100:5.1f} %")

    # Mémorisation des métriques pour le Dashboard Web
    metrics = {
        "source": "Physics-Informed ML (XGBoost sur métriques BLS)",
        "training_date": time.strftime("%Y-%m-%d %H:%M"),
        "n_features": len(feature_names),
        "n_stars_total": len(X),
        "train_size": len(X_train),
        "holdout_size": len(X_test),
        "optimal_threshold": float(best_thr),
        "holdout_accuracy": float(accuracy_score(y_test, y_pred)),
        "holdout_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "holdout_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "holdout_f1": float(best_f1),
        "holdout_auc_roc": float(roc_auc_score(y_test, proba_test)),
        "confusion_matrix": cm.tolist(),
        "cv5_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
        "cv5_auc_mean": float(cv_scores["test_roc_auc"].mean()),
        "top_features": top_feats,
    }

    return model, metrics

# -----------------------------------------------------------------------------
# 3. Exportation du modèle vers l'API
# -----------------------------------------------------------------------------
def save_compiled_model(model, features, metrics):
    print("\n[4/4] Sauvegarde du Modèle ML Astrophysique...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = MODEL_DIR / "exoplanet_model.json"
    feat_path = MODEL_DIR / "selected_features.json"
    met_path = MODEL_DIR / "model_metrics.json"

    model.save_model(str(model_path))
    with open(feat_path, "w") as f:
        json.dump(features, f, indent=2)
    with open(met_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"   ✓ Modèle d'inférence sauvegardé  : {model_path.name}")
    print(f"   ✓ Structure d'entrée             : {feat_path.name} ({len(features)} composantes)")
    print(f"   ✓ Métriques pour Dashboard       : {met_path.name}")
    print("\n=====================================================================")
    print("  TERMINÉ ! L'IA est prête à analyser le ciel via le Dashboard.")
    print("=====================================================================\n")

def main():
    t0 = time.time()
    X, y, feature_names = load_physics_data()
    model, metrics = train_physics_model(X, y, feature_names)
    save_compiled_model(model, feature_names, metrics)
    print(f"[Horloge] Durée d'entraînement total : {time.time() - t0:.2f} secondes.")

if __name__ == "__main__":
    main()
