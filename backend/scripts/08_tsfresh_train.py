"""
=============================================================================
Script d'entraînement XGBoost sur les features brutes de Signal Processing
(Option B - TSFRESH + Stats Scientifiques)
=============================================================================
Ce modèle se libère totalement de l'influence du catalogue NASA (Méta-données).
Il analyse _exclusivement_ la forme mathématique des transits et du bruit
stellaire (variance, asymétrie, profondeur calculée) générés par le
fichier overnight_features.csv.
"""

import os
import json
import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ============================================================================
# CONFIGURATION DES CHEMINS
# ============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

DATA_PATH = DATA_DIR / "overnight_features.csv"
MODEL_PATH = MODELS_DIR / "exoplanet_model.json"
FEATURES_PATH = MODELS_DIR / "selected_features.json"
METRICS_PATH = MODELS_DIR / "model_metrics.json"

RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")


def run_tsfresh_training():
    logging.info("=== Lancement de l'entraînement Signal Processing (TSFRESH) ===")
    
    if not DATA_PATH.exists():
        logging.error(f"Le dataset n'existe pas : {DATA_PATH}")
        return

    logging.info(f"Lecture de {DATA_PATH.name} ...")
    df = pd.read_csv(DATA_PATH)
    
    # Validation du Dataset
    if "target_label" not in df.columns:
        logging.error("Colonne cible 'target_label' introuvable.")
        return

    # 1. SÉLECTION STRICTE DES FEATURES DE SIGNAL
    # On garantit l'abolition totale du data leakage spatial/astrométrique :
    # Les seules colonnes admissibles pour l'IA sont les statistiques
    # relatives au flux mesuré.
    signal_cols = [c for c in df.columns if c.startswith("flux__") or c.startswith("sci_")]
    
    # 2. PRÉPARATION DE LA MATRICE
    X = df[signal_cols].copy()
    y = df["target_label"].astype(int)
    
    # Nettoyage des infinités logiques extraites mathématiquement
    X = X.replace([np.inf, -np.inf], np.nan)
    # Imputation minimale (les signaux non viables sont remis à 0)
    X = X.fillna(0)
    
    logging.info(f"Dimensions : {len(X)} courbes analysées | {X.shape[1]} caractéristiques temporelles retenues.")
    logging.info(f"Répartition Planètes = {(y == 1).sum()} / Faux Positifs = {(y == 0).sum()}")

    # 3. SPLIT STRATIFIÉ
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # 4. MODÈLE XGBOOST - Paramétrage orienté traitement de signal
    # Les arbres de décision sont contraints pour prévenir l'overfitting
    # sur des anomalies temporelles du flux.
    pos_weight = (y_train == 0).sum() / max(1, y_train.sum())
    
    model = XGBClassifier(
        n_estimators=250,
        max_depth=6,         # Légèrement plus profond pour capter les micro-variations corrélées
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    t0 = time.time()
    
    # 5. VALIDATION CROISÉE
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
    logging.info(f"Validation Croisée -> Accuracy: {cv_acc:.3f} | F1: {cv_f1:.3f} | AUC: {cv_auc:.3f}")

    # 6. ENTRAÎNEMENT GLOBAL SUR LE TRAIN SET
    logging.info("Entraînement du modèle final...")
    model.fit(X_train, y_train)

    # 7. METRIQUES FINALES TEST SET
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    logging.info(f"=== RESULTATS SUR LE TEST SET INÉDIT ({len(y_test)} CIBLES) ===")
    logging.info(f"Accuracy Test : {acc:.4f}")
    logging.info(f"F1-Score Test : {f1:.4f}")
    logging.info(f"ROC-AUC Test  : {auc:.4f}")
    logging.info("\n" + classification_report(y_test, y_pred))

    # 8. EXPORTATION DES CONDITIONS EXACTES DU MODÈLE POUR APP.PY
    features_list = list(X.columns)
    imp = model.feature_importances_
    imp_df = pd.DataFrame({"name": features_list, "importance": imp}).sort_values(by="importance", ascending=False)
    logging.info(f"Top 5 des signaux biométriques (Features) les plus importants :\n{imp_df.head(5).to_string(index=False)}")

    if not MODELS_DIR.exists():
        MODELS_DIR.mkdir(parents=True)
        
    model.save_model(MODEL_PATH)
    
    # /!\ SAUVEGARDE CRITIQUE POUR L'API FLASK
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

    logging.info(f"Le Modèle a été verrouillé en {time.time()-t0:.1f}s et exporté dans {MODEL_PATH.name}")


if __name__ == "__main__":
    run_tsfresh_training()
