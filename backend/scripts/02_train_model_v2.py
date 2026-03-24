"""
=============================================================================
ENTRAINEMENT DU MODELE V2 - XGBoost avec cross-validation rigoureuse
=============================================================================
Remplace l'ancien 02_train_model.py.

Améliorations :
- Cross-validation 5-fold stratifiée (pas juste un split)
- Métriques complètes : Accuracy, Precision, Recall, F1, AUC-ROC
- Feature selection basée sur l'importance réelle
- Sauvegarde du modèle + features + métriques + matrice de confusion
- Gestion propre du déséquilibre des classes

Usage : python 02_train_model_v2.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    roc_auc_score, precision_recall_fscore_support
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFromModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_exoplanet_model():
    """
    Pipeline d'entraînement complet avec validation rigoureuse.
    """
    train_path = "data/processed/training_dataset.csv"
    test_path = "data/processed/test_dataset.csv"
    model_dir = "models"
    
    # =========================================================================
    # 1. Chargement des données
    # =========================================================================
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Erreur : datasets introuvables. Lancez d'abord 01_generate_dataset_v2.py")
        return
    
    print("=" * 60)
    print("  ENTRAINEMENT DU MODELE EXOPLANET V2")
    print("=" * 60)
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Colonnes à exclure (métadonnées, pas des features)
    metadata_cols = [
        'target_label', 'target_id', 'kepid', 'index',
        'catalog_period', 'catalog_depth_ppm', 'catalog_duration_hr',
        'catalog_planet_radius', 'catalog_star_temp', 'catalog_star_radius',
        'catalog_kepmag'
    ]
    
    # Séparation features / labels
    y_train = df_train['target_label'].astype(int)
    y_test = df_test['target_label'].astype(int)
    
    feature_cols = [c for c in df_train.columns 
                    if c not in metadata_cols 
                    and df_train[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    X_train = df_train[feature_cols].fillna(0)
    X_test = df_test[feature_cols].fillna(0)
    
    print(f"\nDonnées chargées :")
    print(f"  Train : {len(y_train)} ({sum(y_train==1)} planètes / {sum(y_train==0)} faux positifs)")
    print(f"  Test  : {len(y_test)} ({sum(y_test==1)} planètes / {sum(y_test==0)} faux positifs)")
    print(f"  Features brutes : {len(feature_cols)}")
    
    # =========================================================================
    # 2. Feature Selection
    # =========================================================================
    print("\n[1/4] Sélection des features pertinentes...")
    
    # Premier modèle rapide pour évaluer l'importance
    selector_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=42, verbosity=0
    )
    selector_model.fit(X_train, y_train)
    
    # On garde les features au-dessus de la médiane d'importance
    selector = SelectFromModel(selector_model, threshold="median", prefit=True)
    selected_mask = selector.get_support()
    selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selected_mask[i]]
    
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    
    print(f"  -> {len(selected_features)} features conservées sur {len(feature_cols)}")
    
    # =========================================================================
    # 3. Cross-validation 5-fold (sur le train set uniquement)
    # =========================================================================
    print("\n[2/4] Cross-validation 5-fold stratifiée...")
    
    # Calcul du ratio de déséquilibre
    n_neg = sum(y_train == 0)
    n_pos = sum(y_train == 1)
    scale_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    cv_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_accuracy = cross_val_score(cv_model, X_train_sel, y_train, cv=cv, scoring='accuracy')
    cv_f1 = cross_val_score(cv_model, X_train_sel, y_train, cv=cv, scoring='f1')
    cv_roc = cross_val_score(cv_model, X_train_sel, y_train, cv=cv, scoring='roc_auc')
    
    print(f"  Accuracy : {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
    print(f"  F1-Score : {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    print(f"  AUC-ROC  : {cv_roc.mean():.4f} (+/- {cv_roc.std():.4f})")
    
    # =========================================================================
    # 4. Entraînement du modèle final
    # =========================================================================
    print("\n[3/4] Entraînement du modèle final...")
    
    final_model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )
    
    final_model.fit(X_train_sel, y_train)
    
    # =========================================================================
    # 5. Évaluation sur le test set (jamais vu pendant l'entraînement)
    # =========================================================================
    print("\n[4/4] Évaluation sur le test set indépendant...")
    
    y_pred = final_model.predict(X_test_sel)
    y_proba = final_model.predict_proba(X_test_sel)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n  {'='*40}")
    print(f"  RÉSULTATS TEST SET")
    print(f"  {'='*40}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"\n  Matrice de confusion :")
    print(f"  (Vrais Négatifs={cm[0][0]}, Faux Positifs={cm[0][1]})")
    print(f"  (Faux Négatifs={cm[1][0]}, Vrais Positifs={cm[1][1]})")
    print(f"\n  Rapport détaillé :")
    print(classification_report(y_test, y_pred, target_names=['Faux Positif', 'Planète'], zero_division=0))
    
    # Top 15 features les plus importantes
    importances = final_model.feature_importances_
    feat_importance = sorted(
        zip(selected_features, importances), 
        key=lambda x: x[1], reverse=True
    )
    
    print("  Top 15 features :")
    for fname, fimp in feat_importance[:15]:
        print(f"    {fname}: {fimp:.4f}")
    
    # =========================================================================
    # 6. Sauvegarde
    # =========================================================================
    os.makedirs(model_dir, exist_ok=True)
    
    # Modèle
    final_model.save_model(f"{model_dir}/exoplanet_model.json")
    
    # Liste des features sélectionnées
    with open(f"{model_dir}/selected_features.json", "w") as f:
        json.dump(selected_features, f)
    
    # Métriques complètes (pour le dashboard et la soutenance)
    metrics = {
        "test_accuracy": float(acc),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_auc_roc": float(auc),
        "cv_accuracy_mean": float(cv_accuracy.mean()),
        "cv_accuracy_std": float(cv_accuracy.std()),
        "cv_f1_mean": float(cv_f1.mean()),
        "cv_f1_std": float(cv_f1.std()),
        "cv_auc_mean": float(cv_roc.mean()),
        "cv_auc_std": float(cv_roc.std()),
        "confusion_matrix": cm.tolist(),
        "n_features_selected": len(selected_features),
        "n_features_total": len(feature_cols),
        "train_size": len(y_train),
        "test_size": len(y_test),
        "top_features": [{"name": n, "importance": float(v)} for n, v in feat_importance[:20]]
    }
    
    with open(f"{model_dir}/model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n  Modèle sauvegardé dans {model_dir}/")
    print(f"    -> exoplanet_model.json")
    print(f"    -> selected_features.json ({len(selected_features)} features)")
    print(f"    -> model_metrics.json (métriques complètes)")


if __name__ == "__main__":
    train_exoplanet_model()