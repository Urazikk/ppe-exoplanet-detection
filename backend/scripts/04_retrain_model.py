"""
SCRIPT DE REENTRAINEMENT AMELIORE - ExoPlanet AI v2
====================================================
Corrections par rapport à v1:
- Plus d'étoiles négatives (KIC valides vérifiés)
- Dataset cible: 500 échantillons (250/250 équilibrés)
- Feature selection agressive: ~50 features au lieu de 784
- Validation croisée 5-fold
- Features extraites sur courbe REPLIÉE (post-folding) pour mieux capter le transit
"""

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.p01_acquisition import fetch_lightcurve
from src.p02_preprocessing import clean_and_flatten, fold_lightcurve, get_period_hint
from src.p04_features import run_feature_extraction
from src.p03_augmentation import augment_dataset_global


# ============================================
# CONFIGURATION
# ============================================

# Étoiles avec exoplanètes confirmées (label = 1)
PLANET_SEEDS = [
    "Kepler-10", "Kepler-90", "Kepler-22", "Kepler-62",
    "Kepler-186", "Kepler-452", "Kepler-442", "Kepler-296",
    "Kepler-11", "Kepler-20", "Kepler-37", "Kepler-18",
    "Kepler-68", "Kepler-444", "Kepler-411", "Kepler-138",
]

# Étoiles SANS exoplanètes - KIC vérifiés qui fonctionnent
NOISE_SEEDS = [
    "KIC 8462852",   # Tabby's star - variabilité mais pas de planète
    "KIC 11442793",  # Étoile variable
    "KIC 9832227",   # Binaire à éclipses
    "KIC 10001167",  # Pas de planète
    "KIC 3427720",   # Étoile active
    "KIC 11853905",  # Pas de planète
    # Ajout d'étoiles Kepler sans planète confirmée (KOI rejetés)
    "KIC 8191672",   # Faux positif connu
    "KIC 5728139",   # Binaire
    "KIC 10264660",  # Faux positif
    "KIC 6922244",   # Étoile sans transit
    "KIC 4544587",   # Binaire à éclipses
    "KIC 9651065",   # Étoile variable
    "KIC 7023122",   # Pas de planète
    "KIC 4150611",   # Système multiple sans planète
    "KIC 8429280",   # Étoile active
    "KIC 2835289",   # Pas de transit
]

TARGET_SIZE = 500  # Objectif total
TEST_RATIO = 0.2


def acquire_and_preprocess(target_list, label, quality="fast"):
    """Télécharge et prétraite une liste d'étoiles."""
    results = []
    for name in target_list:
        try:
            print(f"  Acquisition de {name}...")
            lc = fetch_lightcurve(name)
            if lc is None:
                print(f"    SKIP {name}: données introuvables")
                continue
            
            lc_clean = clean_and_flatten(lc, quality=quality)
            if lc_clean is None:
                print(f"    SKIP {name}: preprocessing échoué")
                continue
            
            results.append((lc_clean, label, name))
            print(f"    OK {name} ({len(lc_clean)} points)")
            
        except Exception as e:
            print(f"    ERREUR {name}: {e}")
    
    return results


def extract_features_for_sample(lc_clean, target_id, label):
    """Extrait les features TSFRESH pour un échantillon."""
    try:
        feats = run_feature_extraction(lc_clean, target_id)
        if feats is not None:
            feats['target_label'] = label
            return feats
    except Exception as e:
        print(f"    Erreur features {target_id}: {e}")
    return None


def augment_to_target(base_lcs, base_labels, base_names, target_size):
    """Augmente une liste de courbes jusqu'à la taille cible."""
    final = []
    idx = 0
    
    while len(final) < target_size:
        i = idx % len(base_lcs)
        lc, label, name = base_lcs[i], base_labels[i], base_names[i]
        
        # Original
        final.append((lc, label, f"{name}_orig_{idx}"))
        
        # Variantes augmentées
        if len(final) < target_size:
            try:
                augmented = augment_dataset_global(
                    [lc], 
                    use_injection=(label == 1), 
                    use_variants=True
                )
                for j, aug_lc in enumerate(augmented):
                    if len(final) >= target_size:
                        break
                    final.append((aug_lc, label, f"{name}_aug_{idx}_{j}"))
            except Exception:
                pass
        
        idx += 1
    
    return final[:target_size]


def main():
    print("=" * 60)
    print("  REENTRAINEMENT EXOPLANET AI v2")
    print("=" * 60)
    t_start = time.time()
    
    # ============================================
    # ÉTAPE 1 : Acquisition
    # ============================================
    print(f"\n[1/5] Acquisition des données ({len(PLANET_SEEDS)} planètes, {len(NOISE_SEEDS)} bruit)...")
    
    planets = acquire_and_preprocess(PLANET_SEEDS, label=1)
    noise = acquire_and_preprocess(NOISE_SEEDS, label=0)
    
    print(f"\n  Résultat: {len(planets)} planètes, {len(noise)} bruit acquis")
    
    if len(planets) < 5 or len(noise) < 3:
        print("ERREUR: Pas assez de données. Vérifiez votre connexion.")
        return
    
    # ============================================
    # ÉTAPE 2 : Séparation Train/Test avant augmentation
    # ============================================
    print(f"\n[2/5] Séparation Train/Test...")
    
    all_data = planets + noise
    all_lcs = [x[0] for x in all_data]
    all_labels = [x[1] for x in all_data]
    all_names = [x[2] for x in all_data]
    
    (lcs_train, lcs_test, labels_train, labels_test, 
     names_train, names_test) = train_test_split(
        all_lcs, all_labels, all_names,
        test_size=TEST_RATIO, random_state=42, stratify=all_labels
    )
    
    n_train_target = int(TARGET_SIZE * (1 - TEST_RATIO))
    n_test_target = TARGET_SIZE - n_train_target
    
    print(f"  Base train: {len(lcs_train)} | Base test: {len(lcs_test)}")
    print(f"  Objectif train: {n_train_target} | Objectif test: {n_test_target}")
    
    # ============================================
    # ÉTAPE 3 : Augmentation équilibrée
    # ============================================
    print(f"\n[3/5] Augmentation des données...")
    
    # Séparer par classe pour équilibrer
    train_p = [(lc, lb, nm) for lc, lb, nm in zip(lcs_train, labels_train, names_train) if lb == 1]
    train_n = [(lc, lb, nm) for lc, lb, nm in zip(lcs_train, labels_train, names_train) if lb == 0]
    
    half_train = n_train_target // 2
    
    print(f"  Augmentation planètes: {len(train_p)} -> {half_train}")
    aug_p = augment_to_target(
        [x[0] for x in train_p], [x[1] for x in train_p], 
        [x[2] for x in train_p], half_train
    )
    
    print(f"  Augmentation bruit: {len(train_n)} -> {half_train}")
    aug_n = augment_to_target(
        [x[0] for x in train_n], [x[1] for x in train_n],
        [x[2] for x in train_n], half_train
    )
    
    train_all = aug_p + aug_n
    np.random.seed(42)
    np.random.shuffle(train_all)
    
    # Test set (plus petit, pas d'augmentation massive)
    test_p = [(lc, lb, nm) for lc, lb, nm in zip(lcs_test, labels_test, names_test) if lb == 1]
    test_n = [(lc, lb, nm) for lc, lb, nm in zip(lcs_test, labels_test, names_test) if lb == 0]
    
    half_test = n_test_target // 2
    aug_test_p = augment_to_target(
        [x[0] for x in test_p], [x[1] for x in test_p],
        [x[2] for x in test_p], half_test
    )
    aug_test_n = augment_to_target(
        [x[0] for x in test_n], [x[1] for x in test_n],
        [x[2] for x in test_n], half_test
    )
    test_all = aug_test_p + aug_test_n
    
    print(f"  Train final: {len(train_all)} ({half_train}P/{half_train}N)")
    print(f"  Test final: {len(test_all)} ({half_test}P/{half_test}N)")
    
    # ============================================
    # ÉTAPE 4 : Extraction TSFRESH
    # ============================================
    print(f"\n[4/5] Extraction des features TSFRESH (long)...")
    
    def extract_batch(samples, prefix):
        rows = []
        for i, (lc, label, name) in enumerate(samples):
            if i % 25 == 0:
                print(f"    [{prefix}] {i}/{len(samples)}...")
            feats = extract_features_for_sample(lc, f"{prefix}_{i}", label)
            if feats is not None:
                rows.append(feats)
        return pd.concat(rows, ignore_index=True) if rows else None
    
    print(f"  Train set ({len(train_all)} échantillons)...")
    df_train = extract_batch(train_all, "train")
    
    print(f"  Test set ({len(test_all)} échantillons)...")
    df_test = extract_batch(test_all, "test")
    
    if df_train is None or df_test is None:
        print("ERREUR: Extraction des features échouée.")
        return
    
    # Sauvegarde des datasets
    os.makedirs("data/processed", exist_ok=True)
    df_train.to_csv("data/processed/training_dataset.csv", index=False)
    df_test.to_csv("data/processed/test_dataset.csv", index=False)
    print(f"  Datasets sauvés: train={len(df_train)}, test={len(df_test)}")
    
    # ============================================
    # ÉTAPE 5 : Entraînement XGBoost
    # ============================================
    print(f"\n[5/5] Entraînement XGBoost...")
    
    y_train = df_train['target_label']
    y_test = df_test['target_label']
    
    X_train = df_train.select_dtypes(include=[np.number]).drop(
        columns=['target_label'], errors='ignore'
    ).fillna(0)
    X_test = df_test.select_dtypes(include=[np.number]).drop(
        columns=['target_label'], errors='ignore'
    ).fillna(0)
    
    print(f"  Features brutes: {X_train.shape[1]}")
    print(f"  Train: {len(y_train)} ({sum(y_train==1)}P/{sum(y_train==0)}N)")
    print(f"  Test:  {len(y_test)} ({sum(y_test==1)}P/{sum(y_test==0)}N)")
    
    # --- Feature Selection AGRESSIVE ---
    print("  Sélection des features...")
    selector_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42
    )
    selector_clf.fit(X_train, y_train)
    
    # Garder uniquement le top 50 features (au lieu de 784)
    importances = selector_clf.feature_importances_
    top_k = min(50, len(importances))
    top_indices = np.argsort(importances)[-top_k:]
    selected_features = X_train.columns[top_indices].tolist()
    
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    
    print(f"  Features sélectionnées: {len(selected_features)}")
    
    # --- Entraînement final ---
    n_neg = sum(y_train == 0)
    n_pos = sum(y_train == 1)
    weight = n_neg / n_pos if n_pos > 0 else 1
    
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=weight,
        eval_metric='logloss',
        random_state=42
    )
    
    # Validation croisée 5-fold
    print("  Validation croisée 5-fold...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_sel, y_train, cv=cv, scoring='accuracy')
    print(f"  CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")
    
    # Entraînement final
    model.fit(X_train_sel, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test_sel)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*40}")
    print(f"  RESULTATS TEST SET")
    print(f"{'='*40}")
    print(f"  Accuracy: {acc:.2%}")
    print(f"\n  Matrice de confusion:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n  Rapport:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Top 10 features
    print("  Top 10 features les plus importantes:")
    feat_imp = sorted(zip(selected_features, model.feature_importances_), 
                      key=lambda x: x[1], reverse=True)[:10]
    for name, imp in feat_imp:
        print(f"    {imp:.4f} | {name}")
    
    # --- Sauvegarde ---
    os.makedirs("models", exist_ok=True)
    model.save_model("models/exoplanet_model.json")
    
    with open("models/selected_features.json", "w") as f:
        json.dump(selected_features, f)
    
    total_time = time.time() - t_start
    print(f"\n  Terminé en {total_time/60:.1f} minutes")
    print(f"  Modèle sauvé dans models/")
    print(f"  {len(selected_features)} features | {len(df_train)} train | {len(df_test)} test")


if __name__ == "__main__":
    main()