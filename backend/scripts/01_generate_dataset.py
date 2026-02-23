import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configuration du PATH pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.p01_acquisition import fetch_lightcurve
from src.p02_preprocessing import clean_and_flatten
from src.p03_augmentation import augment_dataset_global
from src.p04_features import run_feature_extraction


def process_and_extract(lc_list, labels, prefix="sample"):
    """Extrait les caracteristiques pour une liste de courbes."""
    final_rows = []
    if not lc_list:
        return None

    for i, (lc, label) in enumerate(zip(lc_list, labels)):
        target_id = f"{prefix}_{i}"
        feats = run_feature_extraction(lc, target_id)
        if feats is not None:
            feats['target_label'] = label
            final_rows.append(feats)
        if (i + 1) % 50 == 0:
            print(f"   -> {i + 1}/{len(lc_list)} traites...")

    return pd.concat(final_rows, ignore_index=True) if final_rows else None


def augment_balanced(base_lcs, base_labels, target_size_per_class):
    """
    Augmente les donnees de maniere EQUILIBREE entre les classes.
    Chaque classe (0 et 1) atteint exactement target_size_per_class echantillons.
    """
    # Separation par classe
    lcs_pos = [lc for lc, lb in zip(base_lcs, base_labels) if lb == 1]
    lcs_neg = [lc for lc, lb in zip(base_lcs, base_labels) if lb == 0]

    def expand_class(lcs, target, class_label):
        result_lcs = []
        result_labels = []
        if not lcs:
            return result_lcs, result_labels

        # D'abord, ajouter tous les originaux
        for lc in lcs:
            if len(result_lcs) >= target:
                break
            result_lcs.append(lc)
            result_labels.append(class_label)

        # Ensuite, augmenter jusqu'a atteindre la cible
        while len(result_lcs) < target:
            for lc in lcs:
                if len(result_lcs) >= target:
                    break
                use_injection = (class_label == 1)
                augmented = augment_dataset_global(
                    [lc], use_injection=use_injection, use_variants=True
                )
                for aug_lc in augmented:
                    if len(result_lcs) >= target:
                        break
                    result_lcs.append(aug_lc)
                    result_labels.append(class_label)

        return result_lcs, result_labels

    print(f"   -> Expansion classe 1 (Planetes) : {len(lcs_pos)} -> {target_size_per_class}")
    pos_lcs, pos_labels = expand_class(lcs_pos, target_size_per_class, 1)

    print(f"   -> Expansion classe 0 (Bruit)    : {len(lcs_neg)} -> {target_size_per_class}")
    neg_lcs, neg_labels = expand_class(lcs_neg, target_size_per_class, 0)

    # Fusion et melange
    all_lcs = pos_lcs + neg_lcs
    all_labels = pos_labels + neg_labels
    indices = np.random.permutation(len(all_lcs))
    all_lcs = [all_lcs[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]

    return all_lcs, all_labels


def main():
    print("=== GENERATEUR DE DATASET (CORRIGE) ===\n")

    # 1. Configuration
    try:
        total_size = int(input("Combien d'echantillons au total ? (ex: 500) : "))
        test_ratio = 0.2
    except ValueError:
        print("Nombre invalide.")
        return

    # Graines equilibrees : 12 positifs, 12 negatifs
    planet_seeds = [
        "Kepler-10", "Kepler-90", "Kepler-22", "Kepler-62",
        "Kepler-186", "Kepler-452", "Kepler-442", "Kepler-296",
        "Kepler-11", "Kepler-20", "Kepler-37", "Kepler-18",
    ]

    noise_seeds = [
        "KIC 8462852", "KIC 11442793", "KIC 9832227", "KIC 12555938",
        "KIC 7341234", "KIC 10001167", "KIC 3427720", "KIC 5450373",
        "KIC 6380214", "KIC 8957003", "KIC 10387306", "KIC 11853905",
    ]

    all_seeds = planet_seeds + noise_seeds
    all_labels = [1] * len(planet_seeds) + [0] * len(noise_seeds)

    print(f"[1/4] Acquisition des donnees de base ({len(all_seeds)} cibles)...")
    successfully_acquired = []

    for s, l in zip(all_seeds, all_labels):
        lc = fetch_lightcurve(s)
        if lc:
            clean = clean_and_flatten(lc)
            if clean:
                successfully_acquired.append((clean, l, s))
                print(f"   OK : {s}")

    if len(successfully_acquired) < 6:
        print("Pas assez de donnees recuperees.")
        return

    # 2. Separation Train/Test AVANT augmentation
    lcs_total = [x[0] for x in successfully_acquired]
    labels_total = [x[1] for x in successfully_acquired]

    lcs_train_base, lcs_test_base, labels_train_base, labels_test_base = train_test_split(
        lcs_total, labels_total, test_size=test_ratio, random_state=42, stratify=labels_total
    )

    print(f"\n   Train base : {len(lcs_train_base)} ({sum(labels_train_base)} planetes)")
    print(f"   Test base  : {len(lcs_test_base)} ({sum(labels_test_base)} planetes)")

    # 3. Augmentation du TRAIN SET UNIQUEMENT (pas de data leakage)
    train_target = int(total_size * (1 - test_ratio))
    target_per_class = train_target // 2

    print(f"\n[2/4] Augmentation du Train Set uniquement (objectif: {train_target})...")
    train_lcs, train_labels = augment_balanced(
        lcs_train_base, labels_train_base, target_per_class
    )

    # Le TEST SET reste INTACT (donnees originales, pas d'augmentation)
    test_lcs = lcs_test_base
    test_labels = labels_test_base

    print(f"\n   Train final : {len(train_lcs)} ({sum(train_labels)} planetes / {len(train_labels) - sum(train_labels)} bruit)")
    print(f"   Test final  : {len(test_lcs)} (donnees originales, non augmentees)")

    # 4. Extraction des Features
    print(f"\n[3/4] Extraction des caracteristiques via TSFRESH...")
    print(f"   -> Train Set ({len(train_lcs)} echantillons)...")
    df_train = process_and_extract(train_lcs, train_labels, "train")

    print(f"   -> Test Set ({len(test_lcs)} echantillons)...")
    df_test = process_and_extract(test_lcs, test_labels, "test")

    # 5. Sauvegarde
    print(f"\n[4/4] Sauvegarde finale...")
    os.makedirs("data/processed", exist_ok=True)

    if df_train is not None:
        df_train.to_csv("data/processed/training_dataset.csv", index=False)
        print(f"   Train Set : {len(df_train)} echantillons")

    if df_test is not None:
        df_test.to_csv("data/processed/test_dataset.csv", index=False)
        print(f"   Test Set  : {len(df_test)} echantillons")

    print("\nDataset pret. Vous pouvez lancer l'entrainement.")


if __name__ == "__main__":
    main()