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
    """Prétraite et extrait les caractéristiques pour une liste de courbes."""
    final_rows = []
    if not lc_list:
        return None
        
    for i, (lc, label) in enumerate(zip(lc_list, labels)):
        target_id = f"{prefix}_{i}"
        # Extraction via TSFRESH (défini dans p04)
        feats = run_feature_extraction(lc, target_id)
        if feats is not None:
            feats['target_label'] = label
            final_rows.append(feats)
    return pd.concat(final_rows, ignore_index=True) if final_rows else None

def augment_to_target(base_lcs, base_labels, target_size):
    """Augmente une liste de courbes jusqu'à atteindre la taille cible de manière équilibrée."""
    final_lcs = []
    final_labels = []
    
    if not base_lcs:
        return final_lcs, final_labels

    while len(final_lcs) < target_size:
        for lc, label in zip(base_lcs, base_labels):
            if len(final_lcs) >= target_size: break
            
            # 1. On ajoute l'original
            final_lcs.append(lc)
            final_labels.append(label)
            
            # 2. On ajoute des variantes (Injection pour les planètes, Bruit pour le reste)
            if len(final_lcs) < target_size:
                # On utilise l'augmentation globale
                augmented = augment_dataset_global([lc], use_injection=(label==1), use_variants=True)
                for aug_lc in augmented:
                    if len(final_lcs) >= target_size: break
                    final_lcs.append(aug_lc)
                    final_labels.append(label)
                    
    return final_lcs, final_labels

def main():
    print("--- 🌌 GÉNÉRATEUR DE DATASETS MASSif (OBJECTIF 500+) ---")
    
    # 1. Configuration
    try:
        total_size = int(input("Combien d'échantillons au total souhaitez-vous (ex: 500) ? : "))
        test_ratio = 0.2 
    except ValueError:
        print("❌ Nombre invalide.")
        return

    # Graines Kepler variées (Positifs)
    planet_seeds = [
        "Kepler-10", "Kepler-90", "Kepler-22", "Kepler-62", 
        "Kepler-186", "Kepler-452", "Kepler-442", "Kepler-296",
        "Kepler-11", "Kepler-20", "Kepler-37", "Kepler-18", "Kepler-68",
        "Kepler-444", "Kepler-411", "Kepler-138"
    ]
    
    # Étoiles sans planètes ou bruits connus (Négatifs) - Équilibrage crucial
    noise_seeds = [
        "KIC 8462852", "KIC 11442793", "KIC 9832227", "KIC 12555938",
        "KIC 7341234", "KIC 10001167", "KIC 3427720", "KIC 5450373",
        "KIC 6380214", "KIC 8957003", "KIC 10387306", "KIC 11853905"
    ] 

    all_seeds = planet_seeds + noise_seeds
    all_labels = [1] * len(planet_seeds) + [0] * len(noise_seeds)

    print(f"\n[1/4] 📥 Acquisition des données de base ({len(all_seeds)} cibles)...")
    successfully_acquired = []

    for s, l in zip(all_seeds, all_labels):
        lc = fetch_lightcurve(s)
        if lc:
            clean = clean_and_flatten(lc)
            if clean:
                successfully_acquired.append((clean, l, s))
                print(f"   ✅ {s} récupéré.")

    if len(successfully_acquired) < 6:
        print("❌ Pas assez de données récupérées pour un dataset de 500.")
        return

    # 2. Séparation Train/Test
    lcs_total = [x[0] for x in successfully_acquired]
    labels_total = [x[1] for x in successfully_acquired]
    
    lcs_train_base, lcs_test_base, labels_train_base, labels_test_base = train_test_split(
        lcs_total, labels_total, test_size=test_ratio, random_state=42, stratify=labels_total
    )

    # 3. Augmentation massive
    print(f"\n[2/4] 🚀 Génération des {total_size} échantillons par augmentation...")
    train_target = int(total_size * (1 - test_ratio))
    test_target = total_size - train_target

    print(f"   -> Expansion du Train Set (Objectif: {train_target})")
    train_lcs, train_labels = augment_to_target(lcs_train_base, labels_train_base, train_target)

    print(f"   -> Expansion du Test Set (Objectif: {test_target})")
    test_lcs, test_labels = augment_to_target(lcs_test_base, labels_test_base, test_target)

    # 4. Extraction des Features (Attention : c'est très long pour 500)
    print(f"\n[3/4] 📊 Extraction des caractéristiques via TSFRESH (Calcul lourd)...")
    print(f"   -> Traitement du Train Set ({len(train_lcs)} échantillons)...")
    df_train = process_and_extract(train_lcs, train_labels, "train")
    
    print(f"   -> Traitement du Test Set ({len(test_lcs)} échantillons)...")
    df_test = process_and_extract(test_lcs, test_labels, "test")

    # 5. Sauvegarde
    print(f"\n[4/4] 💾 Sauvegarde finale...")
    os.makedirs("data/processed", exist_ok=True)
    
    if df_train is not None:
        df_train.to_csv("data/processed/training_dataset.csv", index=False)
        print(f"   ✨ Train Set : {len(df_train)} échantillons")
        
    if df_test is not None:
        df_test.to_csv("data/processed/test_dataset.csv", index=False)
        print(f"   ✨ Test Set : {len(df_test)} échantillons")

    print("\n🎉 Dataset massif prêt. Vous pouvez maintenant relancer l'entraînement !")

if __name__ == "__main__":
    main()