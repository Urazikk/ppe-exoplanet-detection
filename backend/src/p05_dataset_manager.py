import pandas as pd
import os
from src.p02_preprocessing import clean_and_flatten
from src.p04_features import run_feature_extraction


def build_final_csv(lc_list, labels, output_path="data/processed/training_dataset.csv"):
    """
    Prend des courbes, les nettoie, extrait les features et sauve en CSV.
    lc_list: Liste d'objets LightCurve (reels ou augmentes).
    labels: Liste des etiquettes (0 pour non-planete, 1 pour planete).
    """
    all_rows = []

    for i, (lc, label) in enumerate(zip(lc_list, labels)):
        # 1. Toujours pretraiter
        lc_clean = clean_and_flatten(lc)
        if lc_clean is None:
            continue

        # 2. Extraire les caracteristiques via TSFRESH + scientifiques
        target_id = f"sample_{i}"
        row = run_feature_extraction(lc_clean, target_id)
        if row is not None:
            row['target_label'] = label
            all_rows.append(row)

    if not all_rows:
        print("Aucun echantillon valide extrait.")
        return None

    # 3. Creation du DataFrame et sauvegarde
    df = pd.concat(all_rows, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset sauvegarde : {len(df)} echantillons dans {output_path}")
    return df