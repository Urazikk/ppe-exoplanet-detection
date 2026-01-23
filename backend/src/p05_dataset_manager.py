import pandas as pd
import os
from src.p02_preprocessing import clean_and_flatten
from src.p04_features import extract_lightcurve_features

def build_final_csv(lc_list, labels, output_path="data/processed/training_dataset.csv"):
    """
    Prend des courbes, les nettoie, extrait les features et sauve en CSV.
    lc_list: Liste d'objets LightCurve (réels ou augmentés).
    labels: Liste des étiquettes (0 pour non-planète, 1 pour planète).
    """
    all_rows = []
    
    for lc, label in zip(lc_list, labels):
        # 1. Toujours prétraiter
        lc_clean = clean_and_flatten(lc)
        if lc_clean is None: continue
        
        # 2. Extraire les caractéristiques
        row = extract_lightcurve_features(lc_clean, label=label)
        all_rows.append(row)
        
    # 3. Création du DataFrame et sauvegarde
    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset sauvegardé : {len(df)} échantillons dans {output_path}")
    return df