import pandas as pd
import numpy as np
import os

def clean_raw_dataset():
    input_path = "data/processed/training_dataset.csv"
    output_path = "data/processed/training_dataset_clean.csv"
    
    if not os.path.exists(input_path):
        print(f"❌ Erreur : Le fichier {input_path} est introuvable.")
        return

    print("--- 🧹 NETTOYAGE AGRESSIF DU DATASET ---")
    
    # 1. Chargement
    df = pd.read_csv(input_path)
    initial_rows = len(df)
    print(f"[i] Dataset chargé : {initial_rows} lignes.")

    # 2. Nettoyage des colonnes techniques (index, Unnamed)
    # Ces colonnes empêchent le dédoublonnage car elles sont souvent uniques par ligne
    cols_to_drop = [c for c in df.columns if 'Unnamed' in c or c == 'index']
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"🗑️ Colonnes techniques supprimées : {cols_to_drop}")

    # 3. Création d'une "Signature Numérique"
    # On choisit des colonnes qui ne peuvent pas être identiques par hasard entre deux étoiles
    # Si ces 3 métriques sont identiques, c'est la même étoile.
    signature_cols = [
        'flux__mean', 
        'flux__standard_deviation', 
        'flux__sum_values',
        'flux__variance'
    ]
    
    # On vérifie si ces colonnes existent (TSFRESH les génère normalement)
    existing_sig_cols = [c for c in signature_cols if c in df.columns]
    
    if existing_sig_cols:
        # On arrondit à 8 décimales pour éviter les micro-différences de calcul (float jitter)
        # qui empêcheraient de voir que ce sont des doublons
        temp_df = df.copy()
        temp_df[existing_sig_cols] = temp_df[existing_sig_cols].round(8)
        
        # On identifie les doublons basés sur cette signature
        duplicates = temp_df.duplicated(subset=existing_sig_cols, keep='first')
        df = df[~duplicates]
        
        print(f"✅ Dédoublonnage basé sur la signature scientifique ({existing_sig_cols}) terminé.")
    else:
        # Fallback : Si on n'a pas les colonnes signatures, on compare TOUTES les colonnes
        # en arrondissant toutes les valeurs numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        temp_df = df.copy()
        temp_df[numeric_cols] = temp_df[numeric_cols].round(8)
        
        df = df[~temp_df.duplicated(keep='first')]
        print("⚠️ Signature spécifique non trouvée. Dédoublonnage global sur toutes les colonnes numériques.")

    # 4. Traitement des valeurs invalides (NaN / Inf)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 5. Sauvegarde
    final_rows = len(df)
    df.to_csv(output_path, index=False)
    
    print("\n--- 🏁 RÉSULTAT DU NETTOYAGE ---")
    print(f"📊 Lignes initiales : {initial_rows}")
    print(f"✨ Lignes conservées : {final_rows}")
    print(f"🗑️ Doublons supprimés : {initial_rows - final_rows}")
    print(f"📁 Fichier : {output_path}")

if __name__ == "__main__":
    clean_raw_dataset()