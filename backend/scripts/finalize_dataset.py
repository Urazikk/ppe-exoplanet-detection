import pandas as pd
import os
import sys

# Ajout de la racine du projet (backend/) au PATH pour permettre les imports depuis 'src'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.augmentation import augment_signal
    from src.features import run_feature_extraction
    from src.acquisition import fetch_lightcurve
    from src.preprocessing import clean_lightcurve
except ImportError as e:
    print(f"❌ Erreur d'importation : {e}")
    print("💡 Assurez-vous d'être dans le dossier 'backend' et que le dossier 'src' contient bien __init__.py")
    sys.exit(1)

def find_id_column(df):
    """
    Cherche la colonne qui contient les noms des étoiles.
    On cherche une colonne qui contient des chaînes de caractères (Kepler, TIC, etc.)
    """
    # 1. Candidats prioritaires par nom
    priority_candidates = ['target_id', 'flux__id', 'id', 'target', 'Unnamed: 0']
    for cand in priority_candidates:
        if cand in df.columns:
            # Vérification sommaire : est-ce que ça ressemble à un ID (pas juste des 0 et 1)
            sample_val = str(df[cand].iloc[0])
            if not sample_val.replace('.','').isdigit() or any(c in sample_val for c in ['K', 'T', 'i', 'c']):
                return cand

    # 2. Recherche par contenu (on cherche la première colonne non-numérique ou contenant du texte)
    for col in df.columns:
        sample_val = str(df[col].iloc[0])
        # Un ID d'exoplanète contient souvent des lettres (Kepler, TIC, TOI, Pi...)
        if any(char.isalpha() for char in sample_val):
            return col
            
    return None

def run_augmentation_pipeline():
    """
    Génère le dataset augmenté à partir des 123 étoiles nettoyées.
    Chaque étoile réelle génère 3 variantes artificielles (Noisy, Deep, Shallow).
    """
    input_file = "data/processed/training_dataset_clean.csv"
    output_file = "data/processed/final_augmented_dataset.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ Erreur : {input_file} introuvable. Veuillez d'abord nettoyer vos données.")
        return

    # Chargement du dataset de base
    df_orig = pd.read_csv(input_file)
    
    # --- DÉTECTION INTELLIGENTE DE LA COLONNE ID ---
    id_col = find_id_column(df_orig)
    
    if id_col is None:
        print("❌ ERREUR CRITIQUE : Impossible de trouver la colonne des noms d'étoiles (ID).")
        print("💡 Votre fichier CSV semble ne contenir que des chiffres (metrics).")
        print("💡 Vérifiez que vous n'avez pas supprimé la colonne 'target_id' lors du nettoyage.")
        return

    print(f"🚀 DÉMARRAGE DE L'AUGMENTATION : {len(df_orig)} cibles réelles.")
    print(f"🔍 Colonne identifiée pour les IDs : '{id_col}'")

    all_data = [df_orig] 

    for idx, row in df_orig.iterrows():
        # Extraction sécurisée de l'ID
        target_id = str(row[id_col])
        label = row['target_label']
        
        # On ignore les lignes sans ID valide ou les valeurs aberrantes
        if not target_id or target_id.lower() in ["nan", "none", "0.0", "0", "1.0", "1"]:
            continue

        print(f"🔄 [{idx+1}/{len(df_orig)}] Traitement de {target_id}...")
        
        try:
            # 1. Détection automatique de la mission
            mission = "TESS" if any(x in target_id for x in ["TIC", "TOI", "Pi ", "LHS", "WASP", "AU ", "GJ ", "HD "]) else "Kepler"
            
            # 2. Acquisition des données NASA
            lc_raw = fetch_lightcurve(target_id, mission=mission)
            
            if lc_raw is None:
                print(f"   ⚠️ Données introuvables pour {target_id}.")
                continue
            
            # 3. Prétraitement adaptatif
            lc_clean = clean_lightcurve(lc_raw, quality="auto")
            
            # 4. Génération des clones
            variations = augment_signal(lc_clean)
            
            # 5. Extraction des caractéristiques pour chaque clone
            for suffix, lc_var in variations:
                new_id = f"{target_id}_{suffix}"
                df_var = run_feature_extraction(lc_var, new_id)
                df_var['target_label'] = label
                all_data.append(df_var)
                
            # Sauvegarde de secours régulière
            if idx % 5 == 0:
                pd.concat(all_data, ignore_index=True).to_csv(output_file, index=False)
                
        except Exception as e:
            print(f"   ⚠️ Échec critique pour {target_id} : {e}")
            continue

    # 6. Fusion finale
    print("\n--- 🏁 FINALISATION DU GIGA DATASET ---")
    if len(all_data) > 1:
        final_df = pd.concat(all_data, ignore_index=True).fillna(0)
        final_df.to_csv(output_file, index=False)
        print(f"✨ TERMINÉ : Dataset créé avec {len(final_df)} échantillons.")
    else:
        print("❌ Aucune donnée n'a pu être augmentée.")

if __name__ == "__main__":
    run_augmentation_pipeline()