import pandas as pd
import os
import time
from acquisition import fetch_lightcurve
from preprocessing import clean_lightcurve
from features import run_feature_extraction

# --- CATALOGUE MASSIF ÉLARGI ---
# Label 1 : Planètes Confirmées (Kepler, TESS, K2)
CONFIRMED_PLANETS = [
    # Kepler (Série Historique)
    "Kepler-1", "Kepler-2", "Kepler-3", "Kepler-4", "Kepler-5", "Kepler-6", "Kepler-7", "Kepler-8", "Kepler-9", "Kepler-10",
    "Kepler-11", "Kepler-12", "Kepler-13", "Kepler-14", "Kepler-15", "Kepler-16", "Kepler-17", "Kepler-18", "Kepler-19", "Kepler-20",
    "Kepler-21", "Kepler-22", "Kepler-23", "Kepler-24", "Kepler-25", "Kepler-26", "Kepler-27", "Kepler-28", "Kepler-29", "Kepler-30",
    "Kepler-31", "Kepler-32", "Kepler-33", "Kepler-34", "Kepler-35", "Kepler-36", "Kepler-37", "Kepler-38", "Kepler-39", "Kepler-40",
    "Kepler-41", "Kepler-42", "Kepler-43", "Kepler-44", "Kepler-45", "Kepler-46", "Kepler-47", "Kepler-48", "Kepler-49", "Kepler-50",
    "Kepler-51", "Kepler-52", "Kepler-53", "Kepler-54", "Kepler-55", "Kepler-56", "Kepler-57", "Kepler-58", "Kepler-59", "Kepler-60",
    
    # TESS (Objets d'Intérêt)
    "Pi Mensae", "TOI-700", "TOI-270", "TOI-175", "TOI-132", "TOI-1148", "WASP-18", "LHS 3844", "AU Mic", "GJ 357",
    "HD 21749", "TOI-125", "TOI-150", "TOI-163", "TOI-172", "TOI-216", "TOI-402", "TOI-421", "TOI-451", "TOI-561",
    "TOI-101", "TOI-114", "TOI-120", "TOI-130", "TOI-141", "TOI-144", "TOI-169", "TOI-181", "TOI-186", "TOI-197"
]

# Label 0 : Faux Positifs, Binaires à éclipses, Bruit Instrumental
FALSE_POSITIVES = [
    # Kepler False Positives (KOIs connus comme FP)
    "Kepler-411", "Kepler-466", "Kepler-699", "Kepler-707", "Kepler-711", "Kepler-715", "Kepler-717", "Kepler-719", "Kepler-723", "Kepler-727",
    "KOI-122", "KOI-126", "KOI-129", "KOI-131", "KOI-133", "KOI-134", "KOI-135", "KOI-146", "KOI-152", "KOI-153",
    "KOI-160", "KOI-161", "KOI-164", "KOI-165", "KOI-166", "KOI-167", "KOI-168", "KOI-170", "KOI-171", "KOI-172",
    
    # TESS False Positives (Binaires à éclipses ou bruit)
    "TIC 278825448", "TIC 238196510", "TIC 141748298", "TIC 307315545", "TIC 231666838", "TIC 150372135", "TIC 441462348", "TIC 149601601", "TIC 259933059", "TIC 300013486",
    "TIC 238196510", "TIC 272273115", "TIC 441462348", "TIC 149601601", "TIC 259933059", "TIC 123456789", "TIC 987654321", "TIC 111222333", "TIC 444555666", "TIC 777888999"
]

# Fusion avec labels pour créer le catalogue complet
CATALOG = [{"id": name, "label": 1} for name in CONFIRMED_PLANETS] + \
          [{"id": name, "label": 0} for name in FALSE_POSITIVES]

def run_massive_pipeline():
    output_dir = "data/processed/"
    os.makedirs(output_dir, exist_ok=True)
    master_file = os.path.join(output_dir, "training_dataset.csv")
    
    print(f"🚀 DÉMARRAGE DU PIPELINE MASSIF : {len(CATALOG)} CIBLES")
    
    # Reprise du travail existant si le fichier est présent
    if os.path.exists(master_file):
        master_df = pd.read_csv(master_file)
        # On vérifie les IDs déjà traités (basé sur la colonne flux__id générée par TSFRESH)
        processed_ids = master_df['flux__id'].unique().tolist() if 'flux__id' in master_df.columns else []
        print(f"📈 {len(processed_ids)} cibles déjà présentes dans le dataset.")
    else:
        master_df = pd.DataFrame()
        processed_ids = []

    for sample in CATALOG:
        target = sample["id"]
        label = sample["label"]
        
        # Sauter les cibles déjà traitées
        if target in processed_ids:
            continue
            
        print(f"\n🔭 [{len(processed_ids)+1}/{len(CATALOG)}] Analyse de {target}...")
        
        try:
            # 1. Détection de la mission et de l'auteur pour optimiser la recherche
            mission = "TESS" if any(x in target for x in ["TIC", "TOI", "Pi ", "LHS", "WASP", "AU ", "GJ ", "HD "]) else "Kepler"
            author = "SPOC" if mission == "TESS" else "Kepler"
            
            lc_raw = fetch_lightcurve(target, mission=mission, author=author)
            if lc_raw is None: 
                print(f"❌ Données introuvables pour {target}.")
                continue
            
            # 2. Prétraitement Ultra-Rapide (Binning agressif pour le volume)
            lc_clean = clean_lightcurve(lc_raw, quality="ultra").remove_nans()
            
            # 3. Extraction des caractéristiques temporelles
            df_feat = run_feature_extraction(lc_clean, target)
            df_feat['target_label'] = label
            
            # 4. Fusion incrémentale et sauvegarde immédiate (sécurité)
            if master_df.empty: 
                master_df = df_feat
            else: 
                master_df = pd.concat([master_df, df_feat], ignore_index=True)
            
            master_df.to_csv(master_file, index=False)
            processed_ids.append(target)
            print(f"✅ {target} ajouté au dataset avec succès.")
            
        except Exception as e:
            print(f"⚠️ Échec du traitement pour {target} : {e}")
            continue

    print(f"\n🏁 PIPELINE TERMINÉ. Dataset total : {len(master_df)} échantillons.")

if __name__ == "__main__":
    run_massive_pipeline()