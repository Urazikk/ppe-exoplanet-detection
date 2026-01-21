import pandas as pd
import os
import time
from acquisition import fetch_lightcurve
from preprocessing import clean_lightcurve
from features import run_feature_extraction

# --- CATALOGUE DE DONNÉES D'ENTRAÎNEMENT ÉLARGI (KEPLER + TESS) ---
# Label 1 : Planètes Confirmées
# Label 0 : Faux Positifs ou Étoiles Seules
CATALOG = [
    # --- CONFIRMÉES KEPLER (Historique) ---
    {"id": "Kepler-10", "label": 1}, {"id": "Kepler-90", "label": 1},
    {"id": "Kepler-1", "label": 1},  {"id": "Kepler-2", "label": 1},
    {"id": "Kepler-8", "label": 1},  {"id": "Kepler-11", "label": 1},
    
    # --- CONFIRMÉES TESS (Rapides et légères) ---
    {"id": "Pi Mensae", "label": 1},   # TIC 261136679
    {"id": "TOI-700", "label": 1},     # Système multi-planétaire célèbre
    {"id": "TOI-270", "label": 1},     # Super-Terres et mini-Neptunes
    {"id": "TOI-175", "label": 1},     # L 98-59
    {"id": "TOI-132", "label": 1},     # Neptune chaude
    {"id": "TOI-1148", "label": 1},    # Saturne chaude
    {"id": "WASP-18", "label": 1},     # Jupiter ultra-chaude (Transit très profond)
    {"id": "LHS 3844", "label": 1},    # Planète tellurique
    {"id": "AU Mic", "label": 1},      # Étoile jeune avec planète
    {"id": "GJ 357", "label": 1},      # Système avec Super-Terre
    {"id": "HD 21749", "label": 1},    # Système brillant
    
    # --- FAUX POSITIFS / BRUIT (Pour apprendre les erreurs) ---
    {"id": "Kepler-411", "label": 0}, {"id": "Kepler-466", "label": 0},
    {"id": "TIC 278825448", "label": 0}, # Binaire à éclipse (Simule un transit)
    {"id": "TIC 238196510", "label": 0}, # Binaire à éclipse
    {"id": "Kepler-699", "label": 0}, {"id": "Kepler-707", "label": 0},
    {"id": "Kepler-711", "label": 0}, {"id": "Kepler-715", "label": 0},
    {"id": "Kepler-717", "label": 0}, {"id": "Kepler-719", "label": 0},
]

def build_training_data():
    output_dir = "data/processed/"
    os.makedirs(output_dir, exist_ok=True)
    master_file = os.path.join(output_dir, "training_dataset.csv")
    
    print(f"🚀 Lancement du pipeline massif sur {len(CATALOG)} cibles.")
    
    if os.path.exists(master_file):
        master_df = pd.read_csv(master_file)
        # On vérifie les IDs déjà présents pour ne pas les refaire
        processed_ids = master_df['flux__id'].unique().tolist() if 'flux__id' in master_df.columns else []
    else:
        master_df = pd.DataFrame()
        processed_ids = []

    for sample in CATALOG:
        target = sample["id"]
        label = sample["label"]
        
        if target in processed_ids:
            print(f"⏩ {target} déjà présent dans le dataset.")
            continue
            
        print(f"\n🛰️ ANALYSE : {target} (Label: {label})")
        
        try:
            # On laisse fetch_lightcurve décider de la mission automatiquement
            # mais on peut forcer TESS si l'ID commence par 'TIC' ou 'TOI'
            mission = "TESS" if ("TIC" in target or "TOI" in target or "Pi Mensae" in target) else "Kepler"
            author = "SPOC" if mission == "TESS" else "Kepler"
            
            lc_raw = fetch_lightcurve(target, mission=mission, author=author)
            if lc_raw is None: continue
            
            # Pour TESS, on n'a souvent pas besoin de binning (déjà léger)
            quality = "high" if len(lc_raw) < 100000 else "ultra"
            lc_clean = clean_lightcurve(lc_raw, quality=quality).remove_nans()
            
            # Extraction
            df_feat = run_feature_extraction(lc_clean, target)
            df_feat['target_label'] = label
            
            # Concatenation
            if master_df.empty:
                master_df = df_feat
            else:
                master_df = pd.concat([master_df, df_feat], ignore_index=True)
            
            # Sauvegarde à chaque étape
            master_df.to_csv(master_file, index=False)
            print(f"✅ {target} ajouté.")
            
        except Exception as e:
            print(f"⚠️ Erreur sur {target} : {e}")
            continue

    print(f"\n🏁 Terminé. Dataset total : {len(master_df)} exemples.")

if __name__ == "__main__":
    build_training_data()