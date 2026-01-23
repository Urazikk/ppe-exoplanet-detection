from src.p01_acquisition import fetch_lightcurve
from src.p02_preprocessing import clean_lightcurve
from src.p04_features import run_feature_extraction
import os
import time
import numpy as np

def test_features():
    target = "Kepler-10"
    print(f"--- Démarrage du test TURBO : {target} ---")
    
    # 1. Acquisition
    lc_raw = fetch_lightcurve(target)
    
    if lc_raw:
        # 2. Nettoyage AGRESSIF pour la rapidité
        lc_clean = clean_lightcurve(lc_raw, quality="ultra")
        lc_clean = lc_clean.remove_nans()
        
        print(f"[i] Points à analyser après réduction : {len(lc_clean)}")
        
        # 3. Extraction
        try:
            start = time.time()
            # On passe l'objet nettoyé, features.py s'occupe de la conversion de type
            features_df = run_feature_extraction(lc_clean, target)
            
            print(f"\n✅ Terminé en {time.time() - start:.2f}s !")
            print(features_df.iloc[:, :5].to_string())
            
            # Sauvegarde
            os.makedirs("data/processed/", exist_ok=True)
            features_df.to_csv(f"data/processed/{target}_features.csv", index=False)
            print(f"[OK] Fichier sauvegardé dans data/processed/")
            
        except Exception as e:
            print(f"❌ Erreur : {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_features()