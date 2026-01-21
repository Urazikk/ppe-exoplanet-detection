from acquisition import fetch_lightcurve
from preprocessing import clean_lightcurve
from features import run_feature_extraction
import os
import time

def test_features():
    target = "Kepler-10"
    print(f"--- Démarrage du test TURBO : {target} ---")
    
    # 1. Acquisition
    lc_raw = fetch_lightcurve(target)
    
    if lc_raw:
        # 2. Nettoyage AGRESSIF pour la rapidité
        # On utilise 'ultra' pour diviser le nombre de points par 50
        # TSFRESH adore les séries entre 1000 et 10000 points.
        lc_clean = clean_lightcurve(lc_raw, quality="ultra")
        lc_clean = lc_clean.remove_nans()
        
        print(f"[i] Points à analyser après réduction : {len(lc_clean)}")
        
        # 3. Extraction
        try:
            start = time.time()
            features_df = run_feature_extraction(lc_clean, target)
            
            print(f"\n✅ Terminé en {time.time() - start:.2f}s !")
            print(features_df.iloc[:, :5].to_string())
            
            # Sauvegarde
            os.makedirs("data/processed/", exist_ok=True)
            features_df.to_csv(f"data/processed/{target}_features.csv", index=False)
            
        except Exception as e:
            print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    test_features()