import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.p01_acquisition import fetch_lightcurve
from src.p02_preprocessing import clean_and_flatten
from src.p04_features import run_feature_extraction
import time


def test_features():
    target = "Kepler-10"
    print(f"--- Test extraction features : {target} ---")

    # 1. Acquisition
    lc_raw = fetch_lightcurve(target)

    if lc_raw:
        # 2. Nettoyage
        lc_clean = clean_and_flatten(lc_raw, quality="ultra")

        if lc_clean is None:
            print("Echec du preprocessing.")
            return

        print(f"Points a analyser : {len(lc_clean)}")

        # 3. Extraction
        try:
            start = time.time()
            features_df = run_feature_extraction(lc_clean, target)

            elapsed = time.time() - start
            print(f"\nTermine en {elapsed:.2f}s")
            print(f"Nombre de features : {features_df.shape[1]}")
            print(features_df.iloc[:, :5].to_string())

            # Sauvegarde
            os.makedirs("data/processed/", exist_ok=True)
            features_df.to_csv(f"data/processed/{target}_features.csv", index=False)
            print(f"Fichier sauvegarde dans data/processed/")

        except Exception as e:
            print(f"Erreur : {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    test_features()