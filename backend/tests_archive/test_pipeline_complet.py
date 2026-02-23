import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.p01_acquisition import fetch_lightcurve
from src.p02_preprocessing import clean_and_flatten, fold_lightcurve, plot_results

# Parametres pour Pi Mensae
TARGET = "Pi Mensae"
MISSION = "TESS"
AUTHOR = "SPOC"
PERIOD = 6.268


def run_test():
    print(f"--- TEST PIPELINE : {TARGET} ---")

    # 1. Acquisition
    lc_raw = fetch_lightcurve(TARGET, mission=MISSION, author=AUTHOR)

    if lc_raw:
        # 2. Nettoyage
        lc_clean = clean_and_flatten(lc_raw, quality="auto")

        if lc_clean is None:
            print("Echec du preprocessing.")
            return

        # 3. Repliement
        lc_folded = fold_lightcurve(lc_clean, period=PERIOD)

        # 4. Visualisation
        plot_results(lc_clean, lc_folded, TARGET, PERIOD)
        print("Visualisation terminee.")
    else:
        print("Donnees introuvables.")


if __name__ == "__main__":
    run_test()