from acquisition import fetch_lightcurve, save_raw_data
from preprocessing import clean_lightcurve, fold_lightcurve, plot_results

# Paramètres pour Pi Mensae
TARGET = "Pi Mensae"
MISSION = "TESS"
AUTHOR = "SPOC"  
PERIOD = 6.268 

def run_test():
    print(f"--- TEST ÉPURÉ TESS : {TARGET} ---")
    
    # 1. Acquisition
    lc_raw = fetch_lightcurve(TARGET, mission=MISSION, author=AUTHOR)
    
    if lc_raw:
        # Sauvegarde silencieuse en arrière-plan
        save_raw_data(lc_raw, folder="data/raw/")
        
        # 2. Nettoyage
        lc_clean = clean_lightcurve(lc_raw, quality="auto")
        
        # 3. Repliement
        lc_folded = fold_lightcurve(lc_clean, period=PERIOD)
        
        # 4. Visualisation (Appelle la fonction du Canvas)
        plot_results(lc_clean, lc_folded, TARGET, PERIOD)
        print("✅ Visualisation terminée (Bleu & Rouge uniquement).")
    else:
        print("❌ Données introuvables.")

if __name__ == "__main__":
    run_test()