import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk

def clean_and_flatten(lc, quality="auto"):
    """
    Nettoie la courbe avec une adaptation automatique au volume de points.
    
    Args:
        lc: La courbe de lumière brute.
        quality: "auto", "high", "fast" ou "ultra".
    """
    if lc is None: return None
    
    n_points = len(lc)
    
    # 1. Nettoyage de base (Retrait des NaNs et Outliers)
    lc = lc.remove_nans().remove_outliers(sigma=7)
    
    # 2. Adaptation automatique (Binning intelligent)
    # Si on a plus de 100 000 points, on réduit pour la fluidité du CPU
    if quality == "auto":
        if n_points > 100000:
            print(f"[!] Volume important détecté ({n_points} pts). Application d'un binning...")
            lc = lc.bin(time_bin_size=0.01) 
        else:
            print(f"[i] Volume raisonnable ({n_points} pts). Pas de binning nécessaire.")
    elif quality == "fast":
        lc = lc.bin(time_bin_size=0.01)
    elif quality == "ultra":
        lc = lc.bin(time_bin_size=0.05)
    
    # 3. Flattening (Le calcul lourd)
    # window_length=401 est idéal pour retirer les tendances stellaires
    lc_flat = lc.flatten(window_length=401)
    
    return lc_flat

def fold_lightcurve(lc_flat, period, t0=None):
    """
    Replie la courbe de lumière sur la période orbitale.
    """
    if lc_flat is None: return None
    return lc_flat.fold(period=period, epoch_time=t0)

def get_period_hint(lc_flat):
    """
    Trouve une période probable via BLS (Box Least Squares).
    Utile pour l'IA et l'augmentation automatique.
    """
    if lc_flat is None: return 1.0
    period_search = np.linspace(0.5, 20, 5000)
    bls = lc_flat.to_periodogram(method='bls', period=period_search)
    return float(bls.period_at_max_power.value)

def plot_results(lc_clean, lc_folded, target_name, period):
    """
    Affiche graphiquement les résultats du prétraitement.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Graphique 1 : Signal Nettoyé et aplati
    lc_clean.scatter(ax=ax1, s=1, color='blue', label="Aplatit")
    ax1.set_title(f"Signal Nettoyé et Aplatit (Detrended) - {target_name}")
    ax1.set_ylabel("Flux Relatif")
    
    # Graphique 2 : Signal Replié (Folding)
    lc_folded.scatter(ax=ax2, s=3, color='red', label="Replié")
    ax2.set_title(f"Transit Replié (Période : {period} jours)")
    ax2.set_ylabel("Flux Relatif")
    
    plt.tight_layout()
    plt.show()