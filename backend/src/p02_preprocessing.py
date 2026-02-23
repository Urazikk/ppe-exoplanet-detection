import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk


def clean_and_flatten(lc, quality="auto"):
    """
    Nettoie la courbe avec une adaptation automatique au volume de points.
    """
    if lc is None:
        return None

    n_points = len(lc)

    # 1. Nettoyage de base
    lc = lc.remove_nans().remove_outliers(sigma=7)

    # 2. Adaptation automatique (Binning intelligent)
    if quality == "auto":
        if n_points > 100000:
            print(f"   Volume important ({n_points} pts). Binning applique.")
            lc = lc.bin(time_bin_size=0.01)
        else:
            print(f"   Volume raisonnable ({n_points} pts).")
    elif quality == "fast":
        lc = lc.bin(time_bin_size=0.01)
    elif quality == "ultra":
        lc = lc.bin(time_bin_size=0.05)

    # 3. Flattening
    lc_flat = lc.flatten(window_length=401)

    return lc_flat


def fold_lightcurve(lc_flat, period, t0=None):
    """Replie la courbe de lumiere sur la periode orbitale."""
    if lc_flat is None:
        return None
    return lc_flat.fold(period=period, epoch_time=t0)


def get_period_hint(lc_flat):
    """Trouve une periode probable via BLS (Box Least Squares)."""
    if lc_flat is None:
        return 1.0
    period_search = np.linspace(0.5, 20, 5000)
    bls = lc_flat.to_periodogram(method='bls', period=period_search)
    return float(bls.period_at_max_power.value)


def plot_results(lc_clean, lc_folded, target_name, period):
    """Affiche graphiquement les resultats du pretraitement."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    lc_clean.scatter(ax=ax1, s=1, color='blue', label="Aplati")
    ax1.set_title(f"Signal Nettoye - {target_name}")
    ax1.set_ylabel("Flux Relatif")

    lc_folded.scatter(ax=ax2, s=3, color='red', label="Replie")
    ax2.set_title(f"Transit Replie (Periode : {period} jours)")
    ax2.set_ylabel("Flux Relatif")

    plt.tight_layout()
    plt.show()