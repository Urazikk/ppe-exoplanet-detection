import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
import time

def clean_and_flatten(lc, quality="auto", progress_cb=None):
    """
    Nettoie la courbe avec une adaptation automatique au volume de points.
    
    OPTIMISATIONS v2:
    - Binning agressif pour les gros volumes (>50k points)
    - Flattening avec window_length adaptatif
    - Callback de progression pour le dashboard
    
    Args:
        lc: La courbe de lumiere brute.
        quality: "auto", "high", "fast" ou "ultra".
        progress_cb: Fonction callback(step, message) pour la progression.
    """
    if lc is None: 
        return None
    
    def report(msg):
        if progress_cb:
            progress_cb("preprocessing", msg)
    
    n_points = len(lc)
    report(f"Nettoyage de {n_points:,} points...")
    
    # 1. Nettoyage de base (Retrait des NaNs et Outliers)
    lc = lc.remove_nans().remove_outliers(sigma=5)
    
    # 2. Binning adaptatif selon le volume
    if quality == "auto":
        if n_points > 500000:
            report(f"Volume tres important ({n_points:,} pts). Binning agressif...")
            lc = lc.bin(time_bin_size=0.05)
        elif n_points > 100000:
            report(f"Volume important ({n_points:,} pts). Binning moyen...")
            lc = lc.bin(time_bin_size=0.02)
        elif n_points > 50000:
            report(f"Volume moyen ({n_points:,} pts). Binning leger...")
            lc = lc.bin(time_bin_size=0.01)
        else:
            report(f"Volume raisonnable ({n_points:,} pts). Pas de binning.")
    elif quality == "fast":
        # Mode API : toujours binner pour la rapidite
        if n_points > 200000:
            lc = lc.bin(time_bin_size=0.05)
        elif n_points > 50000:
            lc = lc.bin(time_bin_size=0.02)
        else:
            lc = lc.bin(time_bin_size=0.01)
    elif quality == "ultra":
        lc = lc.bin(time_bin_size=0.05)
    
    n_after = len(lc)
    report(f"Apres binning : {n_after:,} points. Flattening...")
    
    # 3. Flattening avec window_length adaptatif
    # Plus la courbe est courte apres binning, plus la fenetre doit etre petite
    if n_after < 500:
        wl = 101
    elif n_after < 2000:
        wl = 201
    elif n_after < 10000:
        wl = 401
    else:
        wl = 601
    
    # S'assurer que window_length est impair et < n_points
    wl = min(wl, n_after // 2)
    if wl % 2 == 0:
        wl -= 1
    wl = max(wl, 5)
    
    try:
        lc_flat = lc.flatten(window_length=wl)
    except Exception:
        # Fallback si le flattening echoue
        report("Flattening standard echoue, tentative avec fenetre reduite...")
        lc_flat = lc.flatten(window_length=max(5, min(101, n_after // 4)))
    
    report(f"Preprocessing termine ({n_after:,} points propres).")
    return lc_flat


def fold_lightcurve(lc_flat, period, t0=None):
    """
    Replie la courbe de lumiere sur la periode orbitale.
    """
    if lc_flat is None: 
        return None
    return lc_flat.fold(period=period, epoch_time=t0)


def get_period_hint(lc_flat, progress_cb=None):
    """
    Trouve une periode probable via BLS (Box Least Squares) - version rapide.
    500 periodes x 3 durations, 1 seule passe.
    """
    if lc_flat is None:
        return 1.0

    from astropy.timeseries import BoxLeastSquares

    def report(msg):
        if progress_cb:
            progress_cb("bls", msg)

    t = np.array(lc_flat.time.value, dtype=float)
    f = np.array(lc_flat.flux.value, dtype=float)
    mask = np.isfinite(t) & np.isfinite(f)
    t, f = t[mask], f[mask]

    if len(t) < 100:
        return 1.0

    time_span = t[-1] - t[0]
    max_period = min(time_span / 2, 400)
    min_period = 0.5

    periods = np.linspace(min_period, max_period, 500)
    durations = [0.02, 0.08, 0.15]

    report(f"BLS rapide : {len(periods)} periodes x {len(durations)} durations...")
    t_start = time.time()

    bls = BoxLeastSquares(t, f)
    try:
        result = bls.power(periods, duration=durations)
        best_idx = int(np.argmax(np.array(result.power)))
        best_period = float(result.period[best_idx])
    except Exception as e:
        report(f"Erreur BLS : {e}")
        return 1.0

    report(f"BLS termine en {time.time()-t_start:.1f}s. Periode : {best_period:.4f} j")
    return best_period


def plot_results(lc_clean, lc_folded, target_name, period):
    """
    Affiche graphiquement les resultats du pretraitement.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    lc_clean.scatter(ax=ax1, s=1, color='blue', label="Aplatit")
    ax1.set_title(f"Signal Nettoye et Aplatit (Detrended) - {target_name}")
    ax1.set_ylabel("Flux Relatif")
    
    lc_folded.scatter(ax=ax2, s=3, color='red', label="Replie")
    ax2.set_title(f"Transit Replie (Periode : {period} jours)")
    ax2.set_ylabel("Flux Relatif")
    
    plt.tight_layout()
    plt.show()