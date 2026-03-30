import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
import time


def clean_only(lc):
    """Nettoyage SANS aplatissement — pour l'extraction de features.
    Le modèle v2 a été entraîné sur du flux brut (Kaggle), pas aplati.
    """
    if lc is None:
        return None

    lc = lc.remove_nans().remove_outliers(sigma=7)

    if len(lc) == 0:
        print("   [Preprocessing] Courbe vide après remove_nans/remove_outliers.")
        return None

    # Binning pour réduire à ~3000-5000 points
    try:
        lc = lc.bin(time_bin_size=0.05)
    except Exception as e:
        print(f"   [Preprocessing] Erreur binning : {e} — on continue sans binning.")

    if len(lc) == 0:
        print("   [Preprocessing] Courbe vide après binning.")
        return None

    return lc


def clean_and_flatten(lc, quality="auto"):
    """Nettoyage + aplatissement — pour la visualisation et le BLS."""
    lc_clean = clean_only(lc)
    if lc_clean is None:
        return None

    # window_length doit être impair et < len(lc)
    win = min(101, len(lc_clean) - 1)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        print("   [Preprocessing] Pas assez de points pour flatten.")
        return None

    lc_flat = lc_clean.flatten(window_length=win)

    if len(lc_flat) == 0:
        print("   [Preprocessing] Courbe vide après flatten.")
        return None

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
    Trouve une période probable via BLS (Box Least Squares).
    Retourne (period, bls_stats) où bls_stats contient power, depth, snr, duration.
    """
    if lc_flat is None:
        return 1.0, {}

    from astropy.timeseries import BoxLeastSquares

    def report(msg):
        if progress_cb:
            progress_cb("bls", msg)

    t = np.array(lc_flat.time.value, dtype=float)
    f = np.array(lc_flat.flux.value, dtype=float)
    mask = np.isfinite(t) & np.isfinite(f)
    t, f = t[mask], f[mask]

    if len(t) < 100:
        return 1.0, {}

    time_span = t[-1] - t[0]
    max_period = min(time_span / 2, 400)
    min_period = 0.5

    periods = np.linspace(min_period, max_period, 500)
    durations = [0.02, 0.05, 0.08, 0.12, 0.15]

    report(f"BLS rapide : {len(periods)} periodes x {len(durations)} durations...")
    t_start = time.time()

    bls = BoxLeastSquares(t, f)
    try:
        result = bls.power(periods, duration=durations)
        best_idx = int(np.argmax(np.array(result.power)))
        best_period = float(result.period[best_idx])
        best_power = float(result.power[best_idx])
        best_duration = float(result.duration[best_idx])
        best_depth = float(result.depth[best_idx]) if hasattr(result, 'depth') else 0

        # SNR du BLS : (best_power - median_power) / std_power
        powers = np.array(result.power)
        powers_clean = powers[np.isfinite(powers)]
        bls_snr = (best_power - np.median(powers_clean)) / (np.std(powers_clean) + 1e-10)

        # Transit depth en ppm
        median_flux = np.median(f)
        depth_ppm = best_depth / median_flux * 1e6 if median_flux > 0 else 0

        # Fraction du transit (durée / période)
        transit_fraction = best_duration / best_period if best_period > 0 else 0

        bls_stats = {
            "bls_power": best_power,
            "bls_snr": float(bls_snr),
            "bls_depth": best_depth,
            "bls_depth_ppm": float(depth_ppm),
            "bls_duration_days": best_duration,
            "bls_transit_fraction": float(transit_fraction),
        }

    except Exception as e:
        report(f"Erreur BLS : {e}")
        return 1.0, {}

    report(f"BLS termine en {time.time()-t_start:.1f}s. Periode : {best_period:.4f} j, SNR={bls_stats['bls_snr']:.1f}")
    return best_period, bls_stats


def compute_transit_score(bls_stats):
    """
    Score de détection basé sur les indicateurs physiques du BLS.
    
    Critères utilisés par la mission Kepler :
    - BLS SNR > 7.1 = seuil de détection officiel Kepler
    - Transit depth entre 100 et 30000 ppm (planètes réalistes)
    - Transit fraction < 0.15 (les transits sont courts)
    - BLS power significatif
    
    Retourne un score entre 0 et 1.
    """
    if not bls_stats:
        return 0.5

    snr = bls_stats.get("bls_snr", 0)
    depth_ppm = abs(bls_stats.get("bls_depth_ppm", 0))
    transit_frac = bls_stats.get("bls_transit_fraction", 0)

    # 1. Score SNR (le plus important — seuil Kepler = 7.1)
    if snr >= 10:
        snr_score = 1.0
    elif snr >= 7.1:
        snr_score = 0.7 + 0.3 * (snr - 7.1) / 2.9
    elif snr >= 5:
        snr_score = 0.4 + 0.3 * (snr - 5) / 2.1
    elif snr >= 3:
        snr_score = 0.15 + 0.25 * (snr - 3) / 2
    else:
        snr_score = snr * 0.05

    # 2. Score profondeur (100-30000 ppm = planètes réalistes)
    if 100 <= depth_ppm <= 30000:
        depth_score = 1.0
    elif 30 <= depth_ppm < 100:
        depth_score = 0.5 + 0.5 * (depth_ppm - 30) / 70
    elif 30000 < depth_ppm <= 100000:
        # Trop profond — probablement binaire à éclipses
        depth_score = max(0, 0.5 - (depth_ppm - 30000) / 140000)
    else:
        depth_score = 0.1

    # 3. Score fraction de transit (< 0.15 attendu pour planètes)
    if transit_frac < 0.05:
        frac_score = 1.0
    elif transit_frac < 0.15:
        frac_score = 0.7 + 0.3 * (0.15 - transit_frac) / 0.10
    elif transit_frac < 0.3:
        frac_score = 0.3
    else:
        # Trop large — éclipse, pas transit
        frac_score = 0.05

    # Combinaison pondérée
    score = 0.55 * snr_score + 0.30 * depth_score + 0.15 * frac_score
    return round(max(0.0, min(1.0, score)), 4)


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