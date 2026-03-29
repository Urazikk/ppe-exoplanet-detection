import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
import time

def clean_and_flatten(lc, quality="auto"):
    if lc is None:
        return None

    lc = lc.remove_nans().remove_outliers(sigma=7)

    if len(lc) == 0:
        print("   [Preprocessing] Courbe vide après remove_nans/remove_outliers.")
        return None

    # Binning agressif : on reduit a ~3000-5000 points
    try:
        lc = lc.bin(time_bin_size=0.05)
    except Exception as e:
        print(f"   [Preprocessing] Erreur binning : {e} — on continue sans binning.")

    if len(lc) == 0:
        print("   [Preprocessing] Courbe vide après binning.")
        return None

    # window_length doit être impair et < len(lc)
    win = min(101, len(lc) - 1)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        print("   [Preprocessing] Pas assez de points pour flatten.")
        return None

    lc_flat = lc.flatten(window_length=win)

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