import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
import time

def clean_and_flatten(lc, quality="auto"):
    if lc is None: return None
    
    lc = lc.remove_nans().remove_outliers(sigma=7)
    
    # Binning agressif : on reduit a ~3000-5000 points
    lc = lc.bin(time_bin_size=0.05)
    
    lc_flat = lc.flatten(window_length=101)
    
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
    Trouve une periode probable via BLS (Box Least Squares).
    
    Utilise directement astropy.timeseries.BoxLeastSquares (plus fiable que
    lightkurve pour les grandes grilles) avec :
    - Grille multi-bande : courtes, moyennes et longues periodes
    - Multi-durations de transit pour detecter tous types de transits
    - Passe d'affinement autour du meilleur candidat
    """
    if lc_flat is None: 
        return 1.0
    
    from astropy.timeseries import BoxLeastSquares
    
    def report(msg):
        if progress_cb:
            progress_cb("bls", msg)
    
    # Preparation des donnees
    t = np.array(lc_flat.time.value, dtype=float)
    f = np.array(lc_flat.flux.value, dtype=float)
    mask = np.isfinite(t) & np.isfinite(f)
    t, f = t[mask], f[mask]
    
    if len(t) < 100:
        return 1.0
    
    time_span = t[-1] - t[0]
    max_period = min(time_span / 2, 400)
    min_period = 0.5  # Minimum 0.5j pour eviter l'erreur duration > period
    
    report(f"Recherche BLS astropy : {min_period:.1f} - {max_period:.1f} jours...")
    t_start = time.time()
    
    bls = BoxLeastSquares(t, f)
    
    # === PASSE 1 : Grille multi-bande ===
    bands = []
    bands.append(np.linspace(min_period, min(5.0, max_period), 2000))
    if max_period > 5.0:
        bands.append(np.linspace(5.0, min(50.0, max_period), 1500))
    if max_period > 50.0:
        bands.append(np.linspace(50.0, max_period, 2000))
    
    periods = np.unique(np.sort(np.concatenate(bands)))
    
    # Multi-durations : transits courts (hot jupiters) a longs (habitable zone)
    durations = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]
    
    report(f"Passe 1/2 : scan multi-bande ({len(periods)} periodes, {len(durations)} durations)...")
    
    try:
        result = bls.power(periods, duration=durations)
        powers = np.array(result.power)
        best_idx = np.argmax(powers)
        best_coarse = float(result.period[best_idx])
        max_power = float(powers[best_idx])
    except Exception as e:
        report(f"Erreur BLS passe 1: {e}")
        return 1.0
    
    t_coarse = time.time() - t_start
    report(f"Passe 1 en {t_coarse:.1f}s. Candidat : {best_coarse:.4f} j (power={max_power:.8f})")
    
    # === PASSE 2 : Affinement +/- 5% ===
    margin = 0.05
    fine_min = max(min_period, best_coarse * (1 - margin))
    fine_max = min(max_period, best_coarse * (1 + margin))
    periods_fine = np.linspace(fine_min, fine_max, 1000)
    
    report(f"Passe 2/3 : affinement [{fine_min:.4f} - {fine_max:.4f}] j...")
    
    try:
        result_fine = bls.power(periods_fine, duration=durations)
        powers_fine = np.array(result_fine.power)
        best_fine_idx = np.argmax(powers_fine)
        best_fine = float(result_fine.period[best_fine_idx])
        fine_power = float(powers_fine[best_fine_idx])
        
        if fine_power >= max_power:
            best_period = best_fine
            max_power = fine_power
        else:
            best_period = best_coarse
    except Exception:
        best_period = best_coarse
    
    # === PASSE 3 : Verification des sous-harmoniques ===
    # Le BLS detecte souvent un multiple de la vraie periode (2P, 3P)
    # On teste P/2 et P/3 pour voir si le signal est plus fort
    report(f"Passe 3/3 : verification sous-harmoniques de {best_period:.4f} j...")
    
    for divisor in [2, 3]:
        sub_period = best_period / divisor
        if sub_period < min_period:
            continue
        sub_min = sub_period * 0.95
        sub_max = sub_period * 1.05
        sub_periods = np.linspace(max(min_period, sub_min), sub_max, 500)
        
        try:
            sub_result = bls.power(sub_periods, duration=durations)
            sub_powers = np.array(sub_result.power)
            sub_best_idx = np.argmax(sub_powers)
            sub_best = float(sub_result.period[sub_best_idx])
            sub_pow = float(sub_powers[sub_best_idx])
            
            # Si la sous-harmonique a au moins 50% de la puissance du pic principal,
            # c'est probablement la vraie periode (plus courte = plus physique)
            if sub_pow > max_power * 0.5:
                report(f"  Sous-harmonique P/{divisor} = {sub_best:.4f} j (power={sub_pow:.8f} > seuil)")
                best_period = sub_best
                max_power = sub_pow
        except Exception:
            pass
    
    total_time = time.time() - t_start
    report(f"BLS termine en {total_time:.1f}s. Periode detectee : {best_period:.4f} j")
    
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