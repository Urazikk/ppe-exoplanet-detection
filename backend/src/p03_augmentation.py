import numpy as np
from lightkurve import LightCurve


def inject_synthetic_transit(lc_flat, period, duration_hours, depth_fraction):
    """
    Injection de signal de transit artificiel dans une courbe reelle.
    """
    time = lc_flat.time.value
    flux = lc_flat.flux.value

    duration_days = duration_hours / 24.0

    t0 = np.random.uniform(0, period)
    phase = ((time - t0) % period) / period

    half_dur = (duration_days / period) / 2
    transit_mask = (phase < half_dur) | (phase > (1 - half_dur))

    new_flux = flux.copy()
    new_flux[transit_mask] -= depth_fraction

    return LightCurve(time=time, flux=new_flux)


def augment_signal_variants(lc_flat):
    """
    Genere 3 variantes d'une courbe (Noisy, Deep, Shallow).
    """
    if lc_flat is None:
        return []

    variations = []
    flux_orig = np.array(lc_flat.flux.value, dtype=float)
    time = lc_flat.time.value

    # Variante bruitee
    noise = np.random.normal(0, 0.00018, len(flux_orig))
    lc_noisy = LightCurve(time=time, flux=flux_orig + noise)
    variations.append(("noisy", lc_noisy))

    # Variante transit profond
    lc_deep = LightCurve(time=time, flux=1.0 + ((flux_orig - 1.0) * 2.0))
    variations.append(("deep", lc_deep))

    # Variante transit faible
    lc_shallow = LightCurve(time=time, flux=1.0 + ((flux_orig - 1.0) * 0.4))
    variations.append(("shallow", lc_shallow))

    return variations


def augment_dataset_global(base_lcs, use_injection=True, use_variants=True):
    """
    Combine les deux strategies d'augmentation.
    """
    augmented = []
    for lc in base_lcs:
        if use_injection:
            for _ in range(2):
                p = np.random.uniform(1.2, 18.0)
                dur = np.random.uniform(2.0, 5.0)
                dep = np.random.uniform(0.002, 0.015)
                augmented.append(inject_synthetic_transit(lc, p, dur, dep))

        if use_variants:
            variants = augment_signal_variants(lc)
            for _, v_lc in variants:
                augmented.append(v_lc)

    return augmented