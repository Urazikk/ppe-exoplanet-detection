import numpy as np
from lightkurve import LightCurve

def inject_synthetic_transit(lc_flat, period, duration_hours, depth_fraction):
    """
    STRATÉGIE A : Injection de nouveau signal.
    Injecte un signal de transit artificiel dans une courbe réelle (étoile calme).
    """
    time = lc_flat.time.value
    flux = lc_flat.flux.value
    
    # Conversion de la durée en fraction de période
    duration_days = duration_hours / 24.0
    
    # Calcul des phases (0 à 1) avec décalage aléatoire
    t0 = np.random.uniform(0, period)
    phase = ((time - t0) % period) / period
    
    # Masque du transit (centré sur la phase 0)
    half_dur = (duration_days / period) / 2
    transit_mask = (phase < half_dur) | (phase > (1 - half_dur))
    
    # Injection
    new_flux = flux.copy()
    new_flux[transit_mask] -= depth_fraction
    
    return LightCurve(time=time, flux=new_flux)

def augment_signal_variants(lc_flat):
    """
    STRATÉGIE B : Variation de signal existant (Ton ancienne méthode).
    Génère 3 variantes d'une courbe contenant déjà un transit (Noisy, Deep, Shallow).
    """
    if lc_flat is None:
        return []
        
    variations = []
    flux_orig = np.array(lc_flat.flux.value, dtype=float)
    time = lc_flat.time.value
    
    # 1. Variante Bruitée (Simule une étoile plus lointaine ou un instrument moins précis)
    noise = np.random.normal(0, 0.00018, len(flux_orig))
    lc_noisy = LightCurve(time=time, flux=flux_orig + noise)
    variations.append(("noisy", lc_noisy))
    
    # 2. Variante "Transit Profond" (Simule une planète plus grosse)
    # On amplifie l'écart par rapport à la ligne de base (1.0)
    lc_deep = LightCurve(time=time, flux=1.0 + ((flux_orig - 1.0) * 2.0))
    variations.append(("deep", lc_deep))
    
    # 3. Variante "Transit Faible" (Simule une planète plus petite, limite de détection)
    lc_shallow = LightCurve(time=time, flux=1.0 + ((flux_orig - 1.0) * 0.4))
    variations.append(("shallow", lc_shallow))

    return variations

def augment_dataset_global(base_lcs, use_injection=True, use_variants=True):
    """
    Combine les deux stratégies pour maximiser la taille du dataset.
    """
    augmented = []
    for lc in base_lcs:
        # Application de la Stratégie A (Nouveaux transits)
        if use_injection:
            for _ in range(2):
                p = np.random.uniform(1.2, 18.0)
                dur = np.random.uniform(2.0, 5.0)
                dep = np.random.uniform(0.002, 0.015)
                augmented.append(inject_synthetic_transit(lc, p, dur, dep))
        
        # Application de la Stratégie B (Variations sur le signal actuel)
        if use_variants:
            variants = augment_signal_variants(lc)
            for _, v_lc in variants:
                augmented.append(v_lc)
                
    return augmented