import numpy as np

def augment_signal(lc_flat):
    """
    Génère 3 variantes synthétiques d'un signal (Noisy, Deep, Shallow).
    Ce module est indépendant et ne nécessite pas d'autres imports internes.
    """
    if lc_flat is None:
        return []
        
    variations = []
    flux_orig = np.array(lc_flat.flux.value, dtype=float)
    
    # 1. Variante Bruitée
    lc_noisy = lc_flat.copy()
    noise = np.random.normal(0, 0.00018, len(flux_orig))
    lc_noisy.flux = flux_orig + noise
    variations.append(("noisy", lc_noisy))
    
    # 2. Variante "Transit Profond"
    lc_deep = lc_flat.copy()
    lc_deep.flux = 1.0 + ((flux_orig - 1.0) * 2.0)
    variations.append(("deep", lc_deep))
    
    # 3. Variante "Transit Faible"
    lc_shallow = lc_flat.copy()
    lc_shallow.flux = 1.0 + ((flux_orig - 1.0) * 0.4)
    variations.append(("shallow", lc_shallow))

    return variations