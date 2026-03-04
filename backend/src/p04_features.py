"""
=============================================================================
P04 - Extraction de features (V2 optimisée)
=============================================================================
Changements par rapport à V1 :
- MinimalFCParameters au lieu de EfficientFCParameters (~30 vs ~780 features)
- Downsampling à 2000 points max avant TSFRESH (sinon c'est trop lent)
- Features scientifiques enrichies (transit-spécifiques)
- Temps par étoile : ~5-15s au lieu de ~90-120s
"""

import pandas as pd
import numpy as np
from scipy import stats
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters


# =============================================================================
# Features scientifiques manuelles (les plus utiles pour la détection)
# =============================================================================

def extract_scientific_features(lc_flat):
    """
    Calcule des métriques statistiques spécifiques à la détection de transits.
    Ces features sont souvent plus discriminantes que celles de TSFRESH
    pour notre problème spécifique.
    """
    flux = np.array(lc_flat.flux.value, dtype=float)
    flux = flux[~np.isnan(flux)]
    
    if len(flux) < 50:
        return {}
    
    median_flux = np.median(flux)
    std_flux = np.std(flux)
    
    # Features statistiques de base
    feats = {
        'sci_std_dev': std_flux,
        'sci_skewness': float(stats.skew(flux)),
        'sci_kurtosis': float(stats.kurtosis(flux)),
        'sci_mad': np.median(np.abs(flux - median_flux)),
        'sci_amplitude': np.ptp(flux),
    }
    
    # Features spécifiques aux transits
    # La profondeur du transit : écart entre la médiane et le percentile bas
    feats['sci_transit_depth_p1'] = median_flux - np.percentile(flux, 1)
    feats['sci_transit_depth_p5'] = median_flux - np.percentile(flux, 5)
    feats['sci_transit_depth_min'] = median_flux - np.min(flux)
    
    # Asymétrie du signal : un transit crée plus de points sous la médiane
    below_median = flux[flux < median_flux]
    above_median = flux[flux >= median_flux]
    feats['sci_below_above_ratio'] = len(below_median) / max(len(above_median), 1)
    
    # Fraction du temps passé en transit (proxy)
    # Un transit typique = flux < médiane - 3*MAD
    threshold = median_flux - 3 * feats['sci_mad']
    feats['sci_transit_fraction'] = np.sum(flux < threshold) / len(flux)
    
    # Concentration des points bas : les transits créent des clusters de points bas
    # alors que le bruit est aléatoire
    low_points = np.where(flux < threshold)[0]
    if len(low_points) > 1:
        gaps = np.diff(low_points)
        feats['sci_low_cluster_mean_gap'] = np.mean(gaps)
        feats['sci_low_cluster_std_gap'] = np.std(gaps)
    else:
        feats['sci_low_cluster_mean_gap'] = 0
        feats['sci_low_cluster_std_gap'] = 0
    
    # RMS du signal (Root Mean Square)
    feats['sci_rms'] = np.sqrt(np.mean(flux**2))
    
    # Rapport signal/bruit approximatif
    feats['sci_snr_approx'] = feats['sci_transit_depth_p1'] / std_flux if std_flux > 0 else 0
    
    # Interquartile range (robuste aux outliers)
    feats['sci_iqr'] = np.percentile(flux, 75) - np.percentile(flux, 25)
    
    # Coefficient de variation
    feats['sci_cv'] = std_flux / abs(median_flux) if median_flux != 0 else 0
    
    # Nombre de sigma de l'outlier le plus extreme
    feats['sci_max_sigma'] = abs(np.min(flux) - median_flux) / std_flux if std_flux > 0 else 0
    
    return feats


# =============================================================================
# Extraction principale
# =============================================================================

def run_feature_extraction(lc_flat, target_id):
    """
    Extrait les features TSFRESH (minimal) + features scientifiques.
    
    Optimisations :
    - Downsampling à 2000 points max pour TSFRESH
    - MinimalFCParameters (~30 features vs ~780)
    - Features scientifiques sur la courbe complète (pas downsampleée)
    """
    if lc_flat is None:
        return None
    
    # Données brutes
    time_arr = np.array(lc_flat.time.value, dtype=float)
    flux_arr = np.array(lc_flat.flux.value, dtype=float)
    
    # Nettoyage NaN
    valid = ~(np.isnan(time_arr) | np.isnan(flux_arr))
    time_arr = time_arr[valid]
    flux_arr = flux_arr[valid]
    
    if len(time_arr) < 50:
        return None
    
    # --- DOWNSAMPLING pour TSFRESH ---
    # On garde 2000 points max, uniformément répartis
    max_points = 2000
    if len(time_arr) > max_points:
        indices = np.linspace(0, len(time_arr) - 1, max_points, dtype=int)
        time_ds = time_arr[indices]
        flux_ds = flux_arr[indices]
    else:
        time_ds = time_arr
        flux_ds = flux_arr
    
    # DataFrame pour TSFRESH
    df_ts = pd.DataFrame({
        'time': time_ds,
        'flux': flux_ds,
        'id': [target_id] * len(time_ds)
    })
    
    try:
        # Extraction TSFRESH minimale
        extracted = extract_features(
            df_ts,
            column_id='id',
            column_sort='time',
            default_fc_parameters=MinimalFCParameters(),
            disable_progressbar=True,
            n_jobs=0
        )
        
        extracted.reset_index(inplace=True)
        extracted.rename(columns={'id': 'target_id'}, inplace=True)
        
        # Ajout des features scientifiques (sur la courbe COMPLETE, pas downsampleée)
        sci_feats = extract_scientific_features(lc_flat)
        for key, value in sci_feats.items():
            extracted[key] = value
        
        return extracted
    
    except Exception as e:
        print(f"   [!] Erreur TSFRESH pour {target_id}: {e}")
        
        # Fallback : si TSFRESH plante, on retourne seulement les features scientifiques
        sci_feats = extract_scientific_features(lc_flat)
        if sci_feats:
            df_fallback = pd.DataFrame([sci_feats])
            df_fallback['target_id'] = target_id
            return df_fallback
        
        return None