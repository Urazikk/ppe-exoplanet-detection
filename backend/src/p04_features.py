"""
=============================================================================
P04 - Extraction de features (V3 — compatible modèle TSFRESH v2)
=============================================================================
Utilise EfficientFCParameters pour correspondre au modèle entraîné
sur le dataset Kaggle Kepler (100 features sélectionnées).

Le modèle attend exactement les features listées dans selected_features.json.
Les features manquantes sont mises à 0 (robuste aux différences de longueur).
"""

import pandas as pd
import numpy as np
from scipy import stats

try:
    from tsfresh import extract_features
    from tsfresh.feature_extraction import EfficientFCParameters
    from tsfresh.utilities.dataframe_functions import impute
    HAS_TSFRESH = True
except ImportError:
    HAS_TSFRESH = False
    print("[WARN] tsfresh non installé — features limitées.")


# =============================================================================
# Features scientifiques (fallback + complément)
# =============================================================================

def extract_scientific_features(lc_flat):
    """Features statistiques manuelles pour la détection de transits."""
    flux = np.array(lc_flat.flux.value, dtype=float)
    flux = flux[~np.isnan(flux)]

    if len(flux) < 50:
        return {}

    median_flux = np.median(flux)
    std_flux = np.std(flux)

    feats = {
        'sci_std_dev': std_flux,
        'sci_skewness': float(stats.skew(flux)),
        'sci_kurtosis': float(stats.kurtosis(flux)),
        'sci_mad': np.median(np.abs(flux - median_flux)),
        'sci_amplitude': np.ptp(flux),
        'sci_transit_depth_p1': median_flux - np.percentile(flux, 1),
        'sci_transit_depth_p5': median_flux - np.percentile(flux, 5),
        'sci_transit_depth_min': median_flux - np.min(flux),
        'sci_rms': np.sqrt(np.mean(flux**2)),
        'sci_iqr': np.percentile(flux, 75) - np.percentile(flux, 25),
        'sci_cv': std_flux / abs(median_flux) if median_flux != 0 else 0,
        'sci_max_sigma': abs(np.min(flux) - median_flux) / std_flux if std_flux > 0 else 0,
        'sci_snr_approx': (median_flux - np.percentile(flux, 1)) / std_flux if std_flux > 0 else 0,
    }

    # Transit fraction
    threshold = median_flux - 3 * feats['sci_mad']
    below = flux[flux < median_flux]
    above = flux[flux >= median_flux]
    feats['sci_below_above_ratio'] = len(below) / max(len(above), 1)
    feats['sci_transit_fraction'] = np.sum(flux < threshold) / len(flux)

    low_points = np.where(flux < threshold)[0]
    if len(low_points) > 1:
        gaps = np.diff(low_points)
        feats['sci_low_cluster_mean_gap'] = np.mean(gaps)
        feats['sci_low_cluster_std_gap'] = np.std(gaps)
    else:
        feats['sci_low_cluster_mean_gap'] = 0
        feats['sci_low_cluster_std_gap'] = 0

    return feats


# =============================================================================
# Extraction principale (EfficientFCParameters)
# =============================================================================

def run_feature_extraction(lc_flat, target_id, bls_stats=None):
    """
    Extrait les features sci_* + BLS pour le modèle 09_bls_enhanced_train.
    bls_stats : dict retourné par get_period_hint() (optionnel, déjà calculé dans app.py)
    """
    if lc_flat is None:
        return None

    sci_feats = extract_scientific_features(lc_flat)
    if not sci_feats:
        return None

    df = pd.DataFrame([sci_feats])
    df['target_id'] = target_id

    # Ajout des features BLS si disponibles
    if bls_stats:
        df['bls_snr']              = float(bls_stats.get('bls_snr', 0))
        df['bls_depth_ppm']        = float(bls_stats.get('bls_depth_ppm', 0))
        df['bls_transit_fraction'] = float(bls_stats.get('bls_transit_fraction', 0))
        df['bls_power']            = float(bls_stats.get('bls_power', 0))
        df['bls_duration_days']    = float(bls_stats.get('bls_duration_days', 0))

    return df