import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters


def extract_scientific_features(lc_flat):
    """
    Calcule des metriques statistiques manuelles specifiques a l'astronomie.
    """
    flux = np.array(lc_flat.flux.value, dtype=float)
    flux = flux[~np.isnan(flux)]

    if len(flux) == 0:
        return {}

    feats = {
        'sci_std_dev': np.std(flux),
        'sci_skewness': float(pd.Series(flux).skew()),
        'sci_kurtosis': float(pd.Series(flux).kurtosis()),
        'sci_transit_depth_min': np.min(flux),
        'sci_mad': np.median(np.abs(flux - np.median(flux))),
        'sci_peak_to_peak': np.ptp(flux),
        'sci_amplitude': np.max(flux) - np.min(flux)
    }
    return feats


def run_feature_extraction(lc_flat, target_id):
    """
    Extrait les caracteristiques TSFRESH + metriques scientifiques.
    """
    if lc_flat is None:
        return None

    # 1. Preparation du DataFrame pour TSFRESH
    df_ts = pd.DataFrame({
        'time': np.array(lc_flat.time.value, dtype=float),
        'flux': np.array(lc_flat.flux.value, dtype=float),
        'id': [target_id] * len(lc_flat)
    })

    # Nettoyage anti-NaN
    df_ts.dropna(subset=['flux', 'time'], inplace=True)

    if len(df_ts) < 10:
        print(f"   Echantillon {target_id} ignore : trop peu de donnees.")
        return None

    # 2. Extraction via TSFRESH
    settings = EfficientFCParameters()

    try:
        extracted = extract_features(
            df_ts,
            column_id='id',
            column_sort='time',
            default_fc_parameters=settings,
            disable_progressbar=True,
            n_jobs=0
        )

        extracted.reset_index(inplace=True)
        extracted.rename(columns={'id': 'target_id'}, inplace=True)

        # 3. Ajout des metriques scientifiques
        sci_feats = extract_scientific_features(lc_flat)
        for key, value in sci_feats.items():
            extracted[key] = value

        return extracted

    except Exception as e:
        print(f"   Erreur TSFRESH pour {target_id}: {e}")
        return None