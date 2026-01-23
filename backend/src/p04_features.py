import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

def extract_scientific_features(lc_flat):
    """
    Calcule des métriques statistiques manuelles spécifiques à l'astronomie.
    Ces caractéristiques "métier" aident l'IA à mieux comprendre la forme des transits.
    """
    # Conversion forcée en numpy array pour éviter les objets complexes astropy
    flux = np.array(lc_flat.flux.value, dtype=float)
    
    # Nettoyage des NaNs pour les calculs numpy
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
    Extrait les caractéristiques TSFRESH et les combine avec les métriques scientifiques.
    Supprime systématiquement les NaNs pour éviter l'erreur ValueError de tsfresh.
    """
    if lc_flat is None: return None
    
    # 1. Préparation du DataFrame pour TSFRESH
    df_ts = pd.DataFrame({
        'time': np.array(lc_flat.time.value, dtype=float),
        'flux': np.array(lc_flat.flux.value, dtype=float),
        'id': [target_id] * len(lc_flat)
    })
    
    # --- NETTOYAGE CRITIQUE ANTI-NaN ---
    # tsfresh plante s'il y a des NaNs dans 'flux' ou 'time'
    initial_len = len(df_ts)
    df_ts.dropna(subset=['flux', 'time'], inplace=True)
    
    # Si la courbe est vide après nettoyage, on ignore cet échantillon
    if len(df_ts) < 10: # Minimum de points pour une extraction cohérente
        print(f"   ⚠️ Échantillon {target_id} ignoré : trop de données manquantes.")
        return None

    # 2. Extraction via TSFRESH
    settings = EfficientFCParameters()
    
    try:
        # On utilise n_jobs=0 pour éviter les problèmes de multiprocessing sur macOS
        extracted = extract_features(
            df_ts, 
            column_id='id', 
            column_sort='time', 
            default_fc_parameters=settings,
            disable_progressbar=True,
            n_jobs=0 
        )
        
        # Nettoyage de l'index
        extracted.reset_index(inplace=True)
        extracted.rename(columns={'id': 'target_id'}, inplace=True)
        
        # 3. Ajout des métriques scientifiques manuelles
        sci_feats = extract_scientific_features(lc_flat)
        for key, value in sci_feats.items():
            extracted[key] = value
            
        return extracted
        
    except Exception as e:
        print(f"   ⚠️ Erreur TSFRESH pour {target_id}: {e}")
        return None