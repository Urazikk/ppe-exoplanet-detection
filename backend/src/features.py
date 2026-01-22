import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

def prepare_for_tsfresh(lc_flat, target_id):
    """Formate les données pour la structure TSFRESH."""
    flux_values = np.array(lc_flat.flux.value, dtype=float)
    time_values = np.array(lc_flat.time.value, dtype=float)
    
    return pd.DataFrame({
        'time': time_values,
        'flux': flux_values,
        'id': [target_id] * len(lc_flat)
    })

def extract_scientific_features(lc_flat):
    """Calcule des métriques statistiques manuelles en plus de TSFRESH."""
    flux = np.array(lc_flat.flux.value, dtype=float)
    return {
        'std_dev': np.std(flux),
        'skewness': pd.Series(flux).skew(),
        'kurtosis': pd.Series(flux).kurtosis(),
        'transit_depth_min': np.min(flux),
        'mad': np.median(np.abs(flux - np.median(flux))),
        'peak_to_peak': np.ptp(flux)
    }

def run_feature_extraction(lc_flat, target_id):
    """
    Exécute l'extraction massive de caractéristiques.
    Important : Transforme l'index en colonne 'target_id' pour le CSV final.
    """
    if lc_flat is None:
        return None

    df_ts = prepare_for_tsfresh(lc_flat, target_id)
    settings = EfficientFCParameters()
    
    # Extraction via TSFRESH
    extracted_features = extract_features(
        df_ts, 
        column_id='id', 
        column_sort='time', 
        default_fc_parameters=settings,
        disable_progressbar=True,
        n_jobs=0 
    )
    
    # --- RÉPARATIONS STRUCTURELLES (VITAL POUR LES DOUBLONS) ---
    # On sort l'identifiant de l'index pour en faire une colonne réelle
    extracted_features.reset_index(inplace=True)
    extracted_features.rename(columns={'id': 'target_id'}, inplace=True)
    
    # Ajout des métriques manuelles
    manual_feats = extract_scientific_features(lc_flat)
    for key, value in manual_feats.items():
        extracted_features[key] = value
        
    return extracted_features