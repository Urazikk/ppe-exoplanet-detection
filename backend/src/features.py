import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

def prepare_for_tsfresh(lc_flat, target_id):
    """
    Transforme un objet LightCurve en DataFrame compatible avec TSFRESH.
    Correction : Force la conversion en numpy array standard pour éviter les erreurs de MaskedNDArray.
    """
    # On utilise np.array() pour "nettoyer" les types Astropy/MaskedArray problématiques
    flux_values = np.array(lc_flat.flux.value, dtype=float)
    time_values = np.array(lc_flat.time.value, dtype=float)
    
    df = pd.DataFrame({
        'time': time_values,
        'flux': flux_values,
        'id': [target_id] * len(lc_flat)
    })
    return df

def extract_scientific_features(lc_flat):
    """
    Extrait manuellement quelques caractéristiques clés.
    """
    # Conversion en numpy array standard ici aussi
    flux = np.array(lc_flat.flux.value, dtype=float)
    return {
        'std_dev': np.std(flux),
        'skewness': pd.Series(flux).skew(),
        'kurtosis': pd.Series(flux).kurtosis(),
        'transit_depth_min': np.min(flux),
        'mad': np.median(np.abs(flux - np.median(flux)))
    }

def run_feature_extraction(lc_flat, target_id):
    """
    Version optimisée et corrigée pour éviter les erreurs de types Numpy/Astropy.
    """
    print(f"--- Extraction optimisée pour {target_id} ---")
    
    # 1. Formatage (avec conversion de type)
    df_ts = prepare_for_tsfresh(lc_flat, target_id)
    
    # 2. Paramètres efficaces
    settings = EfficientFCParameters()
    
    print(f"[i] Calcul sur {len(df_ts)} points...")
    
    # Extraction
    extracted_features = extract_features(
        df_ts, 
        column_id='id', 
        column_sort='time', 
        default_fc_parameters=settings,
        disable_progressbar=False,
        n_jobs=0 
    )
    
    # 3. Ajout des caractéristiques manuelles
    manual_feats = extract_scientific_features(lc_flat)
    for key, value in manual_feats.items():
        extracted_features[key] = value
        
    return extracted_features