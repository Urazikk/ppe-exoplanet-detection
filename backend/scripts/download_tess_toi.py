"""
Télécharge la table TESS Objects of Interest (TOI) depuis l'API NASA TAP,
filtre les CP (Confirmed Planet) et FP (False Positive),
harmonise les colonnes avec le format Kepler KOI,
et sauvegarde le résultat dans data/catalog/tess_toi_binary.csv
"""

import sys
import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
log = logging.info

# ============================================================================
# CONFIGURATION
# ============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data" / "catalog"

OUTPUT_PATH = DATA_DIR / "tess_toi_binary.csv"

# NASA Exoplanet Archive TAP endpoint
TAP_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=SELECT+tid,toi,tfopwg_disp,"
    "pl_orbper,pl_orbpererr1,pl_orbpererr2,"
    "pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,"
    "pl_trandurh,pl_trandurherr1,pl_trandurherr2,"
    "pl_trandep,pl_trandeperr1,pl_trandeperr2,"
    "pl_rade,pl_radeerr1,pl_radeerr2,"
    "pl_eqt,pl_eqterr1,pl_eqterr2,"
    "pl_insol,pl_insolerr1,pl_insolerr2,"
    "st_teff,st_tefferr1,st_tefferr2,"
    "st_logg,st_loggerr1,st_loggerr2,"
    "st_rad,st_raderr1,st_raderr2,"
    "st_tmag,"
    "ra,dec"
    "+FROM+toi&format=csv"
)

# Mapping TESS column names → Kepler KOI column names
COLUMN_MAP = {
    "pl_orbper":      "koi_period",
    "pl_orbpererr1":  "koi_period_err1",
    "pl_orbpererr2":  "koi_period_err2",
    "pl_tranmid":     "koi_time0bk",
    "pl_tranmiderr1": "koi_time0bk_err1",
    "pl_tranmiderr2": "koi_time0bk_err2",
    "pl_trandurh":    "koi_duration",
    "pl_trandurherr1":"koi_duration_err1",
    "pl_trandurherr2":"koi_duration_err2",
    "pl_trandep":     "koi_depth",
    "pl_trandeperr1": "koi_depth_err1",
    "pl_trandeperr2": "koi_depth_err2",
    "pl_rade":        "koi_prad",
    "pl_radeerr1":    "koi_prad_err1",
    "pl_radeerr2":    "koi_prad_err2",
    "pl_eqt":         "koi_teq",
    "pl_insol":       "koi_insol",
    "pl_insolerr1":   "koi_insol_err1",
    "pl_insolerr2":   "koi_insol_err2",
    "st_teff":        "koi_steff",
    "st_tefferr1":    "koi_steff_err1",
    "st_tefferr2":    "koi_steff_err2",
    "st_logg":        "koi_slogg",
    "st_loggerr1":    "koi_slogg_err1",
    "st_loggerr2":    "koi_slogg_err2",
    "st_rad":         "koi_srad",
    "st_raderr1":     "koi_srad_err1",
    "st_raderr2":     "koi_srad_err2",
    "st_tmag":        "koi_kepmag",  # TESS magnitude ≈ Kepler magnitude
}

DISPOSITION_MAP = {
    "CP": 1,   # Confirmed Planet
    "KP": 1,   # Known Planet
    "FP": 0,   # False Positive
    "FA": 0,   # False Alarm
}


def download_tess_toi():
    log("=== Téléchargement du dataset TESS TOI depuis NASA TAP ===")
    
    # Ensure output directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    log(f"Téléchargement depuis : exoplanetarchive.ipac.caltech.edu ...")
    try:
        df = pd.read_csv(TAP_URL)
    except Exception as e:
        log(f"Erreur de téléchargement : {e}")
        sys.exit(1)
    
    log(f"Téléchargé : {len(df)} entrées TOI au total")
    log(f"Dispositions disponibles : {df['tfopwg_disp'].value_counts().to_dict()}")
    
    # Filter only CP, KP, FP, FA (usable labels)
    valid_labels = set(DISPOSITION_MAP.keys())
    df_filtered = df[df["tfopwg_disp"].isin(valid_labels)].copy()
    log(f"Après filtrage (CP/KP/FP/FA) : {len(df_filtered)} entrées")
    
    if len(df_filtered) == 0:
        log("ERREUR : Aucune entrée CP ou FP trouvée. Vérifiez l'API.")
        sys.exit(1)
    
    # Map disposition to binary target
    df_filtered["target_planet"] = df_filtered["tfopwg_disp"].map(DISPOSITION_MAP)
    
    # Rename columns to match Kepler KOI format
    df_filtered = df_filtered.rename(columns=COLUMN_MAP)
    
    # Add mission identifier
    df_filtered["mission"] = "TESS"
    
    # Keep only relevant columns (those that match Kepler + identifiers)
    kepler_cols = list(COLUMN_MAP.values()) + ["ra", "dec", "target_planet", "mission", "tid", "toi"]
    existing_cols = [c for c in kepler_cols if c in df_filtered.columns]
    df_out = df_filtered[existing_cols].copy()
    
    # Force numeric types
    numeric_cols = [c for c in df_out.columns if c not in ["mission", "tid", "toi", "target_planet"]]
    for col in numeric_cols:
        df_out[col] = pd.to_numeric(df_out[col], errors="coerce")
    
    # Impute NaN with median
    for col in numeric_cols:
        if df_out[col].isna().any():
            median_val = df_out[col].median()
            df_out[col] = df_out[col].fillna(median_val)
    
    # Stats
    n_confirmed = (df_out["target_planet"] == 1).sum()
    n_fp = (df_out["target_planet"] == 0).sum()
    log(f"Dataset final TESS : {len(df_out)} entrées")
    log(f"  → Planètes confirmées : {n_confirmed}")
    log(f"  → Faux positifs       : {n_fp}")
    log(f"  → Colonnes            : {list(df_out.columns)}")
    
    # Save
    df_out.to_csv(OUTPUT_PATH, index=False)
    log(f"Sauvegardé dans : {OUTPUT_PATH}")
    log("=== Téléchargement TESS TOI terminé ===")


if __name__ == "__main__":
    download_tess_toi()
