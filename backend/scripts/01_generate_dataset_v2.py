"""
=============================================================================
GENERATEUR DE DATASET V3 - Fold avant extraction, sans BLS
=============================================================================
Usage : python 01_generate_dataset_v2.py --total 400
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.p01_acquisition import fetch_lightcurve
from src.p02_preprocessing import clean_and_flatten, fold_lightcurve
from src.p04_features import run_feature_extraction


def load_kepler_catalog():
    """Charge le catalogue NASA (cache local)."""
    catalog_cache = "data/catalog/kepler_koi_catalog.csv"
    
    if os.path.exists(catalog_cache):
        df = pd.read_csv(catalog_cache)
        print(f"[OK] Catalogue en cache : {len(df)} entrees.")
        return df
    
    try:
        import requests
        from io import StringIO
        
        url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        query = """
        SELECT kepid, koi_disposition, koi_period, koi_depth, koi_duration,
               koi_prad, koi_steff, koi_srad, koi_kepmag
        FROM koi
        WHERE koi_disposition IN ('CONFIRMED', 'FALSE POSITIVE')
        AND koi_period IS NOT NULL
        AND koi_depth IS NOT NULL
        """
        response = requests.get(url, params={"query": query, "format": "csv"}, timeout=120)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        os.makedirs("data/catalog", exist_ok=True)
        df.to_csv(catalog_cache, index=False)
        print(f"[OK] Catalogue telecharge : {len(df)} entrees.")
        return df
    except Exception as e:
        print(f"[FATAL] {e}")
        sys.exit(1)


def select_targets(catalog_df, n_per_class):
    """Selection equilibree par classe."""
    catalog_unique = catalog_df.drop_duplicates(subset='kepid', keep='first')
    
    confirmed = catalog_unique[catalog_unique['koi_disposition'] == 'CONFIRMED']
    false_pos = catalog_unique[catalog_unique['koi_disposition'] == 'FALSE POSITIVE']
    
    print(f"   Disponibles : {len(confirmed)} CONFIRMED, {len(false_pos)} FALSE POSITIVE")
    
    if 'koi_kepmag' in confirmed.columns:
        confirmed_sorted = confirmed.sort_values('koi_kepmag')
        false_pos_sorted = false_pos.sort_values('koi_kepmag')
        step_c = max(1, len(confirmed_sorted) // n_per_class)
        step_f = max(1, len(false_pos_sorted) // n_per_class)
        selected_confirmed = confirmed_sorted.iloc[::step_c].head(n_per_class)
        selected_false_pos = false_pos_sorted.iloc[::step_f].head(n_per_class)
    else:
        selected_confirmed = confirmed.sample(n=min(n_per_class, len(confirmed)), random_state=42)
        selected_false_pos = false_pos.sample(n=min(n_per_class, len(false_pos)), random_state=42)
    
    selected_confirmed = selected_confirmed.copy()
    selected_false_pos = selected_false_pos.copy()
    selected_confirmed['label'] = 1
    selected_false_pos['label'] = 0
    
    targets = pd.concat([selected_confirmed, selected_false_pos]).sample(frac=1, random_state=42)
    print(f"   Selection : {sum(targets['label']==1)} planetes + {sum(targets['label']==0)} FP = {len(targets)} total")
    
    return targets


def process_single_target(row):
    """
    Pipeline par etoile :
    1. Telechargement (cache Lightkurve)
    2. Clean + Flatten
    3. Fold sur la periode du catalogue NASA
    4. Extraction features TSFRESH + scientifiques sur la courbe FOLDEE
    """
    kepid = int(row['kepid'])
    target_name = f"KIC {kepid}"
    catalog_period = row.get('koi_period', None)
    label = row['label']
    
    try:
        # 1. Acquisition (en cache normalement)
        lc_raw = fetch_lightcurve(target_name, mission="Kepler")
        if lc_raw is None:
            return None
        
        # 2. Clean + Flatten
        lc_clean = clean_and_flatten(lc_raw, quality="fast")
        if lc_clean is None or len(lc_clean) < 100:
            return None
        
        # 3. Fold sur la periode du catalogue
        if catalog_period and not np.isnan(catalog_period) and catalog_period > 0:
            period = catalog_period
        else:
            return None  # Pas de periode connue = on skip
        
        lc_folded = fold_lightcurve(lc_clean, period=period)
        if lc_folded is None or len(lc_folded) < 50:
            return None
        
        # 4. Features sur la courbe FOLDEE
        features_df = run_feature_extraction(lc_folded, target_name)
        if features_df is None:
            return None
        
        # 5. Metadonnees du catalogue
        features_df['catalog_period'] = catalog_period
        features_df['catalog_depth_ppm'] = row.get('koi_depth', np.nan)
        features_df['catalog_duration_hr'] = row.get('koi_duration', np.nan)
        features_df['catalog_planet_radius'] = row.get('koi_prad', np.nan)
        features_df['catalog_star_temp'] = row.get('koi_steff', np.nan)
        features_df['catalog_star_radius'] = row.get('koi_srad', np.nan)
        features_df['catalog_kepmag'] = row.get('koi_kepmag', np.nan)
        features_df['kepid'] = kepid
        features_df['target_label'] = label
        
        return features_df
    
    except Exception:
        return None


def build_dataset(targets_df):
    """Boucle principale."""
    print(f"\n[3/5] Construction du dataset ({len(targets_df)} cibles)...\n")
    
    all_features = []
    success = 0
    fail = 0
    start_time = time.time()
    
    for i, (idx, row) in enumerate(targets_df.iterrows()):
        kepid = int(row['kepid'])
        label_str = "PLANET" if row['label'] == 1 else "FP"
        
        if i > 0:
            elapsed = time.time() - start_time
            eta = (elapsed / i) * (len(targets_df) - i) / 60
            eta_str = f"ETA: {eta:.0f}min"
        else:
            eta_str = "ETA: calcul..."
        
        print(f"   [{i+1}/{len(targets_df)}] KIC {kepid} ({label_str}) ... ", end="", flush=True)
        
        result = process_single_target(row)
        
        if result is not None:
            all_features.append(result)
            success += 1
            print(f"OK ({eta_str})")
        else:
            fail += 1
            print(f"SKIP ({eta_str})")
    
    elapsed_total = (time.time() - start_time) / 60
    print(f"\n   Termine en {elapsed_total:.1f} minutes. Succes : {success} | Echecs : {fail}")
    
    if not all_features:
        return None
    
    return pd.concat(all_features, ignore_index=True)


def split_and_save(df, test_ratio=0.2):
    """Split stratifie et sauvegarde."""
    from sklearn.model_selection import train_test_split
    
    y = df['target_label']
    df_train, df_test = train_test_split(df, test_size=test_ratio, random_state=42, stratify=y)
    
    print(f"   Train : {len(df_train)} ({sum(df_train['target_label']==1)} planetes, {sum(df_train['target_label']==0)} FP)")
    print(f"   Test  : {len(df_test)} ({sum(df_test['target_label']==1)} planetes, {sum(df_test['target_label']==0)} FP)")
    
    os.makedirs("data/processed", exist_ok=True)
    df_train.to_csv("data/processed/training_dataset.csv", index=False)
    df_test.to_csv("data/processed/test_dataset.csv", index=False)
    
    import json
    metadata = {
        "total_samples": len(df),
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "n_features": len([c for c in df.columns if c != 'target_label']),
        "unique_stars": int(df['kepid'].nunique()),
        "source": "NASA Kepler KOI Catalog",
        "method": "Features on FOLDED light curves (period from catalog)",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   Fichiers sauvegardes dans data/processed/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=400)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    args = parser.parse_args()
    
    print("=" * 60)
    print("  DATASET V3 - Features sur courbes FOLDEES (sans BLS)")
    print("=" * 60)
    
    print("\n[1/5] Chargement du catalogue...")
    catalog = load_kepler_catalog()
    
    print(f"\n[2/5] Selection des cibles ({args.total // 2} par classe)...")
    targets = select_targets(catalog, args.total // 2)
    
    df = build_dataset(targets)
    if df is None:
        print("Echec.")
        return
    
    print(f"\n[4/5] Split train/test...")
    split_and_save(df, args.test_ratio)
    
    print(f"\n[5/5] Termine. Lancez 02_train_model_v2.py")


if __name__ == "__main__":
    main()