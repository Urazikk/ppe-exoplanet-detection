"""
=============================================================================
GENERATEUR DE DATASET V2 - Basé sur le catalogue NASA Kepler TCE
=============================================================================
Remplace l'ancien 01_generate_dataset.py qui n'utilisait que 28 étoiles.

Ce script :
1. Télécharge le catalogue Kepler TCE (Threshold Crossing Events) depuis la NASA
2. Sélectionne N étoiles CONFIRMÉES et N FALSE POSITIVES (équilibré)
3. Télécharge chaque courbe de lumière via Lightkurve
4. Prétraite (clean + flatten + fold sur la période connue du catalogue)
5. Extrait les features TSFRESH + features scientifiques manuelles
6. Sauvegarde train/test en CSV prêts pour l'entraînement

Usage : python 01_generate_dataset_v2.py --total 600 --test_ratio 0.2
Temps estimé : ~4-8h pour 600 étoiles (une nuit)
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# Suppression des warnings astropy/lightkurve pour la lisibilité
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import des modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.p01_acquisition import fetch_lightcurve
from src.p02_preprocessing import clean_and_flatten
from src.p04_features import run_feature_extraction


# =============================================================================
# ÉTAPE 1 : Récupération du catalogue NASA Kepler TCE
# =============================================================================

def load_kepler_catalog():
    """
    Charge le catalogue d'exoplanètes confirmées et de faux positifs
    depuis la NASA Exoplanet Archive via l'API TAP.
    
    Retourne un DataFrame avec colonnes :
    - kepid : identifiant KIC de l'étoile
    - koi_disposition : CONFIRMED ou FALSE POSITIVE
    - koi_period : période orbitale en jours
    - koi_depth : profondeur du transit en ppm
    - koi_duration : durée du transit en heures
    - koi_prad : rayon estimé de la planète (en rayons terrestres)
    - koi_steff : température effective de l'étoile (K)
    - koi_srad : rayon stellaire (en rayons solaires)
    """
    print("[1/5] Téléchargement du catalogue NASA Kepler KOI...")
    
    catalog_cache = "data/catalog/kepler_koi_catalog.csv"
    
    # Cache local pour ne pas retélécharger à chaque fois
    if os.path.exists(catalog_cache):
        print("   -> Catalogue trouvé en cache local.")
        df = pd.read_csv(catalog_cache)
        print(f"   -> {len(df)} entrées chargées.")
        return df
    
    try:
        # Méthode 1 : API TAP de la NASA Exoplanet Archive
        import requests
        
        url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        query = """
        SELECT kepid, koi_disposition, koi_period, koi_depth, koi_duration,
               koi_prad, koi_steff, koi_srad, koi_kepmag
        FROM koi
        WHERE koi_disposition IN ('CONFIRMED', 'FALSE POSITIVE')
        AND koi_period IS NOT NULL
        AND koi_depth IS NOT NULL
        """
        
        params = {
            "query": query,
            "format": "csv"
        }
        
        print("   -> Requête à l'API NASA Exoplanet Archive...")
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        
        # Parse le CSV
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        # Sauvegarde en cache
        os.makedirs("data/catalog", exist_ok=True)
        df.to_csv(catalog_cache, index=False)
        
        print(f"   -> {len(df)} entrées récupérées et mises en cache.")
        return df
        
    except Exception as e:
        print(f"   [!] Erreur API NASA : {e}")
        print("   -> Tentative via astroquery...")
        
        try:
            from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
            
            table = NasaExoplanetArchive.query_criteria(
                table="koi",
                select="kepid,koi_disposition,koi_period,koi_depth,koi_duration,koi_prad,koi_steff,koi_srad,koi_kepmag",
                where="koi_disposition in ('CONFIRMED','FALSE POSITIVE') and koi_period is not null"
            )
            df = table.to_pandas()
            
            os.makedirs("data/catalog", exist_ok=True)
            df.to_csv(catalog_cache, index=False)
            
            print(f"   -> {len(df)} entrées récupérées via astroquery.")
            return df
            
        except Exception as e2:
            print(f"   [FATAL] Impossible de charger le catalogue : {e2}")
            sys.exit(1)


# =============================================================================
# ÉTAPE 2 : Sélection équilibrée des cibles
# =============================================================================

def select_targets(catalog_df, n_per_class):
    """
    Sélectionne n_per_class étoiles CONFIRMED et n_per_class FALSE POSITIVE.
    
    Stratégie de sélection :
    - On prend des étoiles avec des magnitudes variées (pas que les plus brillantes)
    - On déduplique par kepid (une étoile peut avoir plusieurs KOI)
    - On mélange aléatoirement pour éviter les biais
    """
    print(f"\n[2/5] Sélection de {n_per_class} cibles par classe...")
    
    # Déduplications : une seule entrée par étoile (on garde la première KOI)
    catalog_unique = catalog_df.drop_duplicates(subset='kepid', keep='first')
    
    confirmed = catalog_unique[catalog_unique['koi_disposition'] == 'CONFIRMED']
    false_pos = catalog_unique[catalog_unique['koi_disposition'] == 'FALSE POSITIVE']
    
    print(f"   -> Étoiles disponibles : {len(confirmed)} CONFIRMED, {len(false_pos)} FALSE POSITIVE")
    
    # Échantillonnage stratifié par magnitude (pour avoir des étoiles de luminosités variées)
    if 'koi_kepmag' in confirmed.columns:
        confirmed_sorted = confirmed.sort_values('koi_kepmag')
        false_pos_sorted = false_pos.sort_values('koi_kepmag')
        
        # On prend de manière uniforme dans la distribution de magnitude
        step_c = max(1, len(confirmed_sorted) // n_per_class)
        step_f = max(1, len(false_pos_sorted) // n_per_class)
        
        selected_confirmed = confirmed_sorted.iloc[::step_c].head(n_per_class)
        selected_false_pos = false_pos_sorted.iloc[::step_f].head(n_per_class)
    else:
        selected_confirmed = confirmed.sample(n=min(n_per_class, len(confirmed)), random_state=42)
        selected_false_pos = false_pos.sample(n=min(n_per_class, len(false_pos)), random_state=42)
    
    # Attribution des labels : 1 = planète confirmée, 0 = faux positif
    selected_confirmed = selected_confirmed.copy()
    selected_false_pos = selected_false_pos.copy()
    selected_confirmed['label'] = 1
    selected_false_pos['label'] = 0
    
    targets = pd.concat([selected_confirmed, selected_false_pos]).sample(frac=1, random_state=42)
    
    print(f"   -> Sélection finale : {len(targets)} cibles ({sum(targets['label']==1)} planètes, {sum(targets['label']==0)} faux positifs)")
    
    return targets


# =============================================================================
# ÉTAPE 3 : Téléchargement et extraction des features
# =============================================================================

def process_single_target(row, index, total):
    """
    Pipeline complet pour une seule étoile :
    1. Téléchargement via Lightkurve
    2. Nettoyage + Flattening
    3. Folding sur la période connue du catalogue
    4. Extraction TSFRESH + features scientifiques
    5. Ajout des métadonnées du catalogue
    
    Retourne un dict de features ou None si échec.
    """
    kepid = int(row['kepid'])
    target_name = f"KIC {kepid}"
    period = row.get('koi_period', None)
    label = row['label']
    
    try:
        # 1. Acquisition
        lc_raw = fetch_lightcurve(target_name, mission="Kepler")
        if lc_raw is None:
            return None
        
        # 2. Prétraitement
        lc_clean = clean_and_flatten(lc_raw, quality="fast")
        if lc_clean is None or len(lc_clean) < 100:
            return None
        
        # 3. Extraction des features (sur la courbe nettoyée, pas la foldée)
        features_df = run_feature_extraction(lc_clean, target_name)
        if features_df is None:
            return None
        
        # 4. Ajout des métadonnées du catalogue NASA (vraies données scientifiques)
        features_df['catalog_period'] = period
        features_df['catalog_depth_ppm'] = row.get('koi_depth', np.nan)
        features_df['catalog_duration_hr'] = row.get('koi_duration', np.nan)
        features_df['catalog_planet_radius'] = row.get('koi_prad', np.nan)
        features_df['catalog_star_temp'] = row.get('koi_steff', np.nan)
        features_df['catalog_star_radius'] = row.get('koi_srad', np.nan)
        features_df['catalog_kepmag'] = row.get('koi_kepmag', np.nan)
        features_df['kepid'] = kepid
        features_df['target_label'] = label
        
        return features_df
        
    except Exception as e:
        print(f"      [!] Erreur pour {target_name}: {e}")
        return None


def build_dataset(targets_df):
    """
    Boucle principale de construction du dataset.
    Traite chaque étoile séquentiellement avec suivi de progression.
    """
    print(f"\n[3/5] Construction du dataset ({len(targets_df)} cibles)...")
    print("   Ceci peut prendre plusieurs heures. Progression :\n")
    
    all_features = []
    success_count = 0
    fail_count = 0
    start_time = time.time()
    
    for i, (idx, row) in enumerate(targets_df.iterrows()):
        kepid = int(row['kepid'])
        label_str = "PLANET" if row['label'] == 1 else "FP"
        
        # Estimation du temps restant
        if i > 0:
            elapsed = time.time() - start_time
            avg_per_target = elapsed / i
            remaining = avg_per_target * (len(targets_df) - i)
            eta_str = f"ETA: {remaining/60:.0f}min"
        else:
            eta_str = "ETA: calcul..."
        
        print(f"   [{i+1}/{len(targets_df)}] KIC {kepid} ({label_str}) ... ", end="", flush=True)
        
        result = process_single_target(row, i, len(targets_df))
        
        if result is not None:
            all_features.append(result)
            success_count += 1
            print(f"OK ({eta_str})")
        else:
            fail_count += 1
            print(f"SKIP ({eta_str})")
    
    elapsed_total = (time.time() - start_time) / 60
    print(f"\n   Terminé en {elapsed_total:.1f} minutes.")
    print(f"   Succès : {success_count} | Échecs : {fail_count}")
    
    if not all_features:
        print("   [FATAL] Aucune feature extraite. Vérifiez la connexion réseau.")
        return None
    
    return pd.concat(all_features, ignore_index=True)


# =============================================================================
# ÉTAPE 4 : Split train/test et sauvegarde
# =============================================================================

def split_and_save(df, test_ratio=0.2):
    """
    Split stratifié train/test et sauvegarde en CSV.
    """
    from sklearn.model_selection import train_test_split
    
    print(f"\n[4/5] Split train/test ({int((1-test_ratio)*100)}/{int(test_ratio*100)})...")
    
    y = df['target_label']
    
    df_train, df_test = train_test_split(
        df, test_size=test_ratio, random_state=42, stratify=y
    )
    
    # Stats
    print(f"   Train : {len(df_train)} ({sum(df_train['target_label']==1)} planètes, {sum(df_train['target_label']==0)} FP)")
    print(f"   Test  : {len(df_test)} ({sum(df_test['target_label']==1)} planètes, {sum(df_test['target_label']==0)} FP)")
    
    # Sauvegarde
    os.makedirs("data/processed", exist_ok=True)
    
    df_train.to_csv("data/processed/training_dataset.csv", index=False)
    df_test.to_csv("data/processed/test_dataset.csv", index=False)
    
    print(f"\n[5/5] Fichiers sauvegardés dans data/processed/")
    print(f"   -> training_dataset.csv ({len(df_train)} lignes)")
    print(f"   -> test_dataset.csv ({len(df_test)} lignes)")
    
    # Sauvegarde des métadonnées du dataset pour traçabilité
    metadata = {
        "total_samples": len(df),
        "train_samples": len(df_train),
        "test_samples": len(df_test),
        "n_features": len(df.columns) - 1,
        "positive_ratio": float(y.mean()),
        "unique_stars": int(df['kepid'].nunique()),
        "source": "NASA Kepler KOI Catalog (Exoplanet Archive)",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    import json
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   -> dataset_metadata.json (traçabilité)")
    
    return df_train, df_test


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Génération du dataset exoplanète V2")
    parser.add_argument("--total", type=int, default=600,
                        help="Nombre total d'échantillons cibles (défaut: 600)")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Ratio du test set (défaut: 0.2)")
    args = parser.parse_args()
    
    print("=" * 65)
    print("  GENERATEUR DE DATASET V2 - NASA Kepler KOI Catalog")
    print("=" * 65)
    print(f"  Objectif : {args.total} étoiles réelles ({args.total//2} par classe)")
    print(f"  Source   : NASA Exoplanet Archive (catalogue KOI)")
    print(f"  Split    : {int((1-args.test_ratio)*100)}% train / {int(args.test_ratio*100)}% test")
    print("=" * 65)
    
    # 1. Charger le catalogue
    catalog = load_kepler_catalog()
    
    # 2. Sélectionner les cibles
    n_per_class = args.total // 2
    targets = select_targets(catalog, n_per_class)
    
    # 3. Construire le dataset
    df_features = build_dataset(targets)
    
    if df_features is None:
        print("\nÉchec de la génération.")
        return
    
    # 4. Split et sauvegarde
    split_and_save(df_features, args.test_ratio)
    
    print("\n" + "=" * 65)
    print("  DATASET PRÊT - Lancez 02_train_model.py pour entraîner l'IA")
    print("=" * 65)


if __name__ == "__main__":
    main()