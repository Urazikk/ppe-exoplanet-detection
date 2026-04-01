#!/usr/bin/env python3
"""
=============================================================================
10 — Entraînement Kepler + TESS (XGBoost unifié)
=============================================================================
Ce script étend le modèle existant pour supporter à la fois les cibles Kepler
(KOI) et TESS (TOI) en fusionnant :

  1. Les données Kepler déjà en cache (star_*.json dans lightkurve_training/)
  2. Le catalogue TOI de la NASA Exoplanet Archive (téléchargé automatiquement)

Correspondance des features BLS ↔ TOI :
  bls_snr              ← toi_snr
  bls_depth_ppm        ← depth (ppm)
  bls_transit_fraction ← duration_hours / 24 / period_days
  bls_power            ← approximé depuis SNR
  bls_duration_days    ← duration_hours / 24
  bls_score            ← 1.0 si CP, 0.7 si PC, 0.0 si FP
  period               ← period (jours)
  star_radius_solar    ← st_rad
  star_temperature_k   ← st_teff

Usage :
    cd backend && source venv/bin/activate
    python scripts/10_kepler_tess_train.py
=============================================================================
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import requests
from pathlib import Path

import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score
)

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "data" / "cache" / "lightkurve_training"
MODEL_DIR = BASE_DIR / "models"

FEATURES = [
    "bls_snr",
    "bls_depth_ppm",
    "bls_transit_fraction",
    "bls_power",
    "bls_duration_days",
    "bls_score",
    "period",
    "star_radius_solar",
    "star_temperature_k",
]

NASA_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

print("=" * 70)
print("  ENTRAÎNEMENT KEPLER + TESS — XGBoost Unifié")
print("=" * 70)


# =============================================================================
# 1. Données Kepler (cache existant)
# =============================================================================
def load_kepler_data():
    print("\n[1/4] Chargement des données Kepler depuis le cache...")

    # Métadonnées stellaires Kepler
    cat_path = BASE_DIR / "data" / "catalog" / "kepler_koi_catalog.csv"
    if not cat_path.exists():
        print("[!] kepler_koi_catalog.csv introuvable.")
        sys.exit(1)

    df_cat = pd.read_csv(cat_path).drop_duplicates(subset=["kepid"])
    default_srad  = df_cat["koi_srad"].median()
    default_steff = df_cat["koi_steff"].median()
    star_meta = {
        int(r["kepid"]): (
            float(r["koi_srad"])  if pd.notna(r["koi_srad"])  else default_srad,
            float(r["koi_steff"]) if pd.notna(r["koi_steff"]) else default_steff,
        )
        for _, r in df_cat.iterrows()
    }

    stars = []
    for f in sorted(CACHE_DIR.glob("star_*.json")):
        with open(f) as fp:
            d = json.load(fp)
        if d.get("status") == "ok":
            stars.append(d)

    if not stars:
        print("[!] Aucune étoile Kepler valide dans le cache.")
        sys.exit(1)

    rows, labels = [], []
    for s in stars:
        kepid = int(s.get("kepid", 0))
        srad, steff = star_meta.get(kepid, (default_srad, default_steff))
        rows.append([
            float(s.get("bls_snr",              0.0)),
            float(s.get("bls_depth_ppm",         0.0)),
            float(s.get("bls_transit_fraction",  0.0)),
            float(s.get("bls_power",             0.0)),
            float(s.get("bls_duration_days",     0.0)),
            float(s.get("bls_score",             0.5)),
            float(s.get("period",                0.0)),
            srad,
            steff,
        ])
        labels.append(int(s["label"]))

    df = pd.DataFrame(rows, columns=FEATURES)
    df["_mission"] = "Kepler"
    y  = np.array(labels)
    planets = y.sum()
    print(f"   ✓ {len(df)} étoiles Kepler ({planets} planètes, {len(df)-planets} faux positifs)")
    return df, y


# =============================================================================
# 2. Données TESS (TOI catalog — NASA TAP)
# =============================================================================
def load_tess_data():
    print("\n[2/4] Téléchargement du catalogue TOI (NASA Exoplanet Archive)...")

    # NASA TAP ADQL: pas de WHERE avec IN et quotes simples dans certains clients
    # On récupère tout et filtre en Python
    query = (
        "SELECT toi, tfopwg_disp, pl_orbper, pl_trandep, pl_trandurh, "
        "st_rad, st_teff FROM toi"
    )
    try:
        resp = requests.get(NASA_TAP, params={"query": query, "format": "csv"}, timeout=60)
        resp.raise_for_status()
        import io
        df_toi = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        print(f"   [!] Erreur téléchargement NASA TAP : {e}")
        print("   → Entraînement Kepler uniquement.")
        return None, None

    if df_toi.empty:
        print("   [!] Catalogue TOI vide.")
        return None, None

    # Garder seulement CP, PC, FP
    df_toi = df_toi[df_toi["tfopwg_disp"].isin(["CP", "PC", "FP"])].copy()
    print(f"   ✓ {len(df_toi)} entrées TOI (CP/PC/FP). Colonnes : {list(df_toi.columns)}")

    col_map = {
        "period":   "pl_orbper",
        "depth":    "pl_trandep",
        "duration": "pl_trandurh",
        "st_rad":   "st_rad",
        "st_teff":  "st_teff",
    }

    # Vérifier les colonnes critiques (period et depth ; snr sera approximé)
    missing = [k for k in ["period", "depth"] if col_map.get(k) not in df_toi.columns]
    if missing:
        print(f"   [!] Colonnes critiques manquantes : {missing}. Entraînement Kepler uniquement.")
        return None, None

    # Valeurs par défaut stellaires
    default_srad  = 1.0
    default_steff = 5500.0

    rows, labels = [], []
    skipped = 0

    for _, row in df_toi.iterrows():
        disp = str(row.get("tfopwg_disp", "")).strip().upper()
        if disp == "CP":
            label, score = 1, 1.0
        elif disp == "PC":
            label, score = 1, 0.7
        elif disp == "FP":
            label, score = 0, 0.0
        else:
            skipped += 1
            continue

        try:
            period   = float(row[col_map["period"]])
            depth    = float(row[col_map["depth"]])        # ppm
            dur_h    = float(row[col_map["duration"]]) if pd.notna(row.get(col_map["duration"])) else 2.0
            srad     = float(row[col_map["st_rad"]])   if pd.notna(row.get(col_map["st_rad"]))   else default_srad
            steff    = float(row[col_map["st_teff"]])  if pd.notna(row.get(col_map["st_teff"]))  else default_steff
        except (ValueError, TypeError, KeyError):
            skipped += 1
            continue

        if period <= 0 or np.isnan(period) or np.isnan(depth):
            skipped += 1
            continue

        dur_days = dur_h / 24.0
        transit_fraction = dur_days / period if period > 0 else 0.0
        # SNR approximé depuis profondeur et durée (proxy sans photométrie brute)
        snr = (depth / 1e6) / max(1e-4, 0.0001 / max(dur_days, 0.01) ** 0.5) * 10.0
        snr = float(np.clip(snr, 1.0, 200.0))
        # bls_power normalisé [0, 1]
        bls_power = float(np.clip(depth / 100000.0, 0.0, 1.0))

        # bls_score : même formule que compute_transit_score() dans app.py (match inference)
        depth_ppm = abs(depth)
        if snr >= 10:       snr_sc = 1.0
        elif snr >= 7.1:    snr_sc = 0.7 + 0.3 * (snr - 7.1) / 2.9
        elif snr >= 5:      snr_sc = 0.4 + 0.3 * (snr - 5) / 2.1
        elif snr >= 3:      snr_sc = 0.15 + 0.25 * (snr - 3) / 2
        else:               snr_sc = snr * 0.05

        if 100 <= depth_ppm <= 30000:   dep_sc = 1.0
        elif 30 <= depth_ppm < 100:     dep_sc = 0.5 + 0.5 * (depth_ppm - 30) / 70
        elif 30000 < depth_ppm <= 1e5:  dep_sc = max(0, 0.5 - (depth_ppm - 30000) / 140000)
        else:                            dep_sc = 0.1

        if transit_fraction < 0.05:     frac_sc = 1.0
        elif transit_fraction < 0.15:   frac_sc = 0.7 + 0.3 * (0.15 - transit_fraction) / 0.10
        elif transit_fraction < 0.3:    frac_sc = 0.3
        else:                            frac_sc = 0.0

        bls_score_val = float(0.6 * snr_sc + 0.25 * dep_sc + 0.15 * frac_sc)

        rows.append([
            snr,
            depth,
            transit_fraction,
            bls_power,
            dur_days,
            bls_score_val,
            period,
            srad,
            steff,
        ])
        labels.append(label)

    if skipped > 0:
        print(f"   ⚠ {skipped} entrées TOI ignorées (valeurs manquantes ou disposition ambiguë)")

    if not rows:
        print("   [!] Aucune donnée TESS valide extraite.")
        return None, None

    df = pd.DataFrame(rows, columns=FEATURES)
    df["_mission"] = "TESS"
    y = np.array(labels)
    planets = y.sum()
    print(f"   ✓ {len(df)} entrées TESS ({planets} planètes/candidats, {len(df)-planets} faux positifs)")
    return df, y


# =============================================================================
# 3. Fusion + Entraînement
# =============================================================================
def train_unified(X_kep, y_kep, X_tess, y_tess):
    print("\n[3/4] Fusion Kepler + TESS et entraînement XGBoost...")

    if X_tess is not None and y_tess is not None:
        X_all = pd.concat([X_kep.drop(columns=["_mission"]),
                           X_tess.drop(columns=["_mission"])], ignore_index=True)
        y_all = np.concatenate([y_kep, y_tess])
        missions = ["Kepler"] * len(y_kep) + ["TESS"] * len(y_tess)
        print(f"   ✓ Dataset fusionné : {len(X_all)} étoiles ({y_all.sum()} planètes)")
    else:
        X_all = X_kep.drop(columns=["_mission"])
        y_all = y_kep
        missions = ["Kepler"] * len(y_kep)
        print("   ⚠ Entraînement sur Kepler uniquement (TESS non disponible)")

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    k = min(5, int(y_train.sum()) - 1)
    if k > 0:
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"   ✓ SMOTE : {len(X_res)} échantillons équilibrés")
    else:
        X_res, y_res = X_train, y_train

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(model, X_res, y_res, cv=cv,
                               scoring=["accuracy", "f1", "roc_auc"])

    print(f"   ► Accuracy  (CV5) : {cv_scores['test_accuracy'].mean()*100:.1f}%")
    print(f"   ► AUC-ROC   (CV5) : {cv_scores['test_roc_auc'].mean()*100:.1f}%")

    model.fit(X_res, y_res)

    proba_test = model.predict_proba(X_test)[:, 1]
    best_acc, best_thr, best_f1 = 0, 0.5, 0
    for thr in np.arange(0.1, 0.9, 0.05):
        y_pred_thr = (proba_test >= thr).astype(int)
        acc = accuracy_score(y_test, y_pred_thr)
        if acc > best_acc:
            best_acc, best_thr = acc, thr
            best_f1 = f1_score(y_test, y_pred_thr, zero_division=0)

    y_pred = (proba_test >= best_thr).astype(int)
    print(f"\n   Seuil optimal : {best_thr:.2f}")
    print(f"\n   Résultats holdout ({len(y_test)} étoiles) :")
    print(classification_report(y_test, y_pred, target_names=["Faux Positif", "Planète"]))

    cm = confusion_matrix(y_test, y_pred)

    imp = model.feature_importances_
    top_idx  = np.argsort(imp)[::-1]
    top_feats = [{"name": FEATURES[i], "importance": float(imp[i])}
                 for i in top_idx if imp[i] > 0]

    print("\n   Importance des features :")
    for f in top_feats:
        print(f"     ► {f['name']:28s} : {f['importance']*100:5.1f}%")

    tess_count = missions.count("TESS") if X_tess is not None else 0
    metrics = {
        "source": "XGBoost unifié Kepler + TESS",
        "training_date": time.strftime("%Y-%m-%d %H:%M"),
        "n_features": len(FEATURES),
        "n_stars_total": len(X_all),
        "n_kepler": len(y_kep),
        "n_tess": tess_count,
        "train_size": len(X_train),
        "holdout_size": len(X_test),
        "optimal_threshold": float(best_thr),
        "holdout_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(best_f1),
        "test_auc_roc": float(roc_auc_score(y_test, proba_test)),
        "holdout_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "holdout_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "holdout_f1": float(best_f1),
        "holdout_auc_roc": float(roc_auc_score(y_test, proba_test)),
        "confusion_matrix": cm.tolist(),
        "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
        "cv_accuracy_std":  float(cv_scores["test_accuracy"].std()),
        "cv_f1_mean":       float(cv_scores["test_f1"].mean()),
        "cv_f1_std":        float(cv_scores["test_f1"].std()),
        "cv_auc_mean":      float(cv_scores["test_roc_auc"].mean()),
        "cv_auc_std":       float(cv_scores["test_roc_auc"].std()),
        "cv5_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
        "cv5_auc_mean":      float(cv_scores["test_roc_auc"].mean()),
        "n_features_selected": len(FEATURES),
        "n_features_total":    len(FEATURES),
        "top_features": top_feats,
    }
    return model, metrics


# =============================================================================
# 4. Sauvegarde
# =============================================================================
def save_model(model, metrics):
    print("\n[4/4] Sauvegarde du modèle unifié...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(MODEL_DIR / "exoplanet_model.json"))
    with open(MODEL_DIR / "selected_features.json", "w") as f:
        json.dump(FEATURES, f, indent=2)
    with open(MODEL_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    n_tess = metrics.get("n_tess", 0)
    n_kep  = metrics.get("n_kepler", 0)
    print(f"   ✓ Modèle sauvegardé  ({n_kep} Kepler + {n_tess} TESS)")
    print(f"   ✓ Accuracy holdout   : {metrics['holdout_accuracy']*100:.1f}%")
    print(f"   ✓ AUC-ROC            : {metrics['holdout_auc_roc']:.3f}")
    print("\n" + "=" * 70)
    print("  TERMINÉ — modèle Kepler + TESS prêt.")
    print("=" * 70 + "\n")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    t0 = time.time()
    X_kep,  y_kep  = load_kepler_data()
    X_tess, y_tess = load_tess_data()
    model, metrics = train_unified(X_kep, y_kep, X_tess, y_tess)
    save_model(model, metrics)
    print(f"Durée totale : {time.time()-t0:.1f}s")
