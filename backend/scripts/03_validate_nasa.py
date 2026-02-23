"""
Validation croisee avec le catalogue NASA Exoplanet Archive.
Compare les predictions du modele avec les exoplanetes confirmees.
"""
import json
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Catalogue de verite terrain : etoiles connues avec leur statut reel
# Source : NASA Exoplanet Archive (https://exoplanetarchive.ipac.caltech.edu/)
NASA_CATALOG = {
    # Exoplanetes CONFIRMEES
    "Kepler-10":    {"has_planet": True,  "planet_name": "Kepler-10 b",   "period_days": 0.837,  "method": "Transit"},
    "Kepler-22":    {"has_planet": True,  "planet_name": "Kepler-22 b",   "period_days": 289.86, "method": "Transit"},
    "Kepler-90":    {"has_planet": True,  "planet_name": "Kepler-90 b-h", "period_days": 7.008,  "method": "Transit"},
    "Kepler-62":    {"has_planet": True,  "planet_name": "Kepler-62 e/f", "period_days": 122.38, "method": "Transit"},
    "Kepler-186":   {"has_planet": True,  "planet_name": "Kepler-186 f",  "period_days": 129.94, "method": "Transit"},
    "Kepler-452":   {"has_planet": True,  "planet_name": "Kepler-452 b",  "period_days": 384.84, "method": "Transit"},
    "Kepler-442":   {"has_planet": True,  "planet_name": "Kepler-442 b",  "period_days": 112.30, "method": "Transit"},
    "Kepler-296":   {"has_planet": True,  "planet_name": "Kepler-296 e/f","period_days": 34.14,  "method": "Transit"},
    "Kepler-11":    {"has_planet": True,  "planet_name": "Kepler-11 b-g", "period_days": 10.30,  "method": "Transit"},
    "Kepler-20":    {"has_planet": True,  "planet_name": "Kepler-20 b-f", "period_days": 3.70,   "method": "Transit"},
    "Kepler-37":    {"has_planet": True,  "planet_name": "Kepler-37 b",   "period_days": 13.37,  "method": "Transit"},
    "Kepler-18":    {"has_planet": True,  "planet_name": "Kepler-18 b-d", "period_days": 3.50,   "method": "Transit"},

    # Etoiles SANS planete confirmee (faux positifs connus, binaires a eclipses, etc.)
    "KIC 8462852":  {"has_planet": False, "planet_name": None, "period_days": None, "method": "Tabby's Star - dimming non planetaire"},
    "KIC 11442793": {"has_planet": False, "planet_name": None, "period_days": None, "method": "Binaire a eclipses"},
    "KIC 9832227":  {"has_planet": False, "planet_name": None, "period_days": None, "method": "Binaire de contact"},
    "KIC 3427720":  {"has_planet": False, "planet_name": None, "period_days": None, "method": "Variable pulsante"},
    "KIC 10001167": {"has_planet": False, "planet_name": None, "period_days": None, "method": "Bruit stellaire"},
    "KIC 11853905": {"has_planet": False, "planet_name": None, "period_days": None, "method": "Faux positif Kepler"},
}


def validate_with_nasa(model_predictions):
    """
    Compare les predictions du modele avec le catalogue NASA.

    Args:
        model_predictions: dict {target_id: score} (score entre 0 et 1)

    Returns:
        dict avec les resultats de validation
    """
    results = []
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for target_id, pred_score in model_predictions.items():
        if target_id not in NASA_CATALOG:
            continue

        truth = NASA_CATALOG[target_id]
        predicted_planet = pred_score > 0.5
        actual_planet = truth["has_planet"]

        correct = predicted_planet == actual_planet

        if actual_planet and predicted_planet:
            true_positives += 1
        elif not actual_planet and not predicted_planet:
            true_negatives += 1
        elif not actual_planet and predicted_planet:
            false_positives += 1
        elif actual_planet and not predicted_planet:
            false_negatives += 1

        results.append({
            "target": target_id,
            "prediction_score": round(pred_score, 4),
            "predicted": "Planete" if predicted_planet else "Pas de planete",
            "reality_nasa": "Planete confirmee" if actual_planet else "Pas de planete",
            "correct": correct,
            "details": truth.get("planet_name") or truth.get("method", "")
        })

    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = correct_count / total if total > 0 else 0

    validation = {
        "total_tested": total,
        "correct": correct_count,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
        "details": results
    }

    return validation


def run_validation():
    """
    Lance la validation complete en analysant chaque etoile du catalogue.
    Necessite que le backend soit fonctionnel.
    """
    import xgboost as xgb
    from src.p01_acquisition import fetch_lightcurve
    from src.p02_preprocessing import clean_and_flatten, fold_lightcurve, get_period_hint
    from src.p04_features import run_feature_extraction

    model_path = "models/exoplanet_model.json"
    feat_path = "models/selected_features.json"

    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        print("Erreur : modele ou features introuvables. Lancez d'abord l'entrainement.")
        return

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    with open(feat_path, "r") as f:
        selected_features = json.load(f)

    print(f"Modele charge ({len(selected_features)} features)")
    print(f"Validation sur {len(NASA_CATALOG)} etoiles du catalogue NASA\n")

    predictions = {}

    for i, (target_id, truth) in enumerate(NASA_CATALOG.items()):
        print(f"[{i+1}/{len(NASA_CATALOG)}] {target_id}...", end=" ")

        try:
            mission = "Kepler"
            lc_raw = fetch_lightcurve(target_id, mission=mission)

            if lc_raw is None:
                print("SKIP (donnees introuvables)")
                continue

            lc_clean = clean_and_flatten(lc_raw, quality="fast")
            if lc_clean is None:
                print("SKIP (preprocessing echoue)")
                continue

            features_df = run_feature_extraction(lc_clean, target_id)
            if features_df is None:
                print("SKIP (extraction features echouee)")
                continue

            input_data = features_df.reindex(columns=selected_features, fill_value=0).fillna(0)
            score = float(model.predict_proba(input_data)[0][1])
            predictions[target_id] = score

            symbol = "OK" if (score > 0.5) == truth["has_planet"] else "ERREUR"
            print(f"score={score:.2%} -> {symbol}")

        except Exception as e:
            print(f"ERREUR ({str(e)[:50]})")

    print(f"\n{'='*60}")
    print("RESULTATS DE VALIDATION NASA")
    print(f"{'='*60}\n")

    validation = validate_with_nasa(predictions)

    print(f"Etoiles testees  : {validation['total_tested']}")
    print(f"Predictions justes : {validation['correct']}/{validation['total_tested']}")
    print(f"Accuracy          : {validation['accuracy']:.2%}")
    print(f"Precision         : {validation['precision']:.2%}")
    print(f"Recall            : {validation['recall']:.2%}")
    print(f"F1-Score          : {validation['f1_score']:.2%}")

    cm = validation["confusion_matrix"]
    print(f"\nMatrice de Confusion :")
    print(f"  Vrais Positifs  (planete correcte)  : {cm['true_positives']}")
    print(f"  Vrais Negatifs  (bruit correct)     : {cm['true_negatives']}")
    print(f"  Faux Positifs   (fausse detection)  : {cm['false_positives']}")
    print(f"  Faux Negatifs   (planete ratee)     : {cm['false_negatives']}")

    print(f"\nDetail par etoile :")
    for r in validation["details"]:
        mark = "V" if r["correct"] else "X"
        print(f"  [{mark}] {r['target']:20s} score={r['prediction_score']:.2%}  pred={r['predicted']:20s}  NASA={r['reality_nasa']}")

    # Sauvegarde
    os.makedirs("models", exist_ok=True)
    with open("models/nasa_validation.json", "w") as f:
        json.dump(validation, f, indent=2)
    print(f"\nResultats sauvegardes dans models/nasa_validation.json")


if __name__ == "__main__":
    run_validation()