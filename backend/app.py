from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import numpy as np
import xgboost as xgb

# Import des modules locaux (Structure numérotée)
from src.p01_acquisition import fetch_lightcurve
from src.p02_preprocessing import clean_and_flatten, fold_lightcurve, get_period_hint
from src.p04_features import run_feature_extraction

app = Flask(__name__)
CORS(app) # Autorise le dashboard React à communiquer avec Flask

# --- CONFIGURATION & CHARGEMENT ---
MODEL_PATH = "models/exoplanet_model.json"
FEAT_PATH = "models/selected_features.json"

model = None
selected_features = []

def load_resources():
    """Charge le modèle et la liste des caractéristiques au démarrage."""
    global model, selected_features
    
    if os.path.exists(MODEL_PATH):
        try:
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            print("✅ IA chargée et opérationnelle.")
        except Exception as e:
            print(f"⚠️ Erreur chargement modèle : {e}")
            
    if os.path.exists(FEAT_PATH):
        try:
            with open(FEAT_PATH, "r") as f:
                selected_features = json.load(f)
            print(f"✅ Liste des caractéristiques chargée ({len(selected_features)} features).")
        except Exception as e:
            print(f"⚠️ Erreur chargement des features : {e}")

# Initialisation
load_resources()

@app.route('/api/status', methods=['GET'])
def get_status():
    """Vérifie si le backend et l'IA sont prêts."""
    return jsonify({
        "status": "online",
        "ai_loaded": model is not None,
        "features_sync": len(selected_features) > 0,
        "dataset_ready": os.path.exists("data/processed/training_dataset.csv")
    })

@app.route('/api/analyze', methods=['GET'])
def analyze_target():
    """Endpoint principal pour le dashboard React."""
    target_id = request.args.get('id', 'Kepler-10')
    print(f"🔍 Analyse demandée pour : {target_id}")
    
    try:
        # 1. Acquisition
        mission = "TESS" if any(x in target_id for x in ["TIC", "TOI", "Pi", "WASP"]) else "Kepler"
        lc_raw = fetch_lightcurve(target_id, mission=mission)
        
        if lc_raw is None:
            return jsonify({"error": "Cible introuvable dans les archives NASA"}), 404
            
        # 2. Prétraitement complet
        lc_clean = clean_and_flatten(lc_raw, quality="fast")
        
        # 3. Détection de période automatique (BLS)
        period = get_period_hint(lc_clean)
        lc_folded = fold_lightcurve(lc_clean, period=period)
        
        # 4. Prédiction via XGBoost
        score = 0.5 # Neutre par défaut
        if model and selected_features:
            # Extraction des features (TSFRESH + Manuel)
            features_df = run_feature_extraction(lc_clean, target_id)
            
            # Alignement strict avec les colonnes du modèle
            # On ne garde que les colonnes sauvegardées lors de l'entraînement
            if features_df is not None:
                input_data = features_df[selected_features].fillna(0)
                score = float(model.predict_proba(input_data)[0][1])

        # 5. Formatage des données pour le graphique (Recharts)
        step = max(1, len(lc_folded) // 1000)
        chart_data = [
            {"time": round(float(t), 4), "flux": round(float(f), 6)} 
            for t, f in zip(lc_folded.time.value[::step], lc_folded.flux.value[::step])
        ]

        return jsonify({
            "target": target_id,
            "mission": mission,
            "score": score,
            "period": round(period, 4),
            "points_count": len(lc_raw),
            "data": chart_data
        })

    except Exception as e:
        print(f"❌ Erreur API : {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Récupère les informations stellaires simulées."""
    target_id = request.args.get('id', 'Kepler-10')
    return jsonify({
        "target": target_id,
        "star_type": "G-type",
        "distance": "560 ly",
        "estimated_radius": "1.05 R_sun"
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)