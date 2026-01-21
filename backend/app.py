from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np

# Import des modules locaux
from src.acquisition import fetch_lightcurve
from src.preprocessing import clean_lightcurve, fold_lightcurve
from src.features import run_feature_extraction
import xgboost as xgb

app = Flask(__name__)
CORS(app) # Autorise le dashboard React (port 3000) à communiquer avec Flask (port 5000)

# --- CONFIGURATION & CHARGEMENT ---
MODEL_PATH = "models/exoplanet_model.json"
model = None

# Dictionnaire de secours pour les périodes des cibles de test classiques
KNOWN_PERIODS = {
    "Kepler-10": 0.837495,
    "Kepler-90": 14.4491, # Kepler-90i
    "Pi Mensae": 6.268,
    "TOI-700": 37.42,
    "WASP-18": 0.9414
}

def load_ai_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            print("✅ IA chargée et opérationnelle.")
        except Exception as e:
            print(f"⚠️ Erreur chargement modèle : {e}")

# Chargement au démarrage
load_ai_model()

@app.route('/api/status', methods=['GET'])
def get_status():
    """Vérifie si le backend et l'IA sont prêts."""
    return jsonify({
        "status": "online",
        "ai_loaded": model is not None,
        "dataset_ready": os.path.exists("data/processed/training_dataset.csv")
    })

@app.route('/api/analyze', methods=['GET'])
def analyze_target():
    """Endpoint principal pour le dashboard React."""
    target_id = request.args.get('id', 'Kepler-10')
    print(f"🔍 Analyse demandée pour : {target_id}")
    
    try:
        # 1. Acquisition
        # On essaie de détecter si c'est du TESS ou du Kepler via le nom
        mission = "TESS" if ("TIC" in target_id or "TOI" in target_id or "Pi" in target_id) else "Kepler"
        author = "SPOC" if mission == "TESS" else "Kepler"
        
        lc_raw = fetch_lightcurve(target_id, mission=mission, author=author)
        if lc_raw is None:
            return jsonify({"error": "Cible introuvable dans les archives NASA"}), 404
            
        # 2. Nettoyage rapide pour l'affichage web
        lc_clean = clean_lightcurve(lc_raw, quality="fast").remove_nans()
        
        # 3. Récupération de la période (Auto ou Dictionnaire)
        period = KNOWN_PERIODS.get(target_id, 1.0) # 1.0 par défaut si inconnu
        lc_folded = fold_lightcurve(lc_clean, period=period)
        
        # 4. Formatage des données pour Recharts (JS)
        # On limite à 1000 points pour garder une interface fluide
        step = max(1, len(lc_folded) // 1000)
        chart_data = [
            {"time": round(float(t), 4), "flux": round(float(f), 6)} 
            for t, f in zip(lc_folded.time.value[::step], lc_folded.flux.value[::step])
        ]

        # 5. Calcul du score de confiance via XGBoost
        score = 0.0
        if model:
            features = run_feature_extraction(lc_clean, target_id)
            # Nettoyage des features pour le modèle
            features_numeric = features.select_dtypes(include=[np.number]).fillna(0)
            score = float(model.predict_proba(features_numeric)[0][1])
        else:
            print("⚠️ Modèle non chargé. Score simulé.")
            score = 0.5 # Neutre si pas de modèle

        return jsonify({
            "target": target_id,
            "mission": mission,
            "score": score,
            "period": period,
            "points_count": len(lc_raw),
            "data": chart_data
        })

    except Exception as e:
        print(f"❌ Erreur API : {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/metadata', methods=['GET'])
def get_metadata():
    """Récupère les informations stellaires (Optionnel pour le dashboard)."""
    target_id = request.args.get('id', 'Kepler-10')
    # Simulation de métadonnées (À terme, on peut interroger astroquery)
    return jsonify({
        "target": target_id,
        "star_type": "G-type",
        "distance": "560 ly",
        "estimated_radius": "1.05 R_sun"
    })

if __name__ == "__main__":
    # On écoute sur toutes les interfaces pour faciliter les tests
    app.run(debug=True, host='0.0.0.0', port=5001)