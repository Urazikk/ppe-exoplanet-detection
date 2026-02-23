from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from functools import wraps
import os
import json
import numpy as np
import math
import hashlib
import secrets
import time
import xgboost as xgb

# Import des modules locaux
from src.p01_acquisition import fetch_lightcurve
from src.p02_preprocessing import clean_and_flatten, fold_lightcurve, get_period_hint
from src.p04_features import run_feature_extraction

app = Flask(__name__)
CORS(app)

# ============================================
# CONFIGURATION
# ============================================
MODEL_PATH = "models/exoplanet_model.json"
FEAT_PATH = "models/selected_features.json"
SECRET_KEY = os.environ.get("SECRET_KEY", "exodetect-secret-key-2026")
TOKEN_EXPIRY = 3600 * 8  # 8 heures

model = None
selected_features = []
_analysis_cache = {}

# Base utilisateurs (en production, utiliser une vraie BDD)
USERS = {
    "admin": hashlib.sha256("admin123".encode()).hexdigest(),
    "simon": hashlib.sha256("exoplanet".encode()).hexdigest(),
    "charles": hashlib.sha256("exoplanet".encode()).hexdigest(),
    "oscar": hashlib.sha256("exoplanet".encode()).hexdigest(),
    "mederic": hashlib.sha256("exoplanet".encode()).hexdigest(),
    "kamil": hashlib.sha256("exoplanet".encode()).hexdigest(),
    "mathis": hashlib.sha256("exoplanet".encode()).hexdigest(),
}

# Tokens actifs
_active_tokens = {}


# ============================================
# UTILITAIRES
# ============================================
def safe_float(val):
    """Convertit en float, renvoie None si NaN ou Inf."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def safe_json_response(data):
    """Cree une reponse JSON propre sans NaN."""
    json_str = json.dumps(data, allow_nan=False, default=str)
    return Response(json_str, mimetype='application/json')


def generate_token(username):
    """Genere un token unique pour un utilisateur."""
    token = secrets.token_hex(32)
    _active_tokens[token] = {
        "username": username,
        "created": time.time()
    }
    return token


def verify_token(token):
    """Verifie si un token est valide et non expire."""
    if token not in _active_tokens:
        return None
    info = _active_tokens[token]
    if time.time() - info["created"] > TOKEN_EXPIRY:
        del _active_tokens[token]
        return None
    return info["username"]


def require_auth(f):
    """Decorateur pour proteger les endpoints."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({"error": "Token manquant"}), 401

        token = auth_header.split(' ')[1]
        username = verify_token(token)
        if username is None:
            return jsonify({"error": "Token invalide ou expire"}), 401

        request.username = username
        return f(*args, **kwargs)
    return decorated


# ============================================
# CHARGEMENT MODELE
# ============================================
def load_resources():
    """Charge le modele et la liste des caracteristiques au demarrage."""
    global model, selected_features

    if os.path.exists(MODEL_PATH):
        try:
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            print("IA chargee et operationnelle.")
        except Exception as e:
            print(f"Erreur chargement modele : {e}")

    if os.path.exists(FEAT_PATH):
        try:
            with open(FEAT_PATH, "r") as f:
                selected_features = json.load(f)
            print(f"Liste des caracteristiques chargee ({len(selected_features)} features).")
        except Exception as e:
            print(f"Erreur chargement des features : {e}")


load_resources()


# ============================================
# ENDPOINTS AUTH (non proteges)
# ============================================
@app.route('/api/auth/login', methods=['POST'])
def login():
    """Authentification par identifiant + mot de passe."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Donnees manquantes"}), 400

    username = data.get('username', '').strip().lower()
    password = data.get('password', '')

    if not username or not password:
        return jsonify({"error": "Identifiant et mot de passe requis"}), 400

    password_hash = hashlib.sha256(password.encode()).hexdigest()

    if username not in USERS or USERS[username] != password_hash:
        return jsonify({"error": "Identifiant ou mot de passe incorrect"}), 401

    token = generate_token(username)
    return jsonify({
        "token": token,
        "username": username,
        "expires_in": TOKEN_EXPIRY
    })


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Deconnexion (invalidation du token)."""
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header.split(' ')[1]
        _active_tokens.pop(token, None)
    return jsonify({"status": "deconnecte"})


@app.route('/api/auth/verify', methods=['GET'])
def verify():
    """Verifie si le token actuel est encore valide."""
    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return jsonify({"valid": False}), 401

    token = auth_header.split(' ')[1]
    username = verify_token(token)
    if username is None:
        return jsonify({"valid": False}), 401

    return jsonify({"valid": True, "username": username})


# ============================================
# ENDPOINTS PROTEGES
# ============================================
@app.route('/api/status', methods=['GET'])
@require_auth
def get_status():
    """Verifie si le backend et l'IA sont prets."""
    return jsonify({
        "status": "online",
        "ai_loaded": model is not None,
        "features_sync": len(selected_features) > 0,
        "dataset_ready": os.path.exists("data/processed/training_dataset.csv"),
        "cache_size": len(_analysis_cache),
        "user": request.username
    })


@app.route('/api/analyze', methods=['GET'])
@require_auth
def analyze_target():
    """Analyse une etoile et predit la probabilite d'exoplanete."""
    target_id = request.args.get('id', 'Kepler-10')
    force_refresh = request.args.get('refresh', 'false').lower() == 'true'

    if target_id in _analysis_cache and not force_refresh:
        print(f"Cache hit pour : {target_id}")
        return safe_json_response(_analysis_cache[target_id])

    print(f"Analyse demandee par {request.username} pour : {target_id}")

    try:
        # 1. Acquisition
        mission = "TESS" if any(x in target_id for x in ["TIC", "TOI", "Pi", "WASP"]) else "Kepler"
        lc_raw = fetch_lightcurve(target_id, mission=mission)

        if lc_raw is None:
            return jsonify({"error": "Cible introuvable dans les archives NASA"}), 404

        # 2. Preprocessing
        lc_clean = clean_and_flatten(lc_raw, quality="fast")

        if lc_clean is None:
            return jsonify({"error": "Echec du preprocessing"}), 500

        # 3. Detection de periode (BLS)
        period = get_period_hint(lc_clean)
        lc_folded = fold_lightcurve(lc_clean, period=period)

        # 4. Prediction via XGBoost
        score = 0.5
        top_features_list = []

        if model and selected_features:
            features_df = run_feature_extraction(lc_clean, target_id)

            if features_df is not None:
                input_data = features_df.reindex(columns=selected_features, fill_value=0).fillna(0)
                raw_score = model.predict_proba(input_data)[0][1]
                score = safe_float(raw_score) or 0.5

                importances = model.feature_importances_
                feat_imp = sorted(
                    zip(selected_features, importances),
                    key=lambda x: x[1], reverse=True
                )[:5]
                top_features_list = [
                    {"name": name, "importance": round(float(imp), 4)}
                    for name, imp in feat_imp
                ]

        # 5. Formatage des donnees pour Plotly
        step = max(1, len(lc_folded) // 1000)
        chart_data = []
        for t, f in zip(lc_folded.time.value[::step], lc_folded.flux.value[::step]):
            t_val = safe_float(t)
            f_val = safe_float(f)
            if t_val is not None and f_val is not None:
                chart_data.append({
                    "time": round(t_val, 4),
                    "flux": round(f_val, 6)
                })

        period_val = safe_float(period) or 0.0

        result = {
            "target": target_id,
            "mission": mission,
            "score": score,
            "period": round(period_val, 4),
            "points_count": len(lc_raw),
            "top_features": top_features_list,
            "data": chart_data
        }

        _analysis_cache[target_id] = result
        return safe_json_response(result)

    except Exception as e:
        print(f"Erreur API : {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/metadata', methods=['GET'])
@require_auth
def get_metadata():
    """Recupere les informations stellaires."""
    target_id = request.args.get('id', 'Kepler-10')

    metadata = {
        "target": target_id,
        "star_type": "Inconnu",
        "distance": "Inconnue",
        "estimated_radius": "Inconnu",
        "source": "default"
    }

    try:
        import lightkurve as lk
        mission = "TESS" if any(x in target_id for x in ["TIC", "TOI", "Pi", "WASP"]) else "Kepler"
        author = "SPOC" if mission == "TESS" else "Kepler"
        search = lk.search_lightcurve(target_id, mission=mission, author=author)

        if len(search) > 0:
            table = search.table
            if 'distance' in table.colnames:
                dist = table['distance'][0]
                if dist and not np.isnan(dist):
                    metadata["distance"] = f"{dist:.1f} pc"
            if 'target_name' in table.colnames:
                metadata["official_name"] = str(table['target_name'][0])
            metadata["mission"] = mission
            metadata["nb_observations"] = len(search)
            metadata["source"] = "MAST"
    except Exception as e:
        print(f"Erreur metadata pour {target_id}: {e}")

    return jsonify(metadata)


@app.route('/api/cache/clear', methods=['POST'])
@require_auth
def clear_cache():
    """Vide le cache d'analyses."""
    _analysis_cache.clear()
    return jsonify({"status": "cache cleared"})


@app.route('/api/validate', methods=['GET'])
@require_auth
def validate_target():
    """
    Compare la prediction du modele avec le catalogue NASA Exoplanet Archive.
    """
    target_id = request.args.get('id', 'Kepler-10')

    NASA_CATALOG = {
        "Kepler-10":    {"has_planet": True,  "planet_name": "Kepler-10 b",   "period_days": 0.837},
        "Kepler-22":    {"has_planet": True,  "planet_name": "Kepler-22 b",   "period_days": 289.86},
        "Kepler-90":    {"has_planet": True,  "planet_name": "Kepler-90 b-h", "period_days": 7.008},
        "Kepler-62":    {"has_planet": True,  "planet_name": "Kepler-62 e/f", "period_days": 122.38},
        "Kepler-186":   {"has_planet": True,  "planet_name": "Kepler-186 f",  "period_days": 129.94},
        "Kepler-452":   {"has_planet": True,  "planet_name": "Kepler-452 b",  "period_days": 384.84},
        "Kepler-442":   {"has_planet": True,  "planet_name": "Kepler-442 b",  "period_days": 112.30},
        "Kepler-11":    {"has_planet": True,  "planet_name": "Kepler-11 b-g", "period_days": 10.30},
        "Kepler-20":    {"has_planet": True,  "planet_name": "Kepler-20 b-f", "period_days": 3.70},
        "Kepler-37":    {"has_planet": True,  "planet_name": "Kepler-37 b",   "period_days": 13.37},
        "Kepler-18":    {"has_planet": True,  "planet_name": "Kepler-18 b-d", "period_days": 3.50},
        "KIC 8462852":  {"has_planet": False, "planet_name": None, "period_days": None},
        "KIC 11442793": {"has_planet": False, "planet_name": None, "period_days": None},
        "KIC 9832227":  {"has_planet": False, "planet_name": None, "period_days": None},
        "KIC 3427720":  {"has_planet": False, "planet_name": None, "period_days": None},
        "KIC 10001167": {"has_planet": False, "planet_name": None, "period_days": None},
        "KIC 11853905": {"has_planet": False, "planet_name": None, "period_days": None},
    }

    if target_id not in NASA_CATALOG:
        return jsonify({
            "target": target_id,
            "in_catalog": False,
            "message": "Cette etoile n'est pas dans notre catalogue de validation."
        })

    truth = NASA_CATALOG[target_id]

    # Recuperer le score depuis le cache ou renvoyer un message
    if target_id in _analysis_cache:
        score = _analysis_cache[target_id]["score"]
        predicted_planet = score > 0.5
        actual_planet = truth["has_planet"]

        return jsonify({
            "target": target_id,
            "in_catalog": True,
            "prediction_score": score,
            "predicted_planet": predicted_planet,
            "nasa_confirmed": actual_planet,
            "correct": predicted_planet == actual_planet,
            "planet_name": truth.get("planet_name"),
            "known_period": truth.get("period_days"),
            "method": truth.get("method"),
        })

    return jsonify({
        "target": target_id,
        "in_catalog": True,
        "message": "Analysez d'abord cette etoile avant de valider."
    })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)