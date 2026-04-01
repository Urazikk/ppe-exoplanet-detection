"""
=============================================================================
BACKEND FLASK V2 - API Exoplanet Detection
=============================================================================
Remplace l'ancien app.py.

Endpoints :
  POST /api/auth/register   - Création de compte
  POST /api/auth/login      - Connexion (retourne un JWT)
  GET  /api/status           - État du système (public)
  GET  /api/analyze?id=X     - Analyse complète d'une cible (auth requise)
  GET  /api/metrics          - Métriques du modèle (auth requise)
  GET  /api/catalog/search?q=X - Recherche dans le catalogue (auth requise)

Améliorations par rapport à V1 :
  - Authentification JWT
  - Vraies métadonnées stellaires depuis le catalogue NASA
  - Caractérisation : estimation du rayon planétaire, SNR, période
  - Endpoint métriques pour le dashboard
  - Gestion d'erreurs propre
"""

import os
import sys
import json
import time
import hashlib
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from flask import Flask, request, jsonify, g, Response, stream_with_context
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

# JWT
try:
    import jwt as pyjwt
except ImportError:
    print("[!] PyJWT non installé. Lancez : pip install PyJWT")
    sys.exit(1)

# Modules du projet
from src.p01_acquisition import fetch_lightcurve
from src.p02_preprocessing import clean_and_flatten, fold_lightcurve, get_period_hint
from src.p04_features import run_feature_extraction


# =============================================================================
# Configuration
# =============================================================================

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
    return response


# Clé secrète JWT (en production, utiliser une variable d'environnement)
JWT_SECRET = os.environ.get("JWT_SECRET", "exoplanet-detection-secret-key-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Chemins (Fixé avec des chemins absolus via pathlib)
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = str(BASE_DIR / "models" / "exoplanet_model.json")
FEATURES_PATH = str(BASE_DIR / "models" / "selected_features.json")
METRICS_PATH = str(BASE_DIR / "models" / "model_metrics.json")
CATALOG_PATH = str(BASE_DIR / "data" / "catalog" / "exoplanet_binary_full.csv")
USERS_PATH = str(BASE_DIR / "data" / "users.json")
RESULTS_CACHE_PATH = str(BASE_DIR / "data" / "results_cache.json")
HISTORY_PATH = str(BASE_DIR / "data" / "history.json")


# =============================================================================
# Chargement des ressources
# =============================================================================

model = None
selected_features = []
model_metrics = {}
catalog_df = None
results_cache = {}

# Cache in-memory avec TTL (clé → {"result": ..., "ts": float})
_analysis_cache = {}
CACHE_TTL = 600  # 10 minutes

# Catalog index (lightweight, no flux/time arrays)
_catalog_cache_index = []


def is_finite_number(value):
    """Retourne True pour les nombres finis utilisables en JSON."""
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.isfinite(value)
    return False


def json_safe(value):
    """Convertit recursivement les NaN/inf en None pour produire un JSON valide."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def build_chart_data(lc_folded):
    """Construit les points de courbe en ignorant les valeurs invalides."""
    step = max(1, len(lc_folded) // 800)
    chart_data = []
    for t, f in zip(lc_folded.time.value[::step], lc_folded.flux.value[::step]):
        t_val = float(t)
        f_val = float(f)
        if not (np.isfinite(t_val) and np.isfinite(f_val)):
            continue
        chart_data.append({
            "time": round(t_val, 5),
            "flux": round(f_val, 6)
        })
    return chart_data


def load_results_cache():
    global results_cache
    if os.path.exists(RESULTS_CACHE_PATH):
        with open(RESULTS_CACHE_PATH, "r") as f:
            results_cache = json.load(f)
        print(f"[OK] Cache résultats chargé ({len(results_cache)} entrées).")


def save_results_cache():
    os.makedirs(os.path.dirname(RESULTS_CACHE_PATH), exist_ok=True)
    with open(RESULTS_CACHE_PATH, "w") as f:
        json.dump(results_cache, f)


def load_resources():
    """Charge le modèle, les features, les métriques et le catalogue au démarrage."""
    global model, selected_features, model_metrics, catalog_df
    
    # Modèle XGBoost
    if os.path.exists(MODEL_PATH):
        try:
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            print("[OK] Modèle XGBoost chargé.")
        except Exception as e:
            print(f"[!] Erreur chargement modèle : {e}")
    else:
        print("[!] Modèle introuvable. Lancez 02_train_model_v2.py d'abord.")
    
    # Features sélectionnées
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r") as f:
            selected_features = json.load(f)
        print(f"[OK] {len(selected_features)} features chargées.")
    
    # Métriques
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            model_metrics = json.load(f)
        print("[OK] Métriques du modèle chargées.")
    
    # Catalogue Kepler
    if os.path.exists(CATALOG_PATH):
        catalog_df = pd.read_csv(CATALOG_PATH)
        print(f"[OK] Catalogue NASA chargé ({len(catalog_df)} entrées).")
    else:
        print("[!] Catalogue NASA introuvable. Les métadonnées stellaires seront limitées.")


load_results_cache()

load_resources()


def _build_catalog_index():
    """Builds a lightweight index of all ok-status cache files (no flux/time arrays)."""
    global _catalog_cache_index
    cache_dir = os.path.join(os.path.dirname(__file__), "data", "cache", "lightkurve_training")
    entries = []
    for fname in os.listdir(cache_dir):
        if not fname.startswith("star_") or not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(cache_dir, fname)) as f:
                d = json.load(f)
            if d.get("status") != "ok":
                continue
            entries.append({
                "kepid": d["kepid"],
                "label": d.get("label", 0),
                "n_points": d.get("n_points", 0),
                "bls_snr": round(d["bls_snr"], 3) if d.get("bls_snr") is not None else None,
                "bls_depth_ppm": round(d["bls_depth_ppm"], 1) if d.get("bls_depth_ppm") is not None else None,
                "bls_duration_days": round(d["bls_duration_days"], 4) if d.get("bls_duration_days") is not None else None,
                "bls_score": round(d["bls_score"], 4) if d.get("bls_score") is not None else None,
                "period": round(d["period"], 4) if d.get("period") is not None else None,
            })
        except Exception:
            continue
    _catalog_cache_index = sorted(entries, key=lambda x: x.get("bls_snr") or 0, reverse=True)
    print(f"[Catalog] Index built: {len(_catalog_cache_index)} stars")


_build_catalog_index()


# =============================================================================
# Gestion des utilisateurs (fichier JSON simple)
# =============================================================================

def load_users():
    """Charge la base d'utilisateurs."""
    if os.path.exists(USERS_PATH):
        with open(USERS_PATH, "r") as f:
            return json.load(f)
    return {}


def save_users(users):
    """Sauvegarde la base d'utilisateurs."""
    os.makedirs(os.path.dirname(USERS_PATH), exist_ok=True)
    with open(USERS_PATH, "w") as f:
        json.dump(users, f, indent=2)


def hash_password(password):
    """Hash SHA-256 du mot de passe (en production, utiliser bcrypt)."""
    return hashlib.sha256(password.encode()).hexdigest()


# =============================================================================
# Gestion de l'historique par utilisateur
# =============================================================================

def load_history():
    """Charge l'historique des analyses depuis le fichier JSON."""
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r") as f:
            return json.load(f)
    return {}


def save_history_entry(username, entry):
    """Sauvegarde une entrée d'analyse dans l'historique de l'utilisateur (max 50)."""
    history = load_history()
    if username not in history:
        history[username] = []
    history[username].insert(0, entry)
    history[username] = history[username][:50]
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)


# =============================================================================
# Middleware JWT
# =============================================================================

def token_required(f):
    """Décorateur qui vérifie la présence et la validité du token JWT."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Récupération du token dans le header Authorization
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({"error": "Token manquant. Connectez-vous via /api/auth/login"}), 401
        
        try:
            payload = pyjwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            g.current_user = payload.get("username", "unknown")
        except pyjwt.ExpiredSignatureError:
            return jsonify({"error": "Token expiré. Reconnectez-vous."}), 401
        except pyjwt.InvalidTokenError:
            return jsonify({"error": "Token invalide."}), 401
        
        return f(*args, **kwargs)
    return decorated


# =============================================================================
# Routes : Authentification
# =============================================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Création d'un compte utilisateur."""
    data = request.get_json()
    
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Champs 'username' et 'password' requis."}), 400
    
    username = data['username'].strip()
    password = data['password']
    
    if len(username) < 3:
        return jsonify({"error": "Le nom d'utilisateur doit faire au moins 3 caractères."}), 400
    if len(password) < 6:
        return jsonify({"error": "Le mot de passe doit faire au moins 6 caractères."}), 400
    
    users = load_users()
    
    if username in users:
        return jsonify({"error": "Ce nom d'utilisateur existe déjà."}), 409
    
    users[username] = {
        "password_hash": hash_password(password),
        "created_at": datetime.datetime.utcnow().isoformat(),
        "has_seen_tutorial": False,
        "avatar": "user"
    }
    save_users(users)
    
    return jsonify({"message": f"Compte '{username}' créé avec succès."}), 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    """Connexion et génération du token JWT."""
    data = request.get_json()
    
    if not data or 'username' not in data or 'password' not in data:
        return jsonify({"error": "Champs 'username' et 'password' requis."}), 400
    
    username = data['username'].strip()
    password = data['password']
    
    users = load_users()
    
    if username not in users:
        return jsonify({"error": "Identifiants incorrects."}), 401
    
    if users[username]['password_hash'] != hash_password(password):
        return jsonify({"error": "Identifiants incorrects."}), 401
    
    # Génération du JWT
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.datetime.utcnow()
    }
    
    token = pyjwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    return jsonify({
        "token": token,
        "expires_in": JWT_EXPIRATION_HOURS * 3600,
        "username": username,
        "has_seen_tutorial": users[username].get("has_seen_tutorial", False),
        "avatar": users[username].get("avatar", "user")
    })


@app.route('/api/auth/tutorial_seen', methods=['POST'])
@token_required
def tutorial_seen():
    """Marque le tutoriel interactif comme vu par l'utilisateur."""
    username = g.current_user
    users = load_users()
    if username in users:
        users[username]["has_seen_tutorial"] = True
        save_users(users)
    return jsonify({"ok": True})


@app.route('/api/auth/update_profile', methods=['POST'])
@token_required
def update_profile():
    """Modifie la photo de profil (avatar)."""
    data = request.get_json()
    if not data or 'avatar' not in data:
        return jsonify({"error": "Paramètre 'avatar' requis."}), 400
    username = g.current_user
    users = load_users()
    if username in users:
        users[username]["avatar"] = data["avatar"]
        save_users(users)
    return jsonify({"ok": True, "avatar": data["avatar"]})


@app.route('/api/auth/change_password', methods=['POST'])
@token_required
def change_password():
    """Modifie le mot de passe après vérification de l'ancien."""
    data = request.get_json()
    if not data or 'old_password' not in data or 'new_password' not in data:
        return jsonify({"error": "Champs 'old_password' et 'new_password' requis."}), 400
    username = g.current_user
    users = load_users()
    if username not in users:
        return jsonify({"error": "Utilisateur introuvable."}), 404
        
    old_hash = hash_password(data["old_password"])
    if users[username]["password_hash"] != old_hash:
        return jsonify({"error": "Ancien mot de passe incorrect."}), 401
        
    if len(data["new_password"]) < 6:
        return jsonify({"error": "Le nouveau mot de passe doit faire au moins 6 caractères."}), 400
        
    users[username]["password_hash"] = hash_password(data["new_password"])
    save_users(users)
    return jsonify({"message": "Mot de passe modifié avec succès."})


@app.route('/api/auth/change_username', methods=['POST'])
@token_required
def change_username():
    """Change le pseudo utilisateur et migre ses données."""
    data = request.get_json()
    new_username = data.get("new_username", "").strip()
    if not new_username or len(new_username) < 3:
        return jsonify({"error": "Pseudo trop court ou invalide."}), 400
        
    old_username = g.current_user
    users = load_users()
    
    if new_username in users:
        return jsonify({"error": "Ce pseudo est déjà pris."}), 409
        
    if old_username not in users:
        return jsonify({"error": "Utilisateur introuvable."}), 404
        
    # Migrate user
    users[new_username] = users.pop(old_username)
    save_users(users)
    
    # Migrate history
    local_history = load_history()
    if old_username in local_history:
        local_history[new_username] = local_history.pop(old_username)
        save_history(local_history)
        
    # Generate new token
    import datetime, jwt
    token = jwt.encode({
        "username": new_username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.datetime.utcnow()
    }, app.config['SECRET_KEY'], algorithm="HS256")
    
    return jsonify({"message": "Pseudo modifié", "token": token})


@app.route('/', methods=['GET'])
def root():
    """Point d'entree simple pour verifier que l'API repond."""
    return jsonify({
        "name": "Exoplanet Detection API v2",
        "status": "online",
        "docs_hint": "Utilisez /api/status pour verifier l'etat du systeme."
    })


# =============================================================================
# Routes : API publique
# =============================================================================

@app.route('/api/status', methods=['GET'])
def get_status():
    """État du système (pas besoin d'auth)."""
    return jsonify({
        "status": "online",
        "ai_loaded": model is not None,
        "features_count": len(selected_features),
        "features_sync": len(selected_features) > 0,
        "catalog_loaded": catalog_df is not None,
        "catalog_size": len(catalog_df) if catalog_df is not None else 0,
        "dataset_ready": catalog_df is not None
    })


# =============================================================================
# Routes : API protégée (nécessite JWT)
# =============================================================================

def run_full_analysis(target_id, mission, username):
    """
    Pipeline complète d'analyse. Appelée dans un thread séparé avec timeout global.
    Retourne un dict JSON-serializable ou lève une exception.
    """
    t0 = time.time()

    def log(msg):
        print(f"[{time.time()-t0:.1f}s] {msg}")

    log(f"Début acquisition {target_id}")
    lc_raw = fetch_lightcurve(target_id, mission=mission)
    if lc_raw is None:
        raise ValueError(f"Cible '{target_id}' introuvable dans les archives NASA.")
    log(f"Acquisition OK ({len(lc_raw)} points)")

    # Résolution du KIC depuis les métadonnées lightkurve (fonctionne pour Kepler-X aussi)
    resolved_kepid = _extract_kepid_from_lc(lc_raw)
    if resolved_kepid:
        log(f"KIC résolu : {resolved_kepid}")

    log("Prétraitement...")
    lc_clean = clean_and_flatten(lc_raw, quality="fast")
    if lc_clean is None:
        raise ValueError("Échec du prétraitement de la courbe.")
    log(f"Prétraitement OK ({len(lc_clean)} points après nettoyage)")

    log("BLS - recherche de période...")
    period, bls_stats = get_period_hint(lc_clean)
    lc_folded = fold_lightcurve(lc_clean, period=period)
    log(f"BLS OK - période = {period:.4f} j")

    log("Extraction features + prédiction XGBoost...")
    score = 0.5
    feature_importances = []

    if model and selected_features:
        # Déterminer si le modèle utilise les features BLS physiques (nouveau) ou TSFRESH (ancien)
        _BLS_FEATURES = {"bls_snr", "bls_depth_ppm", "bls_transit_fraction",
                         "bls_power", "bls_duration_days", "bls_score",
                         "period", "star_radius_solar", "star_temperature_k"}
        _uses_bls_model = all(f in _BLS_FEATURES for f in selected_features)

        if _uses_bls_model:
            # Nouveau modèle Kepler+TESS : vecteur BLS direct
            from src.p02_preprocessing import compute_transit_score
            bls_score_val = compute_transit_score(bls_stats)

            # Données stellaires depuis catalogue KOI si disponible
            cat_feats = get_catalog_features_dict(target_id, kepid=resolved_kepid)
            srad  = cat_feats.get("koi_srad",  1.0) or 1.0
            steff = cat_feats.get("koi_steff", 5500.0) or 5500.0

            row_vals = {
                "bls_snr":              float(bls_stats.get("bls_snr", 0)),
                "bls_depth_ppm":        float(bls_stats.get("bls_depth_ppm", 0)),
                "bls_transit_fraction": float(bls_stats.get("bls_transit_fraction", 0)),
                "bls_power":            float(bls_stats.get("bls_power", 0)),
                "bls_duration_days":    float(bls_stats.get("bls_duration_days", 0)),
                "bls_score":            float(bls_score_val),
                "period":               float(period),
                "star_radius_solar":    float(srad),
                "star_temperature_k":   float(steff),
            }
            input_data = pd.DataFrame([row_vals])[selected_features].astype(float)
        else:
            # Ancien modèle TSFRESH/KOI
            features_df = run_feature_extraction(lc_clean, target_id, bls_stats=bls_stats)
            input_data = build_input_vector(features_df, target_id, selected_features, resolved_kepid=resolved_kepid)

        score = float(model.predict_proba(input_data)[0][1])
        if hasattr(model, 'feature_importances_'):
            imp = model.feature_importances_
            top_idx = np.argsort(imp)[::-1][:5]
            feature_importances = [
                {"name": selected_features[i], "weight": float(imp[i])}
                for i in top_idx
            ]
    log(f"Prédiction OK - score = {score:.4f}")

    characterization = compute_characterization(lc_clean, lc_folded, period, score)
    metadata = get_real_metadata(target_id)

    if not is_finite_number(score):
        score = 0.5

    chart_data = build_chart_data(lc_folded)

    log(f"Pipeline terminée en {time.time()-t0:.1f}s")

    result = json_safe({
        "target": target_id,
        "mission": mission,
        "score": round(float(score), 4),
        "verdict": classify_score(score),
        "period_days": round(float(period), 4) if is_finite_number(period) else None,
        "points_count": len(lc_raw),
        "characterization": characterization,
        "metadata": metadata,
        "feature_importances": feature_importances,
        "data": chart_data,
        "analyzed_by": username,
    })

    save_history_entry(username, {
        "target": result["target"],
        "score": result["score"],
        "verdict": result["verdict"],
        "period_days": result["period_days"],
        "mission": result["mission"],
        "date": datetime.datetime.utcnow().isoformat(),
    })

    return result


@app.route('/api/analyze', methods=['GET'])
@token_required
def analyze_target():
    """
    Analyse complète d'une cible stellaire.

    Paramètres :
        id (str) : identifiant de la cible (ex: "Kepler-10", "KIC 11446443")
    """
    target_id = request.args.get('id', '').strip()
    if not target_id:
        return jsonify({"error": "Paramètre 'id' requis (ex: ?id=Kepler-10)"}), 400

    username = g.current_user
    cache_key = target_id.lower()

    # Cache in-memory avec TTL
    cached = _analysis_cache.get(cache_key)
    if cached and (time.time() - cached["ts"]) < CACHE_TTL:
        age = int(time.time() - cached["ts"])
        print(f"[Cache hit] {target_id} ({age}s ago)")
        result = dict(cached["result"])
        result["analyzed_by"] = username
        return jsonify(result)

    print(f"[Cache miss] {target_id} — lancement analyse (par {username})")
    if any(x in target_id.upper() for x in ["TIC", "TOI", "WASP"]):
        mission = "TESS"
    elif any(x in target_id for x in ["Kepler-", "KIC", "KOI"]):
        mission = "Kepler"
    else:
        mission = "Kepler"  # défaut

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_full_analysis, target_id, mission, username)
            try:
                result = future.result(timeout=90)
            except FuturesTimeout:
                return jsonify({"error": "Analyse trop longue (>90s). Réessayez."}), 504

        # Mise en cache
        _analysis_cache[cache_key] = {"result": result, "ts": time.time()}
        return jsonify(result)

    except ValueError as e:
        print(f"[Erreur métier] {target_id} : {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        print(f"[Erreur] Analyse de {target_id} : {e}")
        return jsonify({"error": f"Erreur lors de l'analyse : {str(e)}"}), 500


@app.route('/api/analyze/stream', methods=['GET'])
@token_required
def analyze_stream():
    """Analyse SSE : envoie la progression étape par étape."""
    target_id = request.args.get('id', '').strip()
    if not target_id:
        return jsonify({"error": "Paramètre 'id' requis"}), 400

    username = g.current_user

    def generate():
        def evt(name, data):
            return f"event: {name}\ndata: {json.dumps(data)}\n\n"

        # Résultat en cache → réponse instantanée
        cache_key = target_id.lower()
        if cache_key in results_cache:
            print(f"[Cache] Résultat servi pour {target_id}")
            cached = results_cache[cache_key]
            cached["analyzed_by"] = username
            yield evt("progress", {"step": "acquisition", "message": "Chargement depuis le cache...", "percent": 50})
            yield evt("progress", {"step": "done", "message": "Résultat disponible.", "percent": 100})
            yield evt("result", cached)
            return

        try:
            mission = "TESS" if any(x in target_id for x in ["TIC", "TOI", "WASP"]) else "Kepler"

            yield evt("progress", {"step": "acquisition", "message": "Téléchargement de la courbe de lumière...", "percent": 10})
            lc_raw = fetch_lightcurve(target_id, mission=mission)
            if lc_raw is None:
                yield evt("error", {"error": f"Cible '{target_id}' introuvable."})
                return

            resolved_kepid = _extract_kepid_from_lc(lc_raw)

            yield evt("progress", {"step": "preprocessing", "message": "Nettoyage et normalisation...", "percent": 30})
            lc_clean = clean_and_flatten(lc_raw, quality="fast")
            if lc_clean is None:
                yield evt("error", {"error": "Échec du prétraitement."})
                return

            yield evt("progress", {"step": "bls", "message": "Recherche de période (BLS)...", "percent": 50})
            period, bls_stats = get_period_hint(lc_clean)
            lc_folded = fold_lightcurve(lc_clean, period=period)

            yield evt("progress", {"step": "prediction", "message": "Prédiction par le modèle IA...", "percent": 70})
            score = 0.5
            top_features = []
            if model and selected_features:
                _BLS_FEATURES = {"bls_snr", "bls_depth_ppm", "bls_transit_fraction",
                                 "bls_power", "bls_duration_days", "bls_score",
                                 "period", "star_radius_solar", "star_temperature_k"}
                _uses_bls_model = all(f in _BLS_FEATURES for f in selected_features)
                if _uses_bls_model:
                    from src.p02_preprocessing import compute_transit_score
                    bls_score_val = compute_transit_score(bls_stats)
                    cat_feats = get_catalog_features_dict(target_id, kepid=resolved_kepid)
                    srad  = cat_feats.get("koi_srad",  1.0) or 1.0
                    steff = cat_feats.get("koi_steff", 5500.0) or 5500.0
                    row_vals = {
                        "bls_snr":              float(bls_stats.get("bls_snr", 0)),
                        "bls_depth_ppm":        float(bls_stats.get("bls_depth_ppm", 0)),
                        "bls_transit_fraction": float(bls_stats.get("bls_transit_fraction", 0)),
                        "bls_power":            float(bls_stats.get("bls_power", 0)),
                        "bls_duration_days":    float(bls_stats.get("bls_duration_days", 0)),
                        "bls_score":            float(bls_score_val),
                        "period":               float(period),
                        "star_radius_solar":    float(srad),
                        "star_temperature_k":   float(steff),
                    }
                    input_data = pd.DataFrame([row_vals])[selected_features].astype(float)
                else:
                    features_df = run_feature_extraction(lc_clean, target_id, bls_stats=bls_stats)
                    input_data = build_input_vector(features_df, target_id, selected_features, resolved_kepid=resolved_kepid)
                score = float(model.predict_proba(input_data)[0][1])
                if hasattr(model, 'feature_importances_'):
                    imp = model.feature_importances_
                    top_idx = np.argsort(imp)[::-1][:5]
                    top_features = [
                        {"name": selected_features[i], "importance": float(imp[i])}
                        for i in top_idx
                    ]

            yield evt("progress", {"step": "formatting", "message": "Formatage des résultats...", "percent": 90})
            characterization = compute_characterization(lc_clean, lc_folded, period, score)
            if not is_finite_number(score):
                score = 0.5

            chart_data = build_chart_data(lc_folded)

            result = json_safe({
                "target": target_id,
                "mission": mission,
                "score": round(float(score), 4),
                "verdict": classify_score(score),
                "period": round(float(period), 4) if is_finite_number(period) else None,
                "points_count": len(lc_raw),
                "characterization": characterization,
                "top_features": top_features,
                "data": chart_data,
                "analyzed_by": username,
            })
            # Sauvegarde dans le cache pour les prochaines fois
            results_cache[cache_key] = result
            save_results_cache()
            print(f"[Cache] Résultat sauvegardé pour {target_id}")
            yield evt("result", result)

        except Exception as e:
            print(f"[SSE Erreur] {target_id}: {e}")
            yield evt("error", {"error": str(e)})

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route('/api/metadata', methods=['GET'])
@token_required
def get_metadata():
    """Retourne les métadonnées stellaires d'une cible."""
    target_id = request.args.get('id', '').strip()
    if not target_id:
        return jsonify({"error": "Paramètre 'id' requis"}), 400
    meta = get_real_metadata(target_id)
    # Adapter les champs pour le frontend
    return jsonify({
        "star_type": None,
        "distance": None,
        "estimated_radius": f"{meta.get('star_radius_solar', '—')} R☉" if meta.get('star_radius_solar') else None,
        "nb_observations": None,
        "temperature_k": meta.get('star_temperature_k'),
        "kepler_magnitude": meta.get('kepler_magnitude'),
        "known_disposition": meta.get('known_disposition'),
        "source": meta.get('source'),
        "note": meta.get('note'),
    })


@app.route('/api/validate', methods=['GET'])
@token_required
def validate_target():
    """Valide la prédiction IA contre le catalogue NASA."""
    target_id = request.args.get('id', '').strip()
    if not target_id:
        return jsonify({"error": "Paramètre 'id' requis"}), 400

    meta = get_real_metadata(target_id)
    disposition = meta.get('known_disposition', '')
    nasa_confirmed = disposition in ('CONFIRMED', 'CANDIDATE')

    return jsonify({
        "target": target_id,
        "nasa_confirmed": nasa_confirmed,
        "known_disposition": disposition,
        "planet_name": target_id if nasa_confirmed else None,
        "known_period": meta.get('catalog_period'),
    })


@app.route('/api/auth/logout', methods=['POST'])
@token_required
def logout():
    """Déconnexion (côté client, le token est simplement ignoré)."""
    return jsonify({"message": "Déconnecté."})


@app.route('/api/history', methods=['GET'])
@token_required
def get_history():
    """Retourne l'historique des analyses du user connecté."""
    username = g.current_user
    history = load_history()
    return jsonify(history.get(username, []))


@app.route('/api/history', methods=['DELETE'])
@token_required
def clear_history():
    """Vide l'historique des analyses du user connecté."""
    username = g.current_user
    history = load_history()
    history[username] = []
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    return jsonify({"ok": True})


@app.route('/api/metrics', methods=['GET'])
@token_required
def get_metrics():
    """Retourne les métriques du modèle entraîné."""
    if not model_metrics:
        return jsonify({"error": "Aucune métrique disponible. Entraînez le modèle d'abord."}), 404
    
    return jsonify(model_metrics)


@app.route('/api/catalog/search', methods=['GET'])
@token_required
def search_catalog():
    """
    Recherche dans le catalogue Kepler KOI.
    
    Paramètres :
        q (str) : terme de recherche (nom ou KIC ID)
        limit (int) : nombre max de résultats (défaut: 20)
    """
    query = request.args.get('q', '').strip()
    limit = min(int(request.args.get('limit', 20)), 100)
    
    if not query:
        return jsonify({"error": "Paramètre 'q' requis."}), 400
    
    if catalog_df is None:
        return jsonify({"error": "Catalogue non chargé."}), 503
    
    # Recherche par KIC ID
    results = catalog_df[
        catalog_df['kepid'].astype(str).str.contains(query, case=False, na=False)
    ].head(limit)
    
    entries = []
    for _, row in results.iterrows():
        entries.append({
            "kepid": int(row['kepid']),
            "target_name": f"KIC {int(row['kepid'])}",
            "disposition": row.get('koi_disposition', 'UNKNOWN'),
            "period_days": round(float(row['koi_period']), 4) if pd.notna(row.get('koi_period')) else None,
            "depth_ppm": round(float(row['koi_depth']), 1) if pd.notna(row.get('koi_depth')) else None,
            "planet_radius_earth": round(float(row['koi_prad']), 2) if pd.notna(row.get('koi_prad')) else None
        })
    
    return jsonify({
        "query": query,
        "count": len(entries),
        "results": entries
    })


@app.route('/api/catalog/stars', methods=['GET'])
@token_required
def get_catalog_stars():
    page = int(request.args.get('page', 1))
    limit = min(int(request.args.get('limit', 20)), 100)
    search = request.args.get('search', '').strip()
    label = request.args.get('label', 'all')  # 'all', '0', '1'
    sort_by = request.args.get('sort_by', 'snr')  # 'snr', 'period', 'depth', 'score'
    sort_dir = request.args.get('sort_dir', 'desc')  # 'asc', 'desc'
    min_snr = request.args.get('min_snr', None)
    max_snr = request.args.get('max_snr', None)
    min_period = request.args.get('min_period', None)
    max_period = request.args.get('max_period', None)

    data = _catalog_cache_index[:]

    # filters
    if search:
        data = [s for s in data if search in str(s['kepid'])]
    if label == '1':
        data = [s for s in data if s['label'] == 1]
    elif label == '0':
        data = [s for s in data if s['label'] == 0]
    if min_snr is not None:
        data = [s for s in data if s.get('bls_snr') is not None and s['bls_snr'] >= float(min_snr)]
    if max_snr is not None:
        data = [s for s in data if s.get('bls_snr') is not None and s['bls_snr'] <= float(max_snr)]
    if min_period is not None:
        data = [s for s in data if s.get('period') is not None and s['period'] >= float(min_period)]
    if max_period is not None:
        data = [s for s in data if s.get('period') is not None and s['period'] <= float(max_period)]

    # sort
    sort_key_map = {'snr': 'bls_snr', 'period': 'period', 'depth': 'bls_depth_ppm', 'score': 'bls_score'}
    key = sort_key_map.get(sort_by, 'bls_snr')
    reverse = sort_dir != 'asc'
    data = sorted(data, key=lambda x: (x.get(key) is None, x.get(key) or 0), reverse=reverse)

    total = len(data)
    n_planets_filtered = sum(1 for s in data if s['label'] == 1)
    offset = (page - 1) * limit
    page_data = data[offset:offset + limit]

    # summary stats from full dataset
    all_ok = _catalog_cache_index
    n_planets = sum(1 for s in all_ok if s['label'] == 1)
    snrs = [s['bls_snr'] for s in all_ok if s.get('bls_snr') is not None]
    avg_snr = round(sum(snrs) / len(snrs), 2) if snrs else 0

    return jsonify({
        "total": total,
        "n_planets_filtered": n_planets_filtered,
        "page": page,
        "pages": (total + limit - 1) // limit,
        "limit": limit,
        "stars": page_data,
        "stats": {
            "total_stars": len(all_ok),
            "n_planets": n_planets,
            "n_non_planets": len(all_ok) - n_planets,
            "avg_snr": avg_snr,
        }
    })


@app.route('/api/catalog/upload', methods=['POST'])
@token_required
def upload_custom_star():
    """
    Upload a custom star CSV file for analysis.
    Required columns: time, flux
    Optional column: target_id
    """
    import io
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier fourni. Envoyez un champ 'file'."}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({"error": "Format non supporté. Seuls les fichiers .csv sont acceptés."}), 400

    try:
        content = file.read().decode('utf-8')
        df_custom = pd.read_csv(io.StringIO(content))
    except Exception as e:
        return jsonify({"error": f"Impossible de lire le CSV : {e}"}), 400

    # normalize column names
    df_custom.columns = [c.strip().lower() for c in df_custom.columns]

    if 'time' not in df_custom.columns or 'flux' not in df_custom.columns:
        return jsonify({"error": "Colonnes manquantes. Le fichier doit contenir au minimum 'time' et 'flux'."}), 400

    target_id = request.form.get('target_id', '') or df_custom.get('target_id', pd.Series(['Custom Star'])).iloc[0]
    if not isinstance(target_id, str):
        target_id = str(target_id)

    try:
        time_arr = np.array(df_custom['time'].values, dtype=float)
        flux_arr = np.array(df_custom['flux'].values, dtype=float)
    except Exception as e:
        return jsonify({"error": f"Les colonnes time/flux doivent être numériques : {e}"}), 400

    # Remove NaN
    mask = ~(np.isnan(time_arr) | np.isnan(flux_arr))
    time_arr = time_arr[mask]
    flux_arr = flux_arr[mask]

    if len(time_arr) < 50:
        return jsonify({"error": f"Pas assez de points de données ({len(time_arr)} trouvés, minimum 50 requis)."}), 400

    # Build a lightkurve LightCurve and run the pipeline
    try:
        import lightkurve as lk
        lc = lk.LightCurve(time=time_arr, flux=flux_arr)
        from src.p02_preprocessing import preprocess_lightcurve
        from src.p03_bls import run_bls
        from src.p04_features import extract_features
        lc_clean = preprocess_lightcurve(lc)
        if lc_clean is None or len(lc_clean) < 30:
            return jsonify({"error": "Prétraitement échoué : courbe trop courte après nettoyage."}), 400

        best_period, bls_model, bls_results = run_bls(lc_clean)
        if best_period is None:
            return jsonify({"error": "Détection BLS échouée. Vérifiez la qualité de vos données."}), 400

        features_dict, folded_lc = extract_features(lc_clean, best_period, bls_model, bls_results)
        if features_dict is None:
            return jsonify({"error": "Extraction de features échouée."}), 400

        if model is None:
            return jsonify({"error": "Modèle IA non chargé."}), 503

        features_df = pd.DataFrame([features_dict])
        features_df = features_df.reindex(columns=model.feature_names_in_, fill_value=0)
        score = float(model.predict_proba(features_df)[0][1])

        verdict = "Planète probable" if score >= 0.7 else "Signal ambigu" if score >= 0.35 else "Non planétaire"

        result_data = []
        if folded_lc is not None:
            fl_time = np.array(folded_lc.time.value if hasattr(folded_lc.time, 'value') else folded_lc.time, dtype=float)
            fl_flux = np.array(folded_lc.flux.value if hasattr(folded_lc.flux, 'value') else folded_lc.flux, dtype=float)
            mask2 = ~(np.isnan(fl_time) | np.isnan(fl_flux))
            result_data = [{"x": float(t), "y": float(f)} for t, f in zip(fl_time[mask2], fl_flux[mask2])]
            
        save_history_entry(g.current_user, {
            "target": target_id,
            "score": round(score, 4),
            "verdict": verdict,
            "period_days": round(best_period, 4),
            "mission": "Custom CSV",
            "date": datetime.datetime.utcnow().isoformat(),
        })

        return jsonify({
            "target": target_id,
            "score": round(score, 4),
            "verdict": verdict,
            "period_days": round(best_period, 4),
            "data": result_data[:1500],
            "n_points": len(time_arr),
        })
    except Exception as e:
        return jsonify({"error": f"Erreur lors de l'analyse : {e}"}), 500


# =============================================================================
# Fonctions utilitaires
# =============================================================================


def _extract_kepid_from_lc(lc):
    """
    Extrait le KIC (Kepler Input Catalogue) ID depuis les métadonnées d'une
    LightCurve lightkurve. Fonctionne que la cible ait été recherchée par
    'Kepler-10', 'KIC 11904151' ou n'importe quel autre alias.
    Retourne un int ou None.
    """
    if lc is None:
        return None
    # Essai 1 : attribut direct targetid
    try:
        v = lc.targetid
        if v is not None:
            return int(v)
    except Exception:
        pass
    # Essai 2 : clés connues dans lc.meta
    try:
        meta = lc.meta or {}
        for key in ('KEPLERID', 'keplerid', 'TARGETID', 'targetid', 'kepid'):
            v = meta.get(key)
            if v is not None:
                return int(v)
    except Exception:
        pass
    return None


def get_catalog_features_dict(target_id, kepid=None):
    """
    Retourne un dict des features physiques du catalogue KOI pour une cible.
    Accepte soit un kepid entier (prioritaire), soit un target_id string.
    Retourne {} si la cible est introuvable (les features seront mis à 0).
    """
    if catalog_df is None:
        return {}

    # Résolution du kepid : soit passé directement, soit extrait du target_id
    if kepid is None:
        if "KIC" in target_id:
            try:
                kepid = int(target_id.replace("KIC", "").strip())
            except ValueError:
                pass

    if kepid is None:
        return {}

    match = catalog_df[catalog_df['kepid'] == kepid]
    if len(match) == 0:
        return {}

    row = match.iloc[0]
    feats = {}
    
    # Collecte dynamique de TOUTES les colonnes numériques dispo
    for col in catalog_df.columns:
        v = row.get(col)
        if pd.notna(v) and isinstance(v, (int, float, np.number)):
            feats[col] = float(v)

    # Récupération Glon/Glat dynamiquement pour coller à l'entraînement astropy
    if 'ra' in row and 'dec' in row and pd.notna(row['ra']) and pd.notna(row['dec']):
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            coords = SkyCoord(ra=row['ra']*u.deg, dec=row['dec']*u.deg, frame='icrs')
            feats['glon'] = float(coords.galactic.l.degree)
            feats['glat'] = float(coords.galactic.b.degree)
        except Exception:
            pass

    eps = 1e-9
    period   = feats.get('koi_period')
    depth    = feats.get('koi_depth')
    duration = feats.get('koi_duration')
    prad     = feats.get('koi_prad')
    srad     = feats.get('koi_srad')
    kepmag   = feats.get('koi_kepmag')

    # Features dérivées
    if period  is not None: feats['log_koi_period']  = float(np.log1p(max(period, 0)))
    if depth   is not None: feats['log_koi_depth']   = float(np.log1p(max(depth, 0)))
    if prad    is not None: feats['log_koi_prad']    = float(np.log1p(max(prad, 0)))
    if period  is not None and duration is not None:
        feats['duty_cycle'] = duration / (period * 24 + eps)
    if prad is not None and srad is not None:
        feats['ratio_prad_srad'] = prad / (srad * 109.076 + eps)
    if depth is not None and srad is not None:
        feats['depth_per_srad'] = depth / (srad + eps)
    if depth is not None and kepmag is not None:
        snr_p = depth / (10 ** (kepmag / 2.5 + eps))
        feats['snr_proxy']     = snr_p
        feats['log_snr_proxy'] = float(np.log1p(max(snr_p, 0)))

    return feats


def build_input_vector(features_df, target_id, selected_features_list, resolved_kepid=None):
    """
    Construit le DataFrame d'entrée pour le modèle en combinant :
    - Features TSFRESH / scientifiques extraites de la courbe de lumière
    - Features physiques du catalogue KOI (si disponibles)
    Les features manquantes sont mises à 0.
    resolved_kepid : KIC ID entier extrait depuis lightkurve (prioritaire sur target_id string).
    """
    # On force la création d'1 et 1 seule ligne (index=[0])
    input_data = pd.DataFrame(index=[0], columns=selected_features_list)

    # 1. Injecter les features de la courbe de lumière
    if features_df is not None and not features_df.empty:
        for col in selected_features_list:
            if col in features_df.columns:
                input_data.loc[0, col] = features_df[col].iloc[0]

    # 2. Enrichir avec les features du catalogue KOI
    #    On utilise le kepid résolu depuis lightkurve (fonctionne pour Kepler-X)
    #    et on tombe en arrière sur la recherche par string si nécessaire.
    cat_feats = get_catalog_features_dict(target_id, kepid=resolved_kepid)
    for col, val in cat_feats.items():
        if col in selected_features_list:
            input_data.loc[0, col] = val

    # 3. Remplir les manquants par 0
    input_data = input_data.fillna(0)
    
    # 4. Forcer le type float pour éviter l'erreur XGBoost 'object'
    return input_data.astype(float)


def classify_score(score):
    """Traduit le score en verdict lisible."""
    if score >= 0.85:
        return "Exoplanète très probable"
    elif score >= 0.70:
        return "Exoplanète probable"
    elif score >= 0.55:
        return "Candidat à confirmer"
    elif score >= 0.35:
        return "Indéterminé"
    elif score >= 0.15:
        return "Probable faux positif"
    else:
        return "Faux positif très probable"


def compute_characterization(lc_clean, lc_folded, period, score):
    """
    Caractérisation physique du signal détecté.
    Estime le rayon planétaire, le SNR du transit, et la durée.
    """
    try:
        flux_folded = np.array(lc_folded.flux.value, dtype=float)
        flux_folded = flux_folded[~np.isnan(flux_folded)]
        
        if len(flux_folded) < 10:
            return {"error": "Pas assez de données pour caractériser"}
        
        # Profondeur du transit (delta F / F)
        baseline = np.median(flux_folded)
        transit_depth = baseline - np.percentile(flux_folded, 1)  # percentile 1% pour robustesse
        depth_ppm = transit_depth * 1e6  # en parties par million
        
        # Estimation du rayon planétaire (en rayons terrestres)
        # Formule : Rp/R* = sqrt(delta_F)
        # On suppose R* ~ 1 R_soleil par défaut
        rp_over_rstar = np.sqrt(abs(transit_depth)) if transit_depth > 0 else 0
        planet_radius_earth = rp_over_rstar * 109.076  # R_soleil / R_terre
        
        # SNR du transit
        noise_std = np.std(flux_folded)
        snr = transit_depth / noise_std if noise_std > 0 else 0
        
        # Classification du type de planète
        if planet_radius_earth < 1.25:
            planet_type = "Terrestre (type Terre)"
        elif planet_radius_earth < 2.0:
            planet_type = "Super-Terre"
        elif planet_radius_earth < 4.0:
            planet_type = "Mini-Neptune"
        elif planet_radius_earth < 10.0:
            planet_type = "Neptune-like"
        else:
            planet_type = "Géante gazeuse (type Jupiter)"
        
        return {
            "transit_depth_ppm": round(depth_ppm, 1),
            "planet_radius_earth": round(planet_radius_earth, 2),
            "planet_type": planet_type,
            "snr": round(snr, 2),
            "period_days": round(period, 4),
            "confidence": "high" if snr > 10 else "medium" if snr > 5 else "low"
        }
        
    except Exception as e:
        return {"error": f"Caractérisation échouée : {str(e)}"}


def get_real_metadata(target_id):
    """
    Récupère les vraies métadonnées stellaires depuis le catalogue NASA.
    """
    if catalog_df is None:
        return {"note": "Catalogue non disponible"}
    
    # Extraction du KIC ID depuis le nom
    kepid = None
    
    if "KIC" in target_id:
        try:
            kepid = int(target_id.replace("KIC", "").strip())
        except ValueError:
            pass
    elif "Kepler-" in target_id:
        # Pour les noms Kepler-X, on cherche dans le catalogue par nom
        # (lightkurve résout le KIC automatiquement, mais pas nous ici)
        pass
    
    if kepid is not None:
        match = catalog_df[catalog_df['kepid'] == kepid]
        if len(match) > 0:
            row = match.iloc[0]
            return {
                "kepid": int(row['kepid']),
                "star_temperature_k": int(row['koi_steff']) if pd.notna(row.get('koi_steff')) else None,
                "star_radius_solar": round(float(row['koi_srad']), 3) if pd.notna(row.get('koi_srad')) else None,
                "kepler_magnitude": round(float(row['koi_kepmag']), 2) if pd.notna(row.get('koi_kepmag')) else None,
                "known_disposition": row.get('koi_disposition', 'UNKNOWN'),
                "catalog_period": round(float(row['koi_period']), 4) if pd.notna(row.get('koi_period')) else None,
                "catalog_depth_ppm": round(float(row['koi_depth']), 1) if pd.notna(row.get('koi_depth')) else None,
                "catalog_planet_radius": round(float(row['koi_prad']), 2) if pd.notna(row.get('koi_prad')) else None,
                "source": "NASA Exoplanet Archive (KOI Table)"
            }
    
    return {
        "note": "Métadonnées non trouvées dans le catalogue KOI.",
        "hint": "Utilisez un identifiant KIC (ex: KIC 11446443) pour des données complètes."
    }


# =============================================================================
# Star Info — NASA Exoplanet Archive
# =============================================================================

import re as _re
import requests as _requests

_star_info_cache = {}
_STAR_INFO_TTL   = 3600 * 24  # 24 h

_NASA_TAP = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def _hostname_from_target(target: str) -> str:
    """Retire le suffixe de planète éventuel (ex: 'Kepler-452b' → 'Kepler-452')."""
    t = target.strip()
    t = _re.sub(r'(\d)\s*[b-z]$', r'\1', t, flags=_re.IGNORECASE)
    return t


def _query_nasa(adql: str, timeout: int = 8):
    """Lance une requête ADQL sur le TAP NASA et retourne la liste de dicts."""
    resp = _requests.get(_NASA_TAP, params={"query": adql, "format": "json"}, timeout=timeout)
    resp.raise_for_status()
    return resp.json() or []


@app.route('/api/star_info', methods=['GET'])
@token_required
def get_star_info():
    """
    Retourne les données stellaires et planétaires depuis NASA Exoplanet Archive.
    Paramètre GET : target (ex. 'Kepler-452', 'Kepler-452b', 'KIC 11446443')
    """
    target = request.args.get('target', '').strip()
    if not target:
        return jsonify({"error": "Paramètre 'target' requis"}), 400

    cache_key = target.lower()
    if cache_key in _star_info_cache:
        entry = _star_info_cache[cache_key]
        if time.time() - entry['ts'] < _STAR_INFO_TTL:
            return jsonify(entry['data'])

    hostname = _hostname_from_target(target)

    result = {
        "target":   target,
        "hostname": hostname,
        "stellar":  None,
        "planets":  [],
        "source":   None,
    }

    try:
        # -- Données planétaires + stellaires (table ps, une ligne par planète) --
        q_planets = (
            "SELECT hostname, pl_name, pl_orbper, pl_rade, pl_eqt, pl_insol, "
            "st_teff, st_rad, st_mass, st_lum, sy_dist, sy_kmag, "
            "disc_year, disc_method "
            f"FROM ps WHERE UPPER(hostname) = UPPER('{hostname}') "
            "AND default_flag = 1"
        )
        rows = _query_nasa(q_planets)

        if rows:
            r0 = rows[0]
            result["stellar"] = {
                "teff":        r0.get("st_teff"),       # Température eff. (K)
                "radius":      r0.get("st_rad"),         # Rayon (R☉)
                "mass":        r0.get("st_mass"),        # Masse (M☉)
                "luminosity":  r0.get("st_lum"),         # Luminosité (log L☉)
                "distance_pc": r0.get("sy_dist"),        # Distance (pc)
                "kmag":        r0.get("sy_kmag"),        # Magnitude K
            }
            result["planets"] = [
                {
                    "name":         r.get("pl_name"),
                    "period_days":  r.get("pl_orbper"),
                    "radius_earth": r.get("pl_rade"),
                    "eq_temp":      r.get("pl_eqt"),
                    "insolation":   r.get("pl_insol"),
                    "disc_year":    r.get("disc_year"),
                    "disc_method":  r.get("disc_method"),
                }
                for r in rows
            ]
            result["source"] = "NASA Exoplanet Archive"

        else:
            # Fallback : table stellarhosts (étoiles sans planète confirmée)
            q_star = (
                "SELECT hostname, st_teff, st_rad, st_mass, st_lum, sy_dist, sy_kmag "
                f"FROM stellarhosts WHERE UPPER(hostname) = UPPER('{hostname}') "
                "AND default_flag = 1"
            )
            stars = _query_nasa(q_star)
            if stars:
                s = stars[0]
                result["stellar"] = {
                    "teff":        s.get("st_teff"),
                    "radius":      s.get("st_rad"),
                    "mass":        s.get("st_mass"),
                    "luminosity":  s.get("st_lum"),
                    "distance_pc": s.get("sy_dist"),
                    "kmag":        s.get("sy_kmag"),
                }
                result["source"] = "NASA Exoplanet Archive (stellar hosts)"

    except Exception as exc:
        print(f"[star_info] Erreur NASA API pour '{hostname}': {exc}")
        # On ne lève pas — résultat vide renvoyé proprement

    _star_info_cache[cache_key] = {"data": result, "ts": time.time()}
    return jsonify(result)


# =============================================================================
# Gestionnaire d'erreurs global (garantit les headers CORS sur les 500)
# =============================================================================

@app.errorhandler(HTTPException)
def handle_http_exception(e):
    """Conserve les vrais codes HTTP (404, 401, etc.)."""
    return jsonify({"error": e.description}), e.code

@app.errorhandler(Exception)
def handle_exception(e):
    """Capture toutes les exceptions non gérées et retourne du JSON avec CORS."""
    print(f"[Exception non gérée] {type(e).__name__}: {e}")
    return jsonify({"error": str(e)}), 500


# =============================================================================
# Lancement
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Exoplanet Detection API v2")
    print("=" * 50)

    # Le reloader de Flask peut boucler sous Windows quand des libs Python
    # modifient des fichiers dans site-packages. On le coupe pour garder
    # un demarrage stable du backend.
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True, use_reloader=False)
