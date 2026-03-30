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
from src.p02_preprocessing import clean_and_flatten, clean_only, fold_lightcurve, get_period_hint, compute_transit_score
from src.p04_features import run_feature_extraction


# =============================================================================
# Configuration
# =============================================================================

app = Flask(__name__)
CORS(app)

# Clé secrète JWT (en production, utiliser une variable d'environnement)
JWT_SECRET = os.environ.get("JWT_SECRET", "exoplanet-detection-secret-key-change-in-prod")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Chemins
MODEL_PATH = "models/exoplanet_model.json"
FEATURES_PATH = "models/selected_features.json"
METRICS_PATH = "models/model_metrics.json"
CATALOG_PATH = "data/catalog/kepler_koi_catalog.csv"
USERS_PATH = "data/users.json"
RESULTS_CACHE_PATH = "data/results_cache.json"


# =============================================================================
# Chargement des ressources
# =============================================================================

model = None
selected_features = []
model_metrics = {}
catalog_df = None
results_cache = {}
optimal_threshold = 0.5  # seuil de décision (chargé depuis model_metrics.json)

# Cache in-memory avec TTL (clé → {"result": ..., "ts": float})
_analysis_cache = {}
CACHE_TTL = 600  # 10 minutes


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
    global model, selected_features, model_metrics, catalog_df, optimal_threshold
    
    # Modèle XGBoost
    if os.path.exists(MODEL_PATH):
        try:
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            print("[OK] Modèle XGBoost v2 chargé.")
        except Exception as e:
            print(f"[!] Erreur chargement modèle : {e}")
    else:
        print("[!] Modèle introuvable. Lancez 03_kaggle_tsfresh_train.py d'abord.")
    
    # Features sélectionnées
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r") as f:
            selected_features = json.load(f)
        print(f"[OK] {len(selected_features)} features TSFRESH chargées.")
    
    # Métriques + seuil optimal
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            model_metrics = json.load(f)
        optimal_threshold = model_metrics.get("optimal_threshold", 0.5)
        print(f"[OK] Métriques chargées — seuil optimal = {optimal_threshold:.4f}")
    
    # Catalogue Kepler
    if os.path.exists(CATALOG_PATH):
        catalog_df = pd.read_csv(CATALOG_PATH)
        print(f"[OK] Catalogue NASA chargé ({len(catalog_df)} entrées).")
    else:
        print("[!] Catalogue NASA introuvable. Les métadonnées stellaires seront limitées.")


load_results_cache()

load_resources()


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
        "created_at": datetime.datetime.utcnow().isoformat()
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
        "username": username
    })


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

    log("Prétraitement...")
    lc_cleaned = clean_only(lc_raw)
    if lc_cleaned is None:
        raise ValueError("Échec du nettoyage de la courbe.")
    log(f"Nettoyage OK ({len(lc_cleaned)} points)")

    lc_flat = clean_and_flatten(lc_raw, quality="fast")
    if lc_flat is None:
        lc_flat = lc_cleaned
    log(f"Aplatissement OK ({len(lc_flat)} points)")

    log("BLS - recherche de période + détection transit...")
    period, bls_stats = get_period_hint(lc_flat)
    lc_folded = fold_lightcurve(lc_flat, period=period)
    bls_score = compute_transit_score(bls_stats)
    log(f"BLS OK - période = {period:.4f} j, SNR={bls_stats.get('bls_snr',0):.1f}, BLS={bls_score:.3f}")

    # Récupération de la vraie taille de l'étoile pour l'IA
    metadata = get_real_metadata(target_id)
    default_srad, default_steff = 1.0, 5500.0  # valeurs par défaut soleil
    star_radius = metadata.get("star_radius_solar") or default_srad
    star_temp = metadata.get("star_temperature_k") or default_steff

    # Scoring ML Physics-Informed (XGBoost sur variables BLS + Stellaires)
    score = bls_score  # fallback
    feature_importances = [
        {"name": "BLS SNR",            "weight": round(bls_stats.get("bls_snr", 0), 2)},
        {"name": "Transit depth (ppm)", "weight": round(bls_stats.get("bls_depth_ppm", 0), 1)},
        {"name": "Période (jours)",     "weight": round(period, 2)},
        {"name": "Transit fraction",    "weight": round(bls_stats.get("bls_transit_fraction", 0), 4)},
        {"name": "BLS power",           "weight": round(bls_stats.get("bls_power", 0), 4)},
    ]

    if model and selected_features:
        log("Prédiction ML (Physics-Informed XGBoost)...")
        try:
            # Créer le vecteur d'entrée avec EXACTEMENT les selected_features
            input_dict = {}
            for col in selected_features:
                if col == "period":
                    input_dict[col] = [period]
                elif col == "bls_score":
                    input_dict[col] = [bls_score]
                elif col == "bls_duration_days":
                    input_dict[col] = [bls_stats.get("bls_duration_days", 0)]
                elif col == "star_radius_solar":
                    input_dict[col] = [star_radius]
                elif col == "star_temperature_k":
                    input_dict[col] = [star_temp]
                else:
                    input_dict[col] = [bls_stats.get(col, 0)]
            
            input_data = pd.DataFrame(input_dict)
            input_data = input_data.fillna(0)

            score = float(model.predict_proba(input_data)[0][1])
            log(f"ML score = {score:.4f}")

            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
                top_idx = np.argsort(imp)[::-1]
                feature_importances = [
                    {"name": selected_features[i], "weight": float(imp[i])}
                    for i in top_idx if imp[i] > 0
                ]
        except Exception as e:
            log(f"[!] Erreur ML, fallback BLS : {e}")
            score = bls_score

    log(f"Score final = {score:.4f}")

    characterization = compute_characterization(lc_cleaned, lc_folded, period, score)
    # metadata is already fetched above

    if not is_finite_number(score):
        score = 0.5

    chart_data = build_chart_data(lc_folded)

    log(f"Pipeline terminée en {time.time()-t0:.1f}s")

    return json_safe({
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

            yield evt("progress", {"step": "preprocessing", "message": "Nettoyage du signal...", "percent": 30})
            lc_cleaned = clean_only(lc_raw)
            if lc_cleaned is None:
                yield evt("error", {"error": "Échec du nettoyage."})
                return
            lc_flat = clean_and_flatten(lc_raw, quality="fast")
            if lc_flat is None:
                lc_flat = lc_cleaned

            yield evt("progress", {"step": "bls", "message": "Détection de transit (BLS)...", "percent": 50})
            period, bls_stats = get_period_hint(lc_flat)
            lc_folded = fold_lightcurve(lc_flat, period=period)
            bls_score = compute_transit_score(bls_stats)

            yield evt("progress", {"step": "prediction", "message": "Classification ML + BLS...", "percent": 70})
            
            # Récupération métadonnées stellaires
            metadata_sse = get_real_metadata(target_id)
            star_radius = metadata_sse.get("star_radius_solar") or 1.0
            star_temp = metadata_sse.get("star_temperature_k") or 5500.0
            
            score = bls_score
            top_features = [
                {"name": "BLS SNR",            "importance": round(bls_stats.get("bls_snr", 0), 2)},
                {"name": "Transit depth (ppm)", "importance": round(bls_stats.get("bls_depth_ppm", 0), 1)},
                {"name": "Période (jours)",     "importance": round(period, 2)},
                {"name": "Transit fraction",    "importance": round(bls_stats.get("bls_transit_fraction", 0), 4)},
                {"name": "BLS power",           "importance": round(bls_stats.get("bls_power", 0), 4)},
            ]

            if model and selected_features:
                try:
                    input_dict = {}
                    for col in selected_features:
                        if col == "period":
                            input_dict[col] = [period]
                        elif col == "bls_score":
                            input_dict[col] = [bls_score]
                        elif col == "bls_duration_days":
                            input_dict[col] = [bls_stats.get("bls_duration_days", 0)]
                        elif col == "star_radius_solar":
                            input_dict[col] = [star_radius]
                        elif col == "star_temperature_k":
                            input_dict[col] = [star_temp]
                        else:
                            input_dict[col] = [bls_stats.get(col, 0)]
                    
                    input_data = pd.DataFrame(input_dict).fillna(0)
                    score = float(model.predict_proba(input_data)[0][1])

                    if hasattr(model, 'feature_importances_'):
                        imp = model.feature_importances_
                        top_idx = np.argsort(imp)[::-1]
                        top_features = [
                            {"name": selected_features[i], "importance": float(imp[i])}
                            for i in top_idx if imp[i] > 0
                        ]
                except Exception as e:
                    print(f"[!] ML error, BLS fallback: {e}")
                    score = bls_score

            print(f"[Score] {score:.4f}")

            yield evt("progress", {"step": "formatting", "message": "Formatage des résultats...", "percent": 90})
            characterization = compute_characterization(lc_cleaned, lc_folded, period, score)
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


@app.route('/api/metrics', methods=['GET'])
@token_required
def get_metrics():
    """Retourne les métriques du modèle entraîné."""
    if not model_metrics:
        return jsonify({"error": "Aucune métrique disponible. Entraînez le modèle d'abord."}), 404
    
    # Mapper les noms du modèle v2 vers ceux attendus par le frontend
    m = model_metrics
    mapped = {
        # Holdout → test_*
        "test_accuracy":    m.get("holdout_accuracy",  m.get("test_accuracy", 0)),
        "test_precision":   m.get("holdout_precision", m.get("test_precision", 0)),
        "test_recall":      m.get("holdout_recall",    m.get("test_recall", 0)),
        "test_f1":          m.get("holdout_f1",        m.get("test_f1", 0)),
        "test_auc_roc":     m.get("holdout_auc_roc",   m.get("test_auc_roc", 0)),
        # CV → cv_*  (supporte cv5_, cv10_, et cv_ brut)
        "cv_accuracy_mean": m.get("cv5_accuracy_mean", m.get("cv10_accuracy_mean", m.get("cv_accuracy_mean", 0))),
        "cv_accuracy_std":  m.get("cv5_accuracy_std",  m.get("cv10_accuracy_std",  m.get("cv_accuracy_std", 0))),
        "cv_f1_mean":       m.get("cv5_f1_mean",       m.get("cv10_f1_mean",       m.get("cv_f1_mean", 0))),
        "cv_f1_std":        m.get("cv5_f1_std",         m.get("cv10_f1_std",         m.get("cv_f1_std", 0))),
        "cv_auc_mean":      m.get("cv5_roc_auc_mean",  m.get("cv10_roc_auc_mean",  m.get("cv_auc_mean", 0))),
        "cv_auc_std":       m.get("cv5_roc_auc_std",   m.get("cv10_roc_auc_std",   m.get("cv_auc_std", 0))),
        # Infos modèle
        "n_features_selected": m.get("n_features", len(selected_features)),
        "n_features_total":    m.get("n_features", len(selected_features)),
        "train_size":          m.get("train_size", m.get("holdout_size", 0)),
        "test_size":           m.get("holdout_size", m.get("test_size", 0)),
        "confusion_matrix":    m.get("confusion_matrix", [[0,0],[0,0]]),
        "top_features":        m.get("top_features", []),
        "optimal_threshold":       m.get("optimal_threshold", 0.5),
        "accuracy_realistic":      m.get("accuracy_realistic_distribution", None),
        "realistic_planet_rate":   m.get("realistic_planet_rate", 0.007),
        "threshold_strategy":      m.get("threshold_strategy", ""),
        "source":                  m.get("source", "XGBoost v2 + TSFRESH"),
    }
    
    return jsonify(mapped)


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


# =============================================================================
# Fonctions utilitaires
# =============================================================================

def classify_score(score):
    """Traduit la probabilité brute du modèle en verdict lisible."""
    if score >= 0.95:
        return "Exoplanète très probable"
    elif score >= 0.80:
        return "Candidat prometteur"
    elif score >= 0.50:
        return "Signal intéressant"
    elif score >= 0.30:
        return "Signal ambigu"
    elif score >= 0.10:
        return "Probablement un faux positif"
    else:
        return "Faux positif"


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
