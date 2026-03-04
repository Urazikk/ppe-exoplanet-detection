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
import hashlib
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from functools import wraps

from flask import Flask, request, jsonify, g
from flask_cors import CORS

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


# =============================================================================
# Chargement des ressources
# =============================================================================

model = None
selected_features = []
model_metrics = {}
catalog_df = None


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
        "catalog_loaded": catalog_df is not None,
        "catalog_size": len(catalog_df) if catalog_df is not None else 0
    })


# =============================================================================
# Routes : API protégée (nécessite JWT)
# =============================================================================

@app.route('/api/analyze', methods=['GET'])
@token_required
def analyze_target():
    """
    Analyse complète d'une cible stellaire.
    
    Paramètres :
        id (str) : identifiant de la cible (ex: "Kepler-10", "KIC 11446443")
    
    Retourne :
        - score : probabilité de présence d'une exoplanète (0 à 1)
        - period : période orbitale détectée (jours)
        - characterization : estimation du rayon, SNR, type de transit
        - metadata : données stellaires réelles du catalogue NASA
        - data : points de la courbe repliée pour visualisation
    """
    target_id = request.args.get('id', '').strip()
    
    if not target_id:
        return jsonify({"error": "Paramètre 'id' requis (ex: ?id=Kepler-10)"}), 400
    
    print(f"[Analyse] {target_id} (par {g.current_user})")
    
    try:
        # 1. Détection de la mission
        mission = "TESS" if any(x in target_id for x in ["TIC", "TOI", "WASP"]) else "Kepler"
        
        # 2. Acquisition
        lc_raw = fetch_lightcurve(target_id, mission=mission)
        if lc_raw is None:
            return jsonify({"error": f"Cible '{target_id}' introuvable dans les archives NASA."}), 404
        
        # 3. Prétraitement
        lc_clean = clean_and_flatten(lc_raw, quality="fast")
        if lc_clean is None:
            return jsonify({"error": "Échec du prétraitement de la courbe."}), 500
        
        # 4. Détection de période (BLS)
        period = get_period_hint(lc_clean)
        lc_folded = fold_lightcurve(lc_clean, period=period)
        
        # 5. Prédiction
        score = 0.5
        feature_importances = []
        
        if model and selected_features:
            features_df = run_feature_extraction(lc_clean, target_id)
            
            if features_df is not None:
                # Alignement avec les features du modèle
                available = [f for f in selected_features if f in features_df.columns]
                missing = [f for f in selected_features if f not in features_df.columns]
                
                input_data = pd.DataFrame(columns=selected_features)
                for col in available:
                    input_data[col] = features_df[col].values
                for col in missing:
                    input_data[col] = 0
                
                input_data = input_data.fillna(0)
                score = float(model.predict_proba(input_data)[0][1])
                
                # Top features qui ont pesé dans la décision
                if hasattr(model, 'feature_importances_'):
                    imp = model.feature_importances_
                    top_idx = np.argsort(imp)[::-1][:5]
                    feature_importances = [
                        {"name": selected_features[i], "weight": float(imp[i])}
                        for i in top_idx
                    ]
        
        # 6. Caractérisation
        characterization = compute_characterization(lc_clean, lc_folded, period, score)
        
        # 7. Métadonnées stellaires réelles
        metadata = get_real_metadata(target_id)
        
        # 8. Données pour le graphique (downsampled)
        step = max(1, len(lc_folded) // 800)
        chart_data = [
            {"time": round(float(t), 5), "flux": round(float(f), 6)}
            for t, f in zip(
                lc_folded.time.value[::step],
                lc_folded.flux.value[::step]
            )
        ]
        
        return jsonify({
            "target": target_id,
            "mission": mission,
            "score": round(score, 4),
            "verdict": classify_score(score),
            "period_days": round(period, 4),
            "points_count": len(lc_raw),
            "characterization": characterization,
            "metadata": metadata,
            "feature_importances": feature_importances,
            "data": chart_data,
            "analyzed_by": g.current_user
        })
    
    except Exception as e:
        print(f"[Erreur] Analyse de {target_id} : {str(e)}")
        return jsonify({"error": f"Erreur lors de l'analyse : {str(e)}"}), 500


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


# =============================================================================
# Fonctions utilitaires
# =============================================================================

def classify_score(score):
    """Traduit le score en verdict lisible."""
    if score >= 0.85:
        return "Exoplanète très probable"
    elif score >= 0.65:
        return "Candidat prometteur"
    elif score >= 0.4:
        return "Signal ambigu"
    elif score >= 0.2:
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
# Lancement
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Exoplanet Detection API v2")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5001)