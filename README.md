# Détection d'Exoplanètes par Machine Learning

Projet ING4 — ECE Paris  
S. Gallais, M. Rolland, C. De Blauwe, M. Leitao, O. Schwartz, K. Benjelloum

---

## Présentation

Application web de détection d'exoplanètes à partir de courbes de lumière stellaire. L'utilisateur entre le nom d'une étoile (Kepler, TESS ou KIC/TIC) et obtient en quelques secondes une prédiction par un modèle XGBoost entraîné sur les données NASA.

Le pipeline complet : téléchargement de la courbe de lumière via l'archive NASA MAST → prétraitement → extraction de features → prédiction → caractérisation de la planète candidate.

---

## Stack

- **Frontend** : React + Vite
- **Backend** : Flask (Python)
- **Modèle** : XGBoost, entraîné sur Kepler KOI + TESS TOI (9 961 entrées)
- **Données** : NASA Exoplanet Archive, NASA MAST via Lightkurve

---

## Installation

### Prérequis
- Python 3.9+
- Node.js 18+

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Le serveur démarre sur `http://localhost:5001`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

L'interface est accessible sur `http://localhost:5173`.

---

## Modèle

Le modèle XGBoost est entraîné sur la fusion de deux catalogues NASA :

| Source | Entrées |
|--------|---------|
| Kepler KOI | 7 316 |
| TESS TOI | 2 645 |
| **Total** | **9 961** |

**Performances sur le jeu de test (20%) :**

| Métrique | Score |
|----------|-------|
| Précision | 90.1% |
| Recall | 91.3% |
| F1-Score | 90.7% |
| AUC-ROC | **98.1%** |

32 features utilisées : paramètres orbitaux, stellaires et leurs incertitudes de mesure (`koi_period`, `koi_prad`, `koi_steff`, `is_tess`, `glon`/`glat`…).

Pour ré-entraîner le modèle :

```bash
source backend/venv/bin/activate
python backend/scripts/07_kaggle_train.py
```

---

## Structure du projet

```
├── backend/
│   ├── app.py                   # API Flask
│   ├── models/                  # Modèle XGBoost + métriques + features
│   ├── scripts/
│   │   ├── 07_kaggle_train.py   # Entraînement Kepler + TESS
│   │   └── download_tess_toi.py # Téléchargement catalogue TESS
│   └── src/                     # Pipeline : acquisition, prétraitement, features
├── frontend/
│   └── src/App.jsx              # Interface React
├── data/
│   └── catalog/                 # Catalogues CSV Kepler et TESS
└── models/                      # Modèle (copie racine)
```

---

## Fonctionnalités

- **Analyse** : prédiction sur n'importe quelle étoile Kepler ou TESS
- **Comparaison** : jusqu'à 3 étoiles côte à côte
- **Catalogue** : 9 500+ étoiles indexées, filtres par mission / label / SNR / période
- **Upload CSV** : analyse d'une courbe de lumière personnalisée (colonnes `time`, `flux`)
- **Métriques** : dashboard du modèle (matrice de confusion, AUC-ROC, importance des features)
- **Profil** : historique des analyses, statistiques, réalisations

---

## Données

Les catalogues sont issus de la NASA Exoplanet Archive :
- Kepler KOI : `data/catalog/exoplanet_binary_model_imputed.csv`
- TESS TOI : `data/catalog/tess_toi_binary.csv` (généré par `download_tess_toi.py`)

Les courbes de lumière sont téléchargées à la demande via [Lightkurve](https://lightkurve.github.io/lightkurve/) depuis NASA MAST.
