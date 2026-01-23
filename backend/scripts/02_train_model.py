import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import sys
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel

# Configuration du PATH pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train_exoplanet_model():
    """
    Entraîne l'IA sur le dataset d'entraînement et valide sur le set de test indépendant.
    Optimisé pour gérer le déséquilibre des classes.
    """
    train_path = "data/processed/training_dataset.csv"
    test_path = "data/processed/test_dataset.csv"
    model_dir = "models"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("❌ Erreur : Les datasets CSV sont introuvables. Relancez le générateur.")
        return

    print("🚀 Chargement des données CSV...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # 1. Préparation robuste des données
    y_train = df_train['target_label']
    y_test = df_test['target_label']

    # Filtrage des colonnes numériques uniquement
    X_train = df_train.select_dtypes(include=[np.number]).drop(columns=['target_label'], errors='ignore').fillna(0)
    X_test = df_test.select_dtypes(include=[np.number]).drop(columns=['target_label'], errors='ignore').fillna(0)
    
    # Affichage de la répartition pour diagnostic
    print(f"📊 Entraînement : {len(y_train)} échantillons ({sum(y_train==1)} Planètes / {sum(y_train==0)} Bruit)")
    print(f"📊 Test : {len(y_test)} échantillons ({sum(y_test==1)} Planètes / {sum(y_test==0)} Bruit)")

    if sum(y_train==0) == 0:
        print("⚠️ ALERTE : Aucun échantillon de bruit (classe 0) dans l'entraînement. L'IA ne peut pas apprendre à comparer.")

    # 2. Sélection des caractéristiques (Feature Selection)
    print("🔍 Sélection des caractéristiques pertinentes...")
    selector_clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    selector_clf.fit(X_train, y_train)
    
    # On augmente le seuil pour ne garder que le top des caractéristiques
    selection = SelectFromModel(selector_clf, threshold="1.25*median", prefit=True)
    selected_features = X_train.columns[selection.get_support()].tolist()
    
    X_train_v2 = selection.transform(X_train)
    X_test_v2 = selection.transform(X_test)
    
    print(f"✨ {len(selected_features)} caractéristiques critiques conservées.")

    # 3. Apprentissage du Modèle Final
    print("🧠 Entraînement du modèle XGBoost final...")
    
    # Calcul dynamique du poids (Balance)
    n_neg = len(y_train[y_train == 0])
    n_pos = len(y_train[y_train == 1])
    # On donne beaucoup plus de poids à la classe minoritaire (le bruit) pour forcer l'IA à y faire attention
    weight = n_neg / n_pos if n_pos > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators=500,        # Plus d'arbres pour capter les nuances
        learning_rate=0.03,      # Apprentissage plus lent
        max_depth=4,             # Moins profond pour éviter de "mémoriser" (Overfitting)
        scale_pos_weight=weight,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train_v2, y_train)

    # 4. Évaluation sur le Test Set
    print("\n📈 RÉSULTATS SUR LE TEST SET :")
    y_pred = model.predict(X_test_v2)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Précision globale (Accuracy) : {acc:.2%}")
    print("\nMatrice de Confusion :")
    # Rappel : [ [Vrais Négatifs, Faux Positifs], [Faux Négatifs, Vrais Positifs] ]
    print(confusion_matrix(y_test, y_pred))
    print("\nRapport de Classification :")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 5. Sauvegardes
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(f"{model_dir}/exoplanet_model.json")
    
    with open(f"{model_dir}/selected_features.json", "w") as f:
        json.dump(selected_features, f)
        
    print(f"\n✅ IA sauvegardée dans /{model_dir}")

if __name__ == "__main__":
    train_exoplanet_model()