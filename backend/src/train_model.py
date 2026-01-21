import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

def train_exoplanet_model():
    dataset_path = "data/processed/training_dataset.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Erreur : Le fichier {dataset_path} n'existe pas encore.")
        return

    print("--- 🧠 ENTRAÎNEMENT DU MODÈLE XGBOOST ---")
    
    # 1. Chargement des données
    df = pd.read_csv(dataset_path)
    
    # 2. Nettoyage des colonnes non-numériques (IDs, etc.)
    # On garde 'target_label' de côté
    y = df['target_label']
    X = df.drop(columns=['target_label'])
    X = X.select_dtypes(include=[np.number])
    
    # TSFRESH peut générer des colonnes avec des NaNs (ex: variance d'une constante)
    # On remplace les NaNs par 0 pour ne pas bloquer XGBoost
    X = X.fillna(0)
    
    print(f"[i] Dataset chargé : {X.shape[0]} échantillons, {X.shape[1]} caractéristiques.")

    # 3. Séparation Entraînement / Test (80% / 20%)
    # Avec 19 échantillons, le test set ne contient que 4 lignes !
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Configuration et Entraînement
    # Note : 'use_label_encoder' est supprimé pour éviter le UserWarning
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)

    # 5. Évaluation
    y_pred = model.predict(X_test)
    print("\n📈 Performances du modèle :")
    print(f"Précision globale : {accuracy_score(y_test, y_pred) * 100:.2f}%")
    
    print("\nRapport de classification :")
    # zero_division=0 permet d'éviter les gros messages d'erreur si une classe n'est pas trouvée
    print(classification_report(y_test, y_pred, zero_division=0))

    if len(df) < 50:
        print("\n⚠️ Note : Le dataset est très petit. Les scores de précision ne sont pas encore")
        print("significatifs. L'IA a besoin de plus d'exemples de 'Faux Positifs' (Label 0)")
        print("pour apprendre à faire la différence.")

    # 6. Sauvegarde du modèle
    os.makedirs("models/", exist_ok=True)
    model_path = "models/exoplanet_model.json"
    model.save_model(model_path)
    print(f"\n[FILE] Modèle sauvegardé avec succès : {model_path}")

if __name__ == "__main__":
    train_exoplanet_model()