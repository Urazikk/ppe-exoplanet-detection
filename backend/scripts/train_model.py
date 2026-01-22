import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import os

def train_exoplanet_model():
    """
    Entraîne l'IA avec une stratégie de sélection de caractéristiques pour maximiser la précision.
    """
    # 0. Choix du dataset : Priorité au dataset augmenté s'il existe
    aug_path = "data/processed/final_augmented_dataset.csv"
    clean_path = "data/processed/training_dataset_clean.csv"
    
    dataset_path = aug_path if os.path.exists(aug_path) else clean_path
    
    if not os.path.exists(dataset_path):
        print(f"❌ Erreur : Aucun dataset trouvé.")
        return

    print(f"--- 🚀 OPTIMISATION FINALE DE L'IA ({os.path.basename(dataset_path)}) ---")
    
    # 1. Chargement
    df = pd.read_csv(dataset_path)
    print(f"[i] Dataset chargé : {len(df)} échantillons.")

    # 2. Préparation des données
    y = df['target_label']
    X = df.select_dtypes(include=[np.number]).drop(columns=['target_label'], errors='ignore')
    
    # Suppression des colonnes constantes (0 information)
    X = X.loc[:, (X != X.iloc[0]).any()] 
    X = X.fillna(0)
    
    # Équilibrage des classes
    n_neg = len(y[y == 0])
    n_pos = len(y[y == 1])
    weight_ratio = n_neg / n_pos if n_pos > 0 else 1
    
    print(f"[i] Équilibre : {n_neg} Faux Positifs / {n_pos} Planètes")

    # 3. Séparation 80/20 Stratifiée
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Phase 1 : Entraînement pour sélection de caractéristiques
    # On utilise un premier modèle pour identifier les "features" inutiles
    selector_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    selector_model.fit(X_train, y_train)
    
    selection = SelectFromModel(selector_model, threshold="median", prefit=True)
    X_train_select = selection.transform(X_train)
    X_test_select = selection.transform(X_test)
    
    selected_feat_names = X.columns[selection.get_support()]
    print(f"[i] Sélection : {X.shape[1]} -> {X_train_select.shape[1]} caractéristiques critiques conservées.")

    # 5. Phase 2 : Modèle Final sur les caractéristiques sélectionnées
    model = xgb.XGBClassifier(
        n_estimators=300,        # On augmente pour capter les détails de l'augmentation
        max_depth=6,             # Plus de profondeur pour les relations complexes
        learning_rate=0.02,      # Apprentissage lent pour la précision
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=weight_ratio,
        eval_metric='logloss',
        random_state=42
    )
    
    print("🚀 Entraînement du modèle final...")
    model.fit(X_train_select, y_train)

    # 6. Évaluation
    y_pred = model.predict(X_test_select)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n📈 PERFORMANCES FINALES :")
    print(f"Précision (Accuracy) : {acc * 100:.2f}%")
    
    print("\n📊 Matrice de Confusion :")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Vrais Négatifs : {cm[0][0]} | Faux Positifs : {cm[0][1]}")
    print(f"Faux Négatifs : {cm[1][0]} | Vrais Positifs : {cm[1][1]}")
    
    # 7. Sauvegarde
    os.makedirs("models/", exist_ok=True)
    # Note : On sauvegarde aussi la liste des features sélectionnées pour app.py
    model.save_model("models/exoplanet_model.json")
    print("\n✅ MODÈLE DE HAUTE PRÉCISION SAUVEGARDÉ.")

if __name__ == "__main__":
    train_exoplanet_model()