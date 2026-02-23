import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
import sys
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def train_exoplanet_model():
    """
    Entraine l'IA sur le dataset d'entrainement et valide sur le set de test independant.
    """
    train_path = "data/processed/training_dataset.csv"
    test_path = "data/processed/test_dataset.csv"
    model_dir = "models"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Erreur : Les datasets CSV sont introuvables. Relancez le generateur.")
        return

    print("Chargement des donnees CSV...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # 1. Preparation robuste des donnees
    y_train = df_train['target_label']
    y_test = df_test['target_label']

    X_train = df_train.select_dtypes(include=[np.number]).drop(columns=['target_label'], errors='ignore').fillna(0)
    X_test = df_test.select_dtypes(include=[np.number]).drop(columns=['target_label'], errors='ignore').fillna(0)

    print(f"Entrainement : {len(y_train)} echantillons ({sum(y_train==1)} Planetes / {sum(y_train==0)} Bruit)")
    print(f"Test : {len(y_test)} echantillons ({sum(y_test==1)} Planetes / {sum(y_test==0)} Bruit)")

    if sum(y_train == 0) == 0:
        print("ALERTE : Aucun echantillon de bruit dans l'entrainement.")

    # 2. Selection des caracteristiques
    print("\nSelection des caracteristiques pertinentes...")
    selector_clf = xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42)
    selector_clf.fit(X_train, y_train)

    selection = SelectFromModel(selector_clf, threshold="1.25*median", prefit=True)
    selected_features = X_train.columns[selection.get_support()].tolist()

    X_train_v2 = selection.transform(X_train)
    X_test_v2 = selection.transform(X_test)

    print(f"{len(selected_features)} caracteristiques conservees.")

    # 3. Configuration du modele
    n_neg = len(y_train[y_train == 0])
    n_pos = len(y_train[y_train == 1])
    weight = n_neg / n_pos if n_pos > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        scale_pos_weight=weight,
        eval_metric='logloss',
        random_state=42
    )

    # 4. Validation croisee sur le train set (5 folds)
    print("\nValidation croisee (5-fold) sur le train set...")
    cv_scores = cross_val_score(model, X_train_v2, y_train, cv=5, scoring='accuracy')
    print(f"Scores par fold : {[f'{s:.2%}' for s in cv_scores]}")
    print(f"Moyenne : {cv_scores.mean():.2%} (+/- {cv_scores.std():.2%})")

    cv_f1 = cross_val_score(model, X_train_v2, y_train, cv=5, scoring='f1')
    print(f"F1-Score moyen  : {cv_f1.mean():.2%} (+/- {cv_f1.std():.2%})")

    # 5. Entrainement du modele final (sur tout le train set)
    print("\nEntrainement du modele XGBoost final...")
    model.fit(X_train_v2, y_train)

    # 6. Evaluation sur le Test Set
    print("\nRESULTATS SUR LE TEST SET :")
    y_pred = model.predict(X_test_v2)
    acc = accuracy_score(y_test, y_pred)

    print(f"Precision globale (Accuracy) : {acc:.2%}")
    print(f"\nMatrice de Confusion :")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nRapport de Classification :")
    print(classification_report(y_test, y_pred, zero_division=0))

    if len(y_test) < 10:
        print(f"NOTE : Le test set ne contient que {len(y_test)} echantillons.")
        print(f"Les metriques de validation croisee (5-fold) sont plus fiables.")

    # 7. Sauvegardes
    os.makedirs(model_dir, exist_ok=True)
    model.save_model(f"{model_dir}/exoplanet_model.json")

    with open(f"{model_dir}/selected_features.json", "w") as f:
        json.dump(selected_features, f)

    # Sauvegarde des metriques pour la documentation
    metrics = {
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "cv_f1_mean": round(float(cv_f1.mean()), 4),
        "cv_f1_std": round(float(cv_f1.std()), 4),
        "test_accuracy": round(float(acc), 4),
        "test_size": len(y_test),
        "train_size": len(y_train),
        "n_features_selected": len(selected_features),
    }
    with open(f"{model_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nIA sauvegardee dans /{model_dir}")
    print(f"Metriques sauvegardees dans /{model_dir}/metrics.json")


if __name__ == "__main__":
    train_exoplanet_model()