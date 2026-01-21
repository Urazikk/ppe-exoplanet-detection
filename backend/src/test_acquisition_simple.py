from acquisition import fetch_lightcurve, save_raw_data

# On choisit une cible célèbre : Kepler-90
TARGET = "Kepler-90"

print(f"--- TEST D'ACQUISITION POUR {TARGET} ---")

# 1. Téléchargement
lc = fetch_lightcurve(TARGET)

if lc is not None:
    print(f"✅ Succès ! Données reçues pour {lc.targetid}")
    print(f"📊 Nombre de points de mesure : {len(lc)}")
    
    # 2. Sauvegarde locale
    # On utilise "data/raw/" car on lance le script depuis le dossier backend
    # Cela créera le dossier dans backend/data/raw/
    path = save_raw_data(lc, folder="data/raw/")
    print(f"📁 Fichier sauvegardé ici : {path}")
else:
    print("❌ Échec : Aucune donnée reçue. Vérifiez votre connexion.")