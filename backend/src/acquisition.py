import os

try:
    import lightkurve as lk
except ImportError:
    print("Erreur : La bibliothèque 'lightkurve' est manquante.")
    print("Veuillez l'installer avec : pip install lightkurve")
    raise

def fetch_lightcurve(target_id, mission='Kepler', author='Kepler'):
    """
    Recherche et télécharge la courbe de lumière pour une cible donnée.
    """
    print(f"--- Recherche de données pour : {target_id} ---")
    
    try:
        search_result = lk.search_lightcurve(target_id, mission=mission, author=author)
    except Exception as e:
        print(f"[!] Erreur lors de la recherche MAST : {e}")
        return None
    
    if len(search_result) == 0:
        print(f"[/] Aucune donnée trouvée pour {target_id} avec l'auteur {author}.")
        return None
    
    print(f"[+] {len(search_result)} fichiers trouvés. Téléchargement en cours...")
    
    lc_collection = search_result.download_all()
    
    if lc_collection is None:
        print("[!] Erreur lors du téléchargement.")
        return None
        
    stitched_lc = lc_collection.stitch()
    
    print(f"[OK] Courbe de lumière récupérée ({len(stitched_lc)} points).")
    return stitched_lc

def save_raw_data(lc, folder="data/raw/"):
    """
    Sauvegarde la courbe de lumière au format FITS pour une utilisation hors-ligne.
    """
    # Création du dossier si absent (le exist_ok évite les erreurs si déjà là)
    os.makedirs(folder, exist_ok=True)
        
    # FIX: On convertit en string car targetid peut être un entier (ex: 11442793)
    target_name = str(lc.targetid).replace(" ", "_")
    file_path = os.path.join(folder, f"{target_name}_raw.fits")
    
    lc.to_fits(file_path, overwrite=True)
    print(f"[FILE] Sauvegardé sous : {file_path}")
    return file_path