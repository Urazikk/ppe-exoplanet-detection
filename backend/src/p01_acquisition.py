import lightkurve as lk
import os

def fetch_lightcurve(target_id, mission="Kepler", author=None):
    """
    Récupère une courbe de lumière depuis les archives MAST (NASA).
    C'est la fonction attendue par 01_generate_dataset.py
    """
    if author is None:
        author = "Kepler" if mission == "Kepler" else "SPOC"
        
    try:
        # Recherche de la courbe de lumière
        search = lk.search_lightcurve(target_id, mission=mission, author=author)
        if len(search) == 0:
            print(f"   ⚠️ Aucune donnée pour {target_id}")
            return None
            
        # Téléchargement et fusion des secteurs/quarts (stitch)
        lc = search.download_all().stitch()
        return lc
    except Exception as e:
        print(f"   ⚠️ Erreur lightkurve pour {target_id}: {e}")
        return None