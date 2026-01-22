import lightkurve as lk

def fetch_lightcurve(target_id, mission="Kepler", author=None):
    """
    Télécharge la courbe de lumière depuis les archives de la NASA.
    Supporte Kepler et TESS.
    """
    try:
        # Recherche des données
        search = lk.search_lightcurve(target_id, mission=mission, author=author)
        if len(search) == 0 and author: 
            # Tentative de secours sans filtre d'auteur
            search = lk.search_lightcurve(target_id, mission=mission)
        
        if len(search) == 0:
            return None
            
        return search[0].download()
    except Exception as e:
        print(f"Erreur d'acquisition pour {target_id}: {e}")
        return None