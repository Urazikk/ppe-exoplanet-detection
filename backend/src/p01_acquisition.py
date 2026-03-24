import lightkurve as lk
import numpy as np


def fetch_lightcurve(target_id, mission="Kepler", author=None):
    """
    Récupère une courbe de lumière depuis MAST avec 3 stratégies de fallback.
    Limite à 2 fichiers max pour éviter les timeouts.
    """
    strategies = [
        # Stratégie 1 : sans filtre author, limité à 2 résultats
        dict(target=target_id, mission=mission),
        # Stratégie 2 : sans filtre mission du tout
        dict(target=target_id),
        # Stratégie 3 : avec exptime long uniquement
        dict(target=target_id, exptime="long"),
    ]

    for i, kwargs in enumerate(strategies):
        try:
            print(f"   [Acquisition] Stratégie {i+1}/3 : {kwargs}")
            search = lk.search_lightcurve(**kwargs)

            if len(search) == 0:
                print(f"   [Acquisition] 0 résultat pour stratégie {i+1}")
                continue

            print(f"   [Acquisition] {len(search)} fichier(s) trouvé(s), téléchargement des 2 premiers...")

            # Limite à 2 fichiers maximum
            subset = search[:2]
            collection = subset.download_all()

            if collection is None or len(collection) == 0:
                print(f"   [Acquisition] Download vide pour stratégie {i+1}")
                continue

            lc = collection.stitch()
            print(f"   [Acquisition] OK — {len(lc)} points")
            return lc

        except Exception as e:
            print(f"   [Acquisition] Erreur stratégie {i+1} : {e}")
            continue

    print(f"   [Acquisition] Toutes les stratégies ont échoué pour {target_id}")
    return None
