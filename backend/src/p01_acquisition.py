import json
import os
import re
import lightkurve as lk
import numpy as np

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "lightkurve_training")


def _load_from_local_cache(target_id):
    """
    Tente de reconstruire un LightCurve depuis le cache JSON local.
    Retourne un LightCurve ou None si absent/erreur.
    """
    m = re.search(r'\d+', target_id)
    if not m:
        return None
    kepid = m.group(0)
    path = os.path.join(_CACHE_DIR, f"star_{kepid}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        if d.get("status") != "ok":
            return None
        time = np.array(d["time"], dtype=float)
        flux = np.array(d["flux"], dtype=float)
        lc = lk.LightCurve(time=time, flux=flux)
        print(f"   [Acquisition] Cache local OK pour {target_id} ({len(lc)} points)")
        return lc
    except Exception as e:
        print(f"   [Acquisition] Erreur lecture cache local {target_id} : {e}")
        return None


def fetch_lightcurve(target_id, mission="Kepler", author=None):
    """
    Récupère une courbe de lumière : cache local en priorité, puis MAST.
    Limite à 2 fichiers max pour éviter les timeouts.
    """
    lc = _load_from_local_cache(target_id)
    if lc is not None:
        return lc

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
