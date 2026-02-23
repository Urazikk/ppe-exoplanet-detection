import lightkurve as lk
import os


def fetch_lightcurve(target_id, mission="Kepler", author=None):
    """
    Recupere une courbe de lumiere depuis les archives MAST (NASA).
    """
    if author is None:
        author = "Kepler" if mission == "Kepler" else "SPOC"

    try:
        search = lk.search_lightcurve(target_id, mission=mission, author=author)
        if len(search) == 0:
            print(f"   Aucune donnee pour {target_id}")
            return None

        lc = search.download_all().stitch()
        return lc
    except Exception as e:
        print(f"   Erreur lightkurve pour {target_id}: {e}")
        return None