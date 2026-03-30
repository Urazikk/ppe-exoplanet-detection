import sys
import traceback
from src.p01_acquisition import fetch_lightcurve
from src.p02_preprocessing import clean_and_flatten, fold_lightcurve, get_period_hint

try:
    print("Fetching LC...")
    lc_raw = fetch_lightcurve("Kepler-10", mission="Kepler")
    print("Flattening...")
    lc_flat = clean_and_flatten(lc_raw, quality="fast")
    print("BLS...")
    period, bls_stats = get_period_hint(lc_flat)
    print("Folding...")
    lc_folded = fold_lightcurve(lc_flat, period=period)
    print("Done")
except Exception as e:
    traceback.print_exc()
