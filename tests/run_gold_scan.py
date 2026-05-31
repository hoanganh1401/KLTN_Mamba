from datetime import date, timedelta
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from src.Gold.gold_feature_engineering import run_feature_engineering

locations_path = "DataSet/locations.jsonl"
location_keys = ["ha_noi"]

start = date(2026,5,20)
end = date(2026,5,29)

d = start
while d <= end:
    print("\n--- Running for", d)
    try:
        run_feature_engineering(locations_path, d.strftime("%Y-%m-%d"), None, location_keys, disable_time_features=True)
    except Exception as e:
        print("Error:", e)
    d = d + timedelta(days=1)
print("Done")
