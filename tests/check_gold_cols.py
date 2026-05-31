import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from src.common.minio_io import get_client, load_gold_features

client = get_client()
loc = 'ha_noi'
year, month, day = 2026, 5, 24

df = load_gold_features(client, loc, year, month, day)
if df is None:
    print('Gold file not found')
else:
    print('Columns:', df.columns.tolist())
    print(df.head(3))
