from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.data_loader import load_global_temperature,load_alternative_temperature
root=Path(__file__).resolve().parents[1]; data=root/"data"; data.mkdir(exist_ok=True)
try: df=load_global_temperature(); src="NASA GISTEMP v4"
except Exception: df=load_alternative_temperature(); src="NOAA PSL mirror of NASA series"
df.to_csv(data/"global_temperature_latest.csv")
(data/"source_metadata.txt").write_text(f"source={src}\\nrows={len(df)}\\nlatest={df.index.max().isoformat()}\\n",encoding="utf-8")
print(f"Saved {len(df)} observations from {src}")
