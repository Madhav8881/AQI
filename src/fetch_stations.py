"""
Phase 1: Fetch and inspect Delhi NCR monitoring stations from XKDR database.
"""
import os
import json
import urllib.request
import pandas as pd

API_KEY = os.environ.get("XKDR_API_KEY", "aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

def get_delhi_stations():
    """Fetch station metadata from XKDR and filter Delhi NCR stations with GPS coordinates."""
    url = "https://airquality.xkdr.org/v1/stations"
    req = urllib.request.Request(url, headers=HEADERS)
    print("Connecting to XKDR API to fetch station directory...")
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
    
    stations = res.get('data', [])
    delhi_stations = []
    for s in stations:
        lat = s.get('latitude')
        lon = s.get('longitude')
        name = (s.get('station_name') or '').lower()
        city = (s.get('city_name') or '').lower()
        state = (s.get('state_name') or '').lower()
        
        if lat is not None and lon is not None:
            if 'delhi' in name or 'delhi' in city or 'delhi' in state:
                delhi_stations.append({
                    'station_id': s.get('station_id'),
                    'station_name': s.get('station_name'),
                    'latitude': float(lat),
                    'longitude': float(lon),
                    'first_seen': s.get('first_seen'),
                    'last_seen': s.get('last_seen'),
                    'n_rows': s.get('n_rows'),
                    'parameters': s.get('parameters')
                })
                
    df_stations = pd.DataFrame(delhi_stations)
    # Sort by number of rows available
    df_stations = df_stations.sort_values(by='n_rows', ascending=False).reset_index(drop=True)
    out_file = os.path.join(RAW_DIR, "delhi_stations_metadata.csv")
    df_stations.to_csv(out_file, index=False)
    print(f"Found {len(df_stations)} Delhi stations with valid GPS coordinates.")
    print(f"Saved metadata to: {out_file}")
    print("\nTop 15 stations with highest data volume:")
    print(df_stations[['station_id', 'station_name', 'latitude', 'longitude', 'first_seen', 'last_seen', 'n_rows']].head(15).to_string())
    return df_stations

if __name__ == "__main__":
    get_delhi_stations()
