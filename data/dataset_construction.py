import json
import requests
import pandas as pd
from time import sleep
from tqdm import tqdm
import os
from dotenv import load_dotenv
import ijson

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_URL = os.getenv("FRED_URL")
FIPS_URL = os.getenv("FIPS_URL")

def load_yelp_businesses_datasets(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append({
                "business_id": obj.get("business_id"),
                "rating": obj.get("stars"),
                "is_open": obj.get("is_open"),
                "latitude": obj.get("latitude"),
                "logitude": obj.get("longitude"),
                "state": obj.get("state"),
                "city": obj.get("city")
            })
    return pd.DataFrame(rows)

def get_fips(lat, lon):
    fips_url = FIPS_URL+f"?x={lon}&y={lat}&benchmark=4&vintage=4&format=json"
    try:
        r = requests.get(fips_url, timeout=5)
        data = r.json()
        geo = data["result"]["geographies"]["Counties"][0]
        fips = geo["GEOID"]
        return fips
    except:
        return None

def assign_fips(df):
    fips_list = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        lat, lon = row["latitude"], row["longitude"]
        if pd.isna(lat) or pd.isna(lon):
            fips_list.append(None)
            continue
        fips = get_fips(lat, lon)
        fips_list.append(fips)
        sleep(0.10)
    df["county_fips"] = fips_list
    return df

def fetch_fred_series(series_id):
    url = f"{FRED_URL}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()

        observations = data.get("observations", [])
        if not observations:
            return None

        # Return the most recent valid numeric value
        for obs in reversed(observations):
            value = obs.get("value")
            if value not in ["", ".", None]:
                try:
                    return float(value)
                except:
                    return None

    except Exception as e:
        print(f"FRED error for {series_id}: {e}")
        return None

    return None


def get_economic_indicators(fips):
    return {
        "pcpi": fetch_fred_series(f"PCPI{fips}"),
        "disposable_income": fetch_fred_series(f"DPCPI{fips}"),
        "poverty_rate": fetch_fred_series(f"PEAA{fips}"),  # sometimes PEAD
        "median_household_income": fetch_fred_series(f"MEHOINUS{fips}"),
        "unemployment_rate": fetch_fred_series(f"LAUCN{fips}0000000003"),
        "avg_weekly_wages": fetch_fred_series(f"ENUC_{fips}_EW")  # may require fallback
    }

def build_county_econ_cache(yelp_json_path, max_entries=5):
    fips_cache = {}

    with open(yelp_json_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_entries:
                break

            biz = json.loads(line.strip())

            lat = biz.get("latitude")
            lon = biz.get("longitude")
            if lat is None or lon is None:
                continue

            fips = get_fips(lat, lon)
            if fips and fips not in fips_cache:
                econ = get_economic_indicators(fips)
                fips_cache[fips] = econ

            if idx % 500 == 0:
                print(f"Processed {idx} entries...", end="\r")

    return fips_cache

def build_merged_dataset(yelp_json_path, fips_cache, max_entries=5):
    rows = []

    with open(yelp_json_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_entries:
                break

            biz = json.loads(line.strip())

            lat = biz.get("latitude")
            lon = biz.get("longitude")
            fips = get_fips(lat, lon)

            if not fips or fips not in fips_cache:
                continue

            rows.append({
                "rating": biz.get("stars"),
                "is_open": biz.get("is_open"),
                "latitude": lat,
                "longitude": lon,
                "fips": fips,
                **fips_cache[fips]
            })

    return pd.DataFrame(rows)


# ___________________________________________________________________________
if __name__ == "__main__":
    yelp_path = "raw_data/yelp_academic_dataset_business.json"

    print("Building FIPS → Economic Indicator Cache...")
    fips_cache = build_county_econ_cache(yelp_path, max_entries=5)

    print(f"\nCache contains {len(fips_cache)} counties.")

    print("Building merged dataset...")
    final_df = build_merged_dataset(yelp_path, fips_cache, max_entries=5)

    final_df.to_parquet("yelp_fred_merged.parquet", index=False)
    print("Dataset saved to yelp_fred_merged.parquet")


