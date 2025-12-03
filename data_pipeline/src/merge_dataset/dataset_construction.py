import json
import requests
import pandas as pd
from time import sleep
from tqdm import tqdm
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_URL = os.getenv("FRED_URL")
FIPS_URL = os.getenv("FIPS_URL")
CENSUS_URL = os.getenv("CENSUS_URL")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
BLS_URL = os.getenv("BLS_URL")

session = requests.Session()  # reuse connections

def safe_fetch(fn, *args, max_retries=5, base_delay=1):
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return fn(*args)
        except requests.HTTPError as e:
            status = e.response.status_code
            if status == 429:
                retry_after = e.response.headers.get("Retry-After")
                sleep_seconds = int(retry_after) if retry_after else delay
                sleep(sleep_seconds)
                delay *= 2
                continue
            elif 500 <= status < 600:
                sleep(delay)
                delay *= 2
                continue
            else:
                raise
        except Exception:
            sleep(delay)
            delay *= 2
    return None

def fetch_one_county(fips):
    # check disk cache first (implement your own cache)
    # call census/fred/bls inside here using 'session'
    return get_economic_indicators(fips)

def fetch_all_counties(fips_list, max_workers=6):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(safe_fetch, fetch_one_county, f): f for f in fips_list}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching counties"):
            fips = futures[fut]
            res = fut.result()
            results[fips] = res
    return results

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

def fetch_census_series(series_id, fips):
    # https://api.census.gov/data/2023/acs/acs5/profile?get=DP03_0065E&for=county:083&in=state:06&key=5e02d264229f90df7cd40bcdafaa9a529888a2a5
    if series_id == "SAEPOVRTALL_PT":
        census_url = CENSUS_URL+f"/timeseries/poverty/saipe?get={series_id}"
        parameters = f"&for=county:{fips[2:]}&in=state:{fips[:2]}&time=2023&key={CENSUS_API_KEY}"
        census_url += parameters
    else:
        census_url = CENSUS_URL+f"/2023/acs/acs5/profile?get={series_id}"
        parameters = f"&for=county:{fips[2:]}&in=state:{fips[:2]}&key={CENSUS_API_KEY}"
        census_url += parameters
    try:
            r = requests.get(census_url, timeout=10)
            r.raise_for_status()

            data = r.json()

            headers = data[0]
            values = data[1]

            idx = headers.index(f"{series_id}")
            poverty_rate = float(values[idx])

            return poverty_rate

    except Exception as e:
            print(f"Error fetching SAIPE poverty rate for FIPS {fips}: {e}")
            return None

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

def fetch_bls_series(fips):
    bls_url = BLS_URL+f"{fips}.csv"
    df = pd.read_csv(bls_url)

    first_data_row = df.iloc[0]

    avg_weekly_wage = first_data_row["avg_wkly_wage"]
    return avg_weekly_wage


def get_economic_indicators(fips):
    return {
        "pcpi": fetch_fred_series(f"PCPI{fips}"),
        "poverty_rate": fetch_census_series("SAEPOVRTALL_PT", fips),  # sometimes PEAD
        "median_household_income": fetch_census_series("DP03_0065E", fips),
        "unemployment_rate": fetch_census_series("DP03_0009PE", fips),
        "avg_weekly_wages": fetch_bls_series(fips)  # may require fallback
    }

# def build_county_econ_cache(yelp_json_path, max_entries):
    
#     fips_cache = {}
#     with open(yelp_json_path, "r", encoding="utf-8") as f:
#         for idx, line in enumerate(tqdm(f, total=max_entries, desc="Businesses")):
#             if idx >= max_entries:
#                 break

#             biz = json.loads(line.strip())

#             lat = biz.get("latitude")
#             lon = biz.get("longitude")
#             if lat is None or lon is None:
#                 continue

#             fips = get_fips(lat, lon)
#             if fips and fips not in fips_cache:
#                 econ = safe_fetch(get_economic_indicators, fips)
#                 fips_cache[fips] = econ

#     return fips_cache

def build_county_econ_cache(yelp_json_path, max_entries):

    fips_cache = {}          # fips → econ indicators
    biz_fips_map = {}        # business_id → fips

    with open(yelp_json_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(tqdm(f, total=max_entries, desc="Businesses")):
            if idx >= max_entries:
                break

            biz = json.loads(line)

            lat = biz.get("latitude")
            lon = biz.get("longitude")
            business_id = biz.get("business_id")

            if lat is None or lon is None:
                continue

            # ONLY CALL ONCE
            fips = get_fips(lat, lon)

            if fips:
                biz_fips_map[business_id] = fips

                if fips not in fips_cache:
                    econ = safe_fetch(get_economic_indicators, fips)
                    fips_cache[fips] = econ

    return fips_cache, biz_fips_map

# def build_merged_dataset(yelp_json_path, fips_cache, max_entries):
#     rows = []

#     with open(yelp_json_path, "r", encoding="utf-8") as f:
#         for idx, line in enumerate(f):
#             if idx >= max_entries:
#                 break

#             biz = json.loads(line.strip())

#             lat = biz.get("latitude")
#             lon = biz.get("longitude")
#             fips = get_fips(lat, lon)

#             if not fips or fips not in fips_cache:
#                 continue

#             rows.append({
#                 "rating": biz.get("stars"),
#                 "is_open": biz.get("is_open"),
#                 "latitude": lat,
#                 "longitude": lon,
#                 "fips": fips,
#                 **fips_cache[fips]
#             })

#     return pd.DataFrame(rows)

def build_merged_dataset(yelp_json_path, fips_cache, biz_fips_map, max_entries):
    rows = []

    with open(yelp_json_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_entries:
                break

            biz = json.loads(line)
            business_id = biz.get("business_id")

            # USE CACHED FIPS (NO API CALL)
            fips = biz_fips_map.get(business_id)
            if not fips:
                continue

            econ = fips_cache.get(fips)
            if not econ:
                continue

            rows.append({
                "business_id": business_id,
                "rating": biz.get("stars"),
                "is_open": biz.get("is_open"),
                "latitude": biz.get("latitude"),
                "longitude": biz.get("longitude"),
                "fips": fips,
                **econ
            })

    return pd.DataFrame(rows)

# ___________________________________________________________________________
if __name__ == "__main__":
    yelp_path = "../raw_data/yelp_academic_dataset_business.json"
    max_entries = 1000

    print("Building FIPS → Economic Indicator Cache...")

    # unpack both values
    fips_cache, biz_fips_map = build_county_econ_cache(yelp_path, max_entries)

    print(f"\nCached counties: {len(fips_cache)}")
    print(f"Businesses with FIPS assigned: {len(biz_fips_map)}")

    print("Building merged dataset...")
    final_df = build_merged_dataset(
        yelp_path,
        fips_cache,
        biz_fips_map,
        max_entries
    )

    final_df = final_df.dropna(subset=[
        "rating", "is_open", "pcpi", "poverty_rate",
        "median_household_income", "unemployment_rate",
        "avg_weekly_wages"
    ])

    open_count = (final_df["is_open"] == 1).sum()
    closed_count = (final_df["is_open"] == 0).sum()
    total = len(final_df)

    print(f"Open: {open_count} ({open_count/total:.1%})")
    print(f"Closed: {closed_count} ({closed_count/total:.1%})")

    final_df.to_parquet("../dataset/yelp_fred_merged.parquet", index=False)
    print("Dataset saved to dataset/yelp_fred_merged.parquet")

