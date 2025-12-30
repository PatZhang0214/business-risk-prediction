import json
import requests
import pandas as pd
from time import sleep
from tqdm import tqdm
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import log
from collections import Counter
from datetime import datetime

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_URL = os.getenv("FRED_URL")
FIPS_URL = os.getenv("FIPS_URL")
CENSUS_URL = os.getenv("CENSUS_URL")
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
BLS_URL = os.getenv("BLS_URL")

session = requests.Session()  # reuse connections

def get_top_k_categories(yelp_json_path, max_entries, k=75):
    """
    Returns a list of the top-K most frequent Yelp categories.
    """
    counter = Counter()

    with open(yelp_json_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_entries:
                break

            biz = json.loads(line)
            cats = biz.get("categories")
            if not cats:
                continue

            for c in cats.split(", "):
                counter[c] += 1

    top_categories = [c for c, _ in counter.most_common(k)]
    return top_categories

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
                "review_count": obj.get("review_count"),
                "categories": obj.get('categories'),
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
    if series_id == "SAEPOVRTALL_PT":
        census_url = CENSUS_URL+f"/timeseries/poverty/saipe?get={series_id}"
        parameters = f"&for=county:{fips[2:]}&in=state:{fips[:2]}&time=2018&key={CENSUS_API_KEY}"
        census_url += parameters
    else:
        census_url = CENSUS_URL+f"/2018/acs/acs5/profile?get={series_id}"
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

# def fetch_fred_series(series_id):
#     url = f"{FRED_URL}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
#     try:
#         r = requests.get(url, timeout=10)
#         data = r.json()

#         observations = data.get("observations", [])
#         if not observations:
#             return None

#         # Return the most recent valid numeric value
#         for obs in reversed(observations):
#             value = obs.get("value")
#             if value not in ["", ".", None]:
#                 try:
#                     return float(value)
#                 except:
#                     return None

#     except Exception as e:
#         print(f"FRED error for {series_id}: {e}")
#         return None

#     return None

def fetch_fred_series(series_id, target_date="2018-01-01"):
    url = f"{FRED_URL}?series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()

        observations = data.get("observations", [])
        if not observations:
            return None

        # Find the observation matching the target date
        for obs in observations:
            if obs.get("date") == target_date:
                value = obs.get("value")
                if value not in ["", ".", None]:
                    try:
                        return float(value)
                    except:
                        return None
        
        # If target date not found, return None
        return None

    except Exception as e:
        print(f"FRED error for {series_id}: {e}")
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

def build_business_age_map(checkin_json_path, max_entries=None):
    age_map = {}

    with open(checkin_json_path, "r", encoding="utf-8") as f:
        iterator = enumerate(f)

        if max_entries:
            iterator = tqdm(
                iterator,
                total=max_entries,
                desc="Processing check-ins"
            )
        else:
            iterator = tqdm(
                iterator,
                desc="Processing check-ins"
            )

        for idx, line in iterator:
            if max_entries and idx >= max_entries:
                break

            obj = json.loads(line)
            business_id = obj.get("business_id")
            dates_str = obj.get("date")

            if not dates_str:
                continue

            try:
                dates = [
                    datetime.strptime(d.strip(), "%Y-%m-%d %H:%M:%S")
                    for d in dates_str.split(",")
                ]
            except Exception:
                continue

            first = min(dates)
            last = max(dates)

            age_map[business_id] = {
                "years_in_business": (last - first).days / 365.25,
                "num_checkins": len(dates),
                "has_checkin": 1
            }

    return age_map
# def build_merged_dataset(yelp_json_path, fips_cache, biz_fips_map, max_entries):
#     rows = []

#     with open(yelp_json_path, "r", encoding="utf-8") as f:
#         for idx, line in enumerate(f):
#             if idx >= max_entries:
#                 break

#             biz = json.loads(line)
#             business_id = biz.get("business_id")

#             # USE CACHED FIPS (NO API CALL)
#             fips = biz_fips_map.get(business_id)
#             if not fips:
#                 continue

#             econ = fips_cache.get(fips)
#             if not econ:
#                 continue

#             review_count = biz.get("review_count")
#             log_review_count = log(review_count + 1)
#             rating = biz.get("stars")
#             rating_x_reviews = rating * log_review_count


#             rows.append({
#                 "business_id": business_id,
#                 "rating_x_reviews": rating_x_reviews,
#                 "is_open": biz.get("is_open"),
#                 "latitude": biz.get("latitude"),
#                 "longitude": biz.get("longitude"),
#                 "fips": fips,
#                 **econ
#             })

#     return pd.DataFrame(rows)

def build_merged_dataset(
    yelp_json_path,
    fips_cache,
    biz_fips_map,
    max_entries,
    top_categories,
    business_age_map
):
    rows = []

    for c in top_categories:
        if "," in c:
            raise ValueError(f"Comma in category name: {c}")

    with open(yelp_json_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_entries:
                break

            biz = json.loads(line)
            business_id = biz.get("business_id")

            fips = biz_fips_map.get(business_id)
            if not fips:
                continue

            econ = fips_cache.get(fips)
            if not econ:
                continue

            review_count = biz.get("review_count", 0)
            rating = biz.get("stars")

            if rating is None:
                continue

            log_review_count = log(review_count + 1)
            rating_x_reviews = rating * log_review_count

            # --- CATEGORY FEATURES ---
            cats = biz.get("categories") or ""
            cat_set = set(cats.split(", "))

            cat_features = {
                f"cat_{c}": int(c in cat_set)
                for c in top_categories
            }
            age_info = business_age_map.get(business_id)

            years_in_business = age_info["years_in_business"] if age_info else 0.0
            num_checkins = age_info["num_checkins"] if age_info else 0
            has_checkin = age_info["has_checkin"] if age_info else 0

            rows.append({
                "business_id": business_id,
                "rating_x_reviews": rating_x_reviews,
                "review_count": review_count,
                "num_categories": len(cat_set),
                "years_in_business": years_in_business,
                "num_checkins": num_checkins,
                "has_checkin": has_checkin,
                "is_open": biz.get("is_open"),
                "latitude": biz.get("latitude"),
                "longitude": biz.get("longitude"),
                "fips": fips,
                **econ,
                **cat_features
            })

    return pd.DataFrame(rows)

# ___________________________________________________________________________
# if __name__ == "__main__":
#     yelp_path = "../raw_data/yelp_academic_dataset_business.json"
#     max_entries = 100

#     print("Building FIPS → Economic Indicator Cache...")

#     # unpack both values
#     fips_cache, biz_fips_map = build_county_econ_cache(yelp_path, max_entries)

#     print(f"\nCached counties: {len(fips_cache)}")
#     print(f"Businesses with FIPS assigned: {len(biz_fips_map)}")

#     print("Building merged dataset...")
#     final_df = build_merged_dataset(
#         yelp_path,
#         fips_cache,
#         biz_fips_map,
#         max_entries
#     )

#     final_df = final_df.dropna(subset=[
#         "is_open", "pcpi", "poverty_rate",
#         "median_household_income", "unemployment_rate",
#         "avg_weekly_wages", "rating_x_reviews"
#     ])

#     open_count = (final_df["is_open"] == 1).sum()
#     closed_count = (final_df["is_open"] == 0).sum()
#     total = len(final_df)

#     print(f"Open: {open_count} ({open_count/total:.1%})")
#     print(f"Closed: {closed_count} ({closed_count/total:.1%})")

#     final_df.to_parquet("../dataset/yelp_fred_merged.parquet", index=False)
#     print("Dataset saved to dataset/yelp_fred_merged.parquet")
if __name__ == "__main__":
    yelp_path = "../raw_data/yelp_academic_dataset_business.json"
    checkin_path = "../raw_data/yelp_academic_dataset_checkin.json"
    max_entries = 17500
    TOP_K = 50

    print("Extracting top categories...")
    top_categories = get_top_k_categories(yelp_path, max_entries, TOP_K)

    print(f"Using {len(top_categories)} category features")

    print("Building FIPS → Economic Indicator Cache...")
    fips_cache, biz_fips_map = build_county_econ_cache(yelp_path, max_entries)

    print("Building business age map from check-ins...")
    business_age_map = build_business_age_map(checkin_path)
    print(f"Businesses with check-in age: {len(business_age_map)}")

    print("Building merged dataset...")
    final_df = build_merged_dataset(
        yelp_path,
        fips_cache,
        biz_fips_map,
        max_entries,
        top_categories,
        business_age_map
    )

    final_df = final_df.dropna(subset=[
        "is_open", "pcpi", "poverty_rate",
        "median_household_income", "unemployment_rate",
        "avg_weekly_wages", "rating_x_reviews"
    ])

    open_count = (final_df["is_open"] == 1).sum()
    closed_count = (final_df["is_open"] == 0).sum()
    total = len(final_df)

    print(f"Open: {open_count} ({open_count/total:.1%})")
    print(f"Closed: {closed_count} ({closed_count/total:.1%})")

    final_df.to_parquet("../dataset/yelp_fred_merged.parquet", index=False)
    print("Dataset saved.")
