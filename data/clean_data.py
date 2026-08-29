"""
Clean the raw EM-DAT Philippines export into the model-ready shape:
one row per (typhoon event, affected province).

Decisions made and why:
- Restricted to Disaster Subtype == 'Tropical cyclone' only, since storm
  category is only a coherently-scaled predictor for storms (not floods).
- Storm category is derived from the Event Name text (e.g. "Typhoon 'Jangmi'"),
  because the Magnitude field (wind speed) is populated for too few rows to use.
  Only 3 levels are reliably distinguishable in the names actually present:
  Tropical Depression (1) < Tropical Storm (2) < Typhoon (4). No event names in
  this export used "Severe Tropical Storm" or "Super Typhoon" explicitly, so
  those levels don't appear even though the category is coded 1-5.
- Admin Units lists 0-27 locations per event, mixing province (adm2) and
  region (adm1) tags inconsistently. Events with zero adm2 entries are dropped
  (no usable province information). Events with multiple adm2 entries are
  EXPLODED into one row per province, with the event's total damage repeated
  across each -- EM-DAT does not sub-divide damage by province within an event,
  so this is an approximation, not a true per-province figure. This is the
  main named limitation of the analysis (see README).
"""

import json
import pandas as pd

df = pd.read_csv("emdat_raw.csv")

# 1. Restrict to tropical cyclones
storm = df[df["Disaster Subtype"] == "Tropical cyclone"].copy()

# 2. Derive storm category from Event Name
CATS = {
    "super typhoon": 5,
    "typhoon": 4,
    "severe tropical storm": 3,
    "tropical storm": 2,
    "tropical depression": 1,
}

def extract_category(name):
    if pd.isna(name):
        return None
    text = name.lower().replace("tyhoon", "typhoon").replace("strom", "storm")
    for label in sorted(CATS, key=len, reverse=True):  # longest match first
        if label in text:
            return CATS[label]
    return None

storm["storm_category"] = storm["Event Name"].apply(extract_category)

# 3. Parse Admin Units -> list of provinces (adm2_name)
def extract_provinces(admin_units_json):
    if pd.isna(admin_units_json):
        return []
    try:
        entries = json.loads(admin_units_json)
    except (json.JSONDecodeError, TypeError):
        return []
    return [e["adm2_name"] for e in entries if "adm2_name" in e]

storm["provinces"] = storm["Admin Units"].apply(extract_provinces)

# 4. Keep only usable rows
usable = storm[
    storm["storm_category"].notna()
    & storm["Total Damage ('000 US$)"].notna()
    & (storm["provinces"].apply(len) > 0)
].copy()

print(f"Tropical cyclone rows: {len(storm)}")
print(f"Usable rows (category + damage + >=1 province): {len(usable)}")

# 5. Explode: one row per (event, province)
usable = usable.rename(columns={"Total Damage ('000 US$)": "damage_000usd",
                                 "Total Deaths": "total_deaths",
                                 "Total Affected": "total_affected"})
exploded = usable.explode("provinces").rename(columns={"provinces": "province"})

keep_cols = [
    "DisNo.", "Event Name", "Start Year", "storm_category",
    "province", "damage_000usd", "total_deaths", "total_affected",
]
final = exploded[keep_cols].reset_index(drop=True)

print(f"Final exploded rows: {len(final)}")
print(f"Distinct provinces: {final['province'].nunique()}")
print()
print(final.head(10))

final.to_csv("emdat_philippines_clean.csv", index=False)
print("\nSaved to emdat_philippines_clean.csv")