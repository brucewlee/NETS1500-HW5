import pandas as pd
import pycountry
import re, os, sys
from unidecode import unidecode

# ------------------------------------------------------------
# 1.  Folder & file settings
# ------------------------------------------------------------
RAW_DIR  = "raw_data_disaster"      
OUT_DIR  = "output_disaster"       
os.makedirs(OUT_DIR, exist_ok=True)

EMDAT_PATH  = os.path.join(RAW_DIR,  "emdat.csv")
CITIES_PATH = os.path.join(RAW_DIR,  "worldcitiespop.csv")

# ------------------------------------------------------------
# 2.  Load the full CSVs
# ------------------------------------------------------------
emdat  = pd.read_csv(EMDAT_PATH,  low_memory=False)
cities = pd.read_csv(CITIES_PATH, low_memory=False,
                      dtype={'Region': str})        # Region codes need str

# ------------------------------------------------------------
# 3.  One‑line helpers
# ------------------------------------------------------------
def iso2_to_iso3(cc):
    """‘ad’ → ‘AND’  (returns None if lookup fails)"""
    try:
        return pycountry.countries.get(alpha_2=cc.upper()).alpha_3
    except Exception:
        return None

def clean_city(s):
    """
    Take EM‑DAT’s free‑text ‘Location’, keep the first token
    before ; / , or ‘and’, strip accents & punctuation, lower‑case.
    """
    if pd.isna(s):
        return None
    s = re.split(r'[;/,]', s)[0]           # first segment
    s = re.split(r'\band\b', s, flags=re.I)[0]
    s = unidecode(s)                       # no accents
    s = re.sub(r'[^A-Za-z0-9\s-]', '', s)  # strip punctuation
    return s.strip().lower()

# ------------------------------------------------------------
# 4.  Prepare join keys
# ------------------------------------------------------------
emdat['city_key']  = emdat['Location'].apply(clean_city)

cities['ISO3']     = cities['Country'].apply(iso2_to_iso3)
cities['city_key'] = cities['City']       .apply(
                         lambda x: unidecode(str(x)).lower().strip())

# keep just one row per city/country – choose the most populated
cities = (cities.sort_values(['ISO3','city_key','Population'],
                             ascending=[True, True, False])
                 .drop_duplicates(['ISO3','city_key'], keep='first'))

# ------------------------------------------------------------
# 5.  Merge – only rows in EM‑DAT that *lack* coordinates
# ------------------------------------------------------------
needs_coords = emdat['Latitude'].isna() | emdat['Longitude'].isna()

merged = emdat.merge(
            cities[['ISO3','city_key','Latitude','Longitude']],
            how='left',
            left_on = ['ISO','city_key'],
            right_on=['ISO3','city_key'],
            suffixes=('', '_city')
         )

# Fill missing values where we got a match
coord_blank = merged['Latitude'].isna()
merged.loc[coord_blank, 'Latitude']  = merged.loc[coord_blank, 'Latitude_city']
merged.loc[coord_blank, 'Longitude'] = merged.loc[coord_blank, 'Longitude_city']

# ------------------------------------------------------------
# 6.  Keep only rows that now have coordinates
# ------------------------------------------------------------
result = merged.dropna(subset=['Latitude','Longitude'])
result = result.drop(columns=['Latitude_city','Longitude_city','ISO3','city_key'])

# ------------------------------------------------------------
# 7.  Save + report
# ------------------------------------------------------------
out_file = os.path.join(OUT_DIR, "emdat_with_coords.csv")
result.to_csv(out_file, index=False)

print(f"Saved {len(result):,} rows with coordinates → {out_file}")
print(f"That’s {(len(result)/len(emdat))*100:.1f}% of the original {len(emdat):,} events.")
