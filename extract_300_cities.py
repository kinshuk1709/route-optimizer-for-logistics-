import pandas as pd

INP = "paste.txt"
OUT = "cities_data.csv"
N = 300

code_to_state = {
    "DL": "Delhi", "WB": "West Bengal", "MH": "Maharashtra", "TG": "Telangana", 
    "KA": "Karnataka", "TN": "Tamil Nadu", "GJ": "Gujarat", "RJ": "Rajasthan", 
    "UP": "Uttar Pradesh", "MP": "Madhya Pradesh", "BR": "Bihar", "AP": "Andhra Pradesh", 
    "PB": "Punjab", "JH": "Jharkhand", "JK": "Jammu and Kashmir", "CH": "Chandigarh", 
    "CT": "Chhattisgarh", "AS": "Assam", "KL": "Kerala", "OR": "Odisha", 
    "PY": "Puducherry", "UT": "Uttarakhand", "TR": "Tripura", "NL": "Nagaland", 
    "MZ": "Mizoram", "GA": "Goa", "AN": "Andaman and Nicobar Islands",
    "DN": "Dadra and Nagar Haveli", "DD": "Dadra and Nagar Haveli",
    "HP": "Himachal Pradesh", "SK": "Sikkim", "AR": "Arunachal Pradesh",
    "ML": "Meghalaya", "LD": "Lakshadweep", "MN": "Manipur", "HR": "Haryana"
}

def tier(pop):
    if pop >= 3_000_000: return "Tier-1"
    if pop >= 700_000: return "Tier-2"
    return "Tier-3"

tier_defaults = {
    "Tier-1": dict(Average_Travel_Time_10km_Minutes=18, Congestion_Level_Peak_Percent=48, Average_Speed_Peak_Hours_kmph=28, Average_Speed_Off_Peak_kmph=40, Delivery_Locations=40000, Avg_Daily_Orders=12000),
    "Tier-2": dict(Average_Travel_Time_10km_Minutes=16, Congestion_Level_Peak_Percent=40, Average_Speed_Peak_Hours_kmph=31, Average_Speed_Off_Peak_kmph=44, Delivery_Locations=20000, Avg_Daily_Orders=6000),
    "Tier-3": dict(Average_Travel_Time_10km_Minutes=14, Congestion_Level_Peak_Percent=30, Average_Speed_Peak_Hours_kmph=35, Average_Speed_Off_Peak_kmph=48, Delivery_Locations=12000, Avg_Daily_Orders=3500)
}

print("📂 Reading paste.txt...")
raw = pd.read_csv(INP, header=None, engine="python")
print(f"📊 Found {len(raw)} total rows in source file")

df = pd.DataFrame({
    "City": raw[1].astype(str).str.strip(),
    "Latitude": pd.to_numeric(raw[4], errors="coerce"),
    "Longitude": pd.to_numeric(raw[5], errors="coerce"),
    "Population": pd.to_numeric(raw[7].astype(str).str.replace(",", "", regex=False), errors="coerce"),
    "StateCode": raw[9].astype(str).str.strip(),
}).dropna(subset=["City", "Latitude", "Longitude", "Population"])

print(f"✅ After cleaning: {len(df)} cities with valid coordinates")

df = df[(df["Latitude"] != 0) & (df["Longitude"] != 0)]
df = df.sort_values("Population", ascending=False).drop_duplicates(subset=["City", "StateCode"], keep="first")
df = df.head(N).copy()

print(f"✅ Extracted top {len(df)} cities by population")

df["State"] = df["StateCode"].map(code_to_state).fillna(df["StateCode"])
df["Tier"] = df["Population"].apply(tier)

for t, d in tier_defaults.items():
    m = df["Tier"].eq(t)
    for k, v in d.items():
        df.loc[m, k] = v

out = df[["City", "Latitude", "Longitude", "Tier", "State", "Average_Travel_Time_10km_Minutes", "Congestion_Level_Peak_Percent", "Average_Speed_Peak_Hours_kmph", "Average_Speed_Off_Peak_kmph", "Delivery_Locations", "Avg_Daily_Orders"]]
out.to_csv(OUT, index=False, encoding="utf-8")

print(f"\n✅ Successfully created {OUT} with {len(out)} cities!")
print(f"\n📊 Distribution by Tier:")
tier_counts = df['Tier'].value_counts().sort_index()
for tier, count in tier_counts.items():
    print(f"  {tier}: {count} cities")

print(f"\n🏙️ Top 15 States by City Count:")
state_counts = df['State'].value_counts().head(15)
for state, count in state_counts.items():
    print(f"  {state}: {count} cities")

print(f"\n📍 Top 15 Cities:")
print(df[['City', 'State', 'Tier']].head(15).to_string(index=False))
