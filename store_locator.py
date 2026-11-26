import re
import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

VALID_SURREY_FSAS = ["V3R", "V3S", "V3T", "V3V", "V3W", "V3X", "V3Z", "V4A", "V4N", "V4P"]

STORE_BRANCHES = [
    {"FSA": "V3T", "Area": "Surrey Central / Whalley, BC", "Store": "Walmart Supercentre", "Branch": "Central City Mall (Surrey Central)", "Address": "10153 King George Blvd, Surrey, BC", "lat": 49.1882, "lon": -122.8455},
    {"FSA": "V3R", "Area": "Guildford, Surrey, BC", "Store": "No Frills", "Branch": "Guildford / 148 St", "Address": "10878 148 St, Surrey, BC", "lat": 49.2001, "lon": -122.8124},
    {"FSA": "V3V", "Area": "Surrey Central / Whalley, BC", "Store": "Real Canadian Superstore", "Branch": "King George & 104 Ave", "Address": "10435 King George Blvd, Surrey, BC", "lat": 49.1935, "lon": -122.8442},
    {"FSA": "V3R", "Area": "Guildford, Surrey, BC", "Store": "Save-On-Foods", "Branch": "Guildford Village", "Address": "10312 152 St, Surrey, BC", "lat": 49.1898, "lon": -122.8018},
    {"FSA": "V4N", "Area": "Fleetwood Surrey, BC", "Store": "FreshCo", "Branch": "Fleetwood Surrey", "Address": "15930 Fraser Hwy, Surrey, BC", "lat": 49.1632, "lon": -122.7812},
    {"FSA": "V3W", "Area": "Newton, Surrey, BC", "Store": "Safeway", "Branch": "Newton Town Centre", "Address": "7380 King George Blvd, Surrey, BC", "lat": 49.1368, "lon": -122.8448},
    {"FSA": "V3Z", "Area": "South Surrey, BC", "Store": "Thrifty Foods", "Branch": "Morgan Crossing South Surrey", "Address": "15745 Croydon Dr, Surrey, BC", "lat": 49.0485, "lon": -122.7865},
]

def resolve_surrey_pincode(pincode):
    """Validates a Surrey, BC postal code and returns (lat, lon, label, is_valid)."""
    raw = pincode.strip().upper()
    compact = raw.replace(" ", "")

    if not re.match(r"^[A-Z]\d[A-Z]\d[A-Z]\d$", compact):
        return None, None, f"Invalid postal code format '{raw}'. Use Canadian format like V3R 0A4 or V3T 2W1.", False

    prefix = compact[:3]
    formatted_code = f"{compact[:3]} {compact[3:]}"

    if prefix not in VALID_SURREY_FSAS:
        return None, None, f"'{formatted_code}' is outside our Surrey, BC service area (valid prefixes: {', '.join(VALID_SURREY_FSAS)}).", False

    # Match to branch with same FSA if available, otherwise default to Central City Surrey
    matched_branch = STORE_BRANCHES
    for branch in STORE_BRANCHES:
        if branch["FSA"] == prefix:
            matched_branch = branch
            break

    lat, lon, area = matched_branch["lat"], matched_branch["lon"], matched_branch["Area"]

    try:
        geolocator = Nominatim(user_agent="grocery_assistant_surrey_v5")
        loc = geolocator.geocode(f"{formatted_code}, Surrey, BC, Canada", timeout=4)
        if loc:
            return loc.latitude, loc.longitude, f"{formatted_code} ({area})", True
    except Exception:
        pass

    return lat, lon, f"{formatted_code} ({area})", True


def get_nearby_stores(pincode, max_radius_km=6.0):
    """Returns (address_or_error, nearby_df, is_valid) for the closest branch per store chain."""
    lat, lon, address, is_valid = resolve_surrey_pincode(pincode)

    if not is_valid:
        return address, pd.DataFrame(), False

    matches = []
    for branch in STORE_BRANCHES:
        dist = round(geodesic((lat, lon), (branch["lat"], branch["lon"])).km, 2)
        if dist <= max_radius_km:
            row = branch.copy()
            row["Distance (km)"] = dist
            matches.append(row)

    matches.sort(key=lambda x: x["Distance (km)"])

    unique_stores = {}
    for store_row in matches:
        name = store_row["Store"]
        if name not in unique_stores:
            unique_stores[name] = store_row

    return address, pd.DataFrame(list(unique_stores.values())), True