import pandas as pd
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

STORE_BRANCHES = [
    {"Store": "No Frills", "Branch": "Surrey Central", "Address": "10153 King George Blvd", "lat": 49.1882, "lon": -122.8455},
    {"Store": "Walmart Supercentre", "Branch": "Guildford Town Centre", "Address": "10355 152 St", "lat": 49.1891, "lon": -122.8045},
    {"Store": "Real Canadian Superstore", "Branch": "King George & 104 Ave", "Address": "10435 King George Blvd", "lat": 49.1935, "lon": -122.8442},
    {"Store": "Save-On-Foods", "Branch": "Guildford Village", "Address": "10312 152 St", "lat": 49.1898, "lon": -122.8018},
    {"Store": "FreshCo", "Branch": "Fleetwood Surrey", "Address": "15930 Fraser Hwy", "lat": 49.1632, "lon": -122.7812},
    {"Store": "Safeway", "Branch": "Newton Town Centre", "Address": "7380 King George Blvd", "lat": 49.1368, "lon": -122.8448},
    {"Store": "Thrifty Foods", "Branch": "Morgan Crossing South Surrey", "Address": "15745 Croydon Dr", "lat": 49.0485, "lon": -122.7865},
]


def get_nearby_stores(pincode, max_radius_km=6.0):
    """Finds the single closest branch per store chain within max_radius_km."""
    clean_code = pincode.strip().upper()
    lat, lon, address = 49.1891, -122.8045, f"{clean_code} (Surrey, BC)"

    try:
        geolocator = Nominatim(user_agent="grocery_assistant_surrey")
        loc = geolocator.geocode(f"{clean_code}, Surrey, BC, Canada", timeout=5)
        if loc:
            lat, lon, address = loc.latitude, loc.longitude, loc.address
    except Exception:
        pass

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

    return address, pd.DataFrame(list(unique_stores.values()))
