import json
from pathlib import Path

class Geocacher:
    """Handles reverse geocoding with a local JSON cache and rate limiting."""
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache = {}
        self.geocoder = None
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load geocache: {e}")

    def _init_geocoder(self):
        if not self.geocoder:
            from geopy.geocoders import Nominatim
            self.geocoder = Nominatim(user_agent="virtual-me-travel-extractor")

    def get_address(self, lat: float, lng: float) -> str:
        # Use 6 decimal places (~11cm precision) for the cache key
        # to avoid mixing up nearby but distinct places.
        key = f"{lat:.6f}_{lng:.6f}"
        if key in self.cache:
            return self.cache[key]

        print(f"🌐 Geocoding {key} ...", flush=True)
        self._init_geocoder()
        import time
        try:
            # Nominatim usage policy asks for 1 req/sec
            time.sleep(1.1)
            location = self.geocoder.reverse((lat, lng), language="en")
            if location and location.address:
                addr = location.address
                self.cache[key] = addr
                self._save_cache()
                return addr
        except Exception as e:
            print(f"      ⚠️ Geocoding failed for {key}: {e}", flush=True)
        
        fallback = f"lat{lat:.6f}_lng{lng:.6f}"
        return fallback

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save geocache: {e}")

def parse_address_hierarchy(addr: str) -> dict:
    """
    Given a Nominatim address string (comma-separated), extracts City and Country.
    Example: '10, Rue de la Paix, Paris, Île-de-France, 75000, France'
    Returns: {'city': 'Paris', 'country': 'France'}
    """
    if not addr:
        return {}
    parts = [p.strip() for p in addr.split(",")]
    if len(parts) < 2:
        return {"country": parts[0] if parts else "Unknown"}
        
    country = parts[-1]
    city = None
    
    # Heuristic for city: 
    # Usually: [Place], [Neighborhood], [City/Town], [Postcode], [Country]
    # Or: [Place], [Neighborhood], [Borough], [City], [Region], [Postcode], [Country]
    
    # We work backwards from the country
    idx = -2
    if idx >= -len(parts) and any(c.isdigit() for c in parts[idx]):
        idx -= 1 # skip zip code
    
    # Skip regions/administrative labels
    skip_keywords = {
        "Metropolitan France", "Region", "Department", "Arrondissement", 
        "Quartier", "Borough", "County", "State", "Province", "District",
        "Ile-de-France", "California", "New York", "England" # Common large regions
    }
    
    while abs(idx) <= len(parts):
        candidate = parts[idx]
        # If the candidate contains a skip keyword or is a known region, keep going back
        if any(sk in candidate for sk in skip_keywords) or candidate in skip_keywords:
            idx -= 1
            continue
        # If we find something that looks like a city name
        city = candidate
        break
        
    if not city and len(parts) >= 2:
        city = parts[0] # Fallback to first part if we exhausted everything
        
    if not city:
        city = "Unknown City"
        
    return {"city": city, "country": country}
