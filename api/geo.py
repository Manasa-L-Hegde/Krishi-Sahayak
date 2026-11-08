from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger(__name__)

OPEN_METEO_GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
NOMINATIM_USER_AGENT = "KrishiSahayak-SprayAdvisor/1.0 (https://github.com/manas/krishi-sahayak)"

GEO_CACHE_TTL_SECONDS = 86400  # 24 hours
GRID_RESOLUTION_DESC = "15-28 km (NWP global grid for India; plot-specific microclimate may vary)"
GRID_RESOLUTION_MIN_KM = 15.0
GRID_RESOLUTION_MAX_KM = 28.0

# In-memory search cache: query -> (timestamp, List[GeoLocation])
_GEO_SEARCH_CACHE: Dict[str, Tuple[float, List[GeoLocation]]] = {}

# In-memory reverse cache: (round(lat, 3), round(lon, 3)) -> (timestamp, Optional[GeoLocation])
_GEO_REVERSE_CACHE: Dict[Tuple[float, float], Tuple[float, Optional[GeoLocation]]] = {}


@dataclass
class GeoLocation:
    name: str
    display_name: str
    latitude: float
    longitude: float
    state: Optional[str] = None
    district: Optional[str] = None
    taluk_or_subdistrict: Optional[str] = None
    postcode: Optional[str] = None
    country: str = "India"
    grid_resolution_desc: str = GRID_RESOLUTION_DESC
    grid_resolution_min_km: float = GRID_RESOLUTION_MIN_KM
    grid_resolution_max_km: float = GRID_RESOLUTION_MAX_KM


def clear_geo_cache() -> None:
    _GEO_SEARCH_CACHE.clear()
    _GEO_REVERSE_CACHE.clear()


def is_within_india(lat: float, lon: float) -> bool:
    """Validate that coordinates roughly fall within India's geographical envelope."""
    return 6.0 <= lat <= 38.0 and 68.0 <= lon <= 98.0


def search_places(
    query: str,
    limit: int = 5,
    client: Optional[httpx.Client] = None,
    use_cache: bool = True,
) -> List[GeoLocation]:
    """
    Search places across India using Open-Meteo Geocoding with OSM Nominatim fallback.
    Caches results for 24h.
    """
    normalized_q = query.strip().lower()
    if not normalized_q:
        return []

    now_epoch = datetime.now(timezone.utc).timestamp()
    if use_cache and normalized_q in _GEO_SEARCH_CACHE:
        cached_time, cached_results = _GEO_SEARCH_CACHE[normalized_q]
        if (now_epoch - cached_time) < GEO_CACHE_TTL_SECONDS:
            return cached_results

    results: List[GeoLocation] = []
    own_client = False
    if client is None:
        client = httpx.Client(timeout=httpx.Timeout(5.0, connect=3.0))
        own_client = True

    try:
        # 1. Try Open-Meteo Geocoding
        om_params = {
            "name": query.strip(),
            "count": limit,
            "language": "en",
            "format": "json",
            "country_code": "IN",
        }
        om_resp = client.get(OPEN_METEO_GEO_URL, params=om_params)
        if om_resp.status_code == 200:
            om_data = om_resp.json()
            raw_places = om_data.get("results") or []
            for item in raw_places:
                lat = float(item.get("latitude", 0.0))
                lon = float(item.get("longitude", 0.0))
                if not is_within_india(lat, lon):
                    continue
                name = item.get("name", "")
                admin1 = item.get("admin1")  # State
                admin2 = item.get("admin2")  # District
                admin3 = item.get("admin3")  # Taluk / sub-district
                
                parts = [p for p in [name, admin3, admin2, admin1, "India"] if p]
                display = ", ".join(parts)

                results.append(
                    GeoLocation(
                        name=name,
                        display_name=display,
                        latitude=lat,
                        longitude=lon,
                        state=admin1,
                        district=admin2,
                        taluk_or_subdistrict=admin3,
                        postcode=item.get("postcode"),
                        country="India",
                        grid_resolution_desc=GRID_RESOLUTION_DESC,
                        grid_resolution_min_km=GRID_RESOLUTION_MIN_KM,
                        grid_resolution_max_km=GRID_RESOLUTION_MAX_KM,
                    )
                )

        # 2. If Open-Meteo returned nothing, try OSM Nominatim
        if not results:
            nom_params = {
                "q": query.strip(),
                "countrycodes": "in",
                "format": "json",
                "addressdetails": "1",
                "limit": limit,
            }
            headers = {"User-Agent": NOMINATIM_USER_AGENT}
            nom_resp = client.get(NOMINATIM_SEARCH_URL, params=nom_params, headers=headers)
            if nom_resp.status_code == 200:
                nom_data = nom_resp.json()
                for item in nom_data:
                    lat = float(item.get("lat", 0.0))
                    lon = float(item.get("lon", 0.0))
                    if not is_within_india(lat, lon):
                        continue
                    addr = item.get("address", {})
                    name = (
                        addr.get("village")
                        or addr.get("town")
                        or addr.get("city")
                        or addr.get("suburb")
                        or item.get("name")
                        or ""
                    )
                    state = addr.get("state")
                    district = addr.get("state_district") or addr.get("county")
                    taluk = addr.get("taluk") or addr.get("subdistrict")
                    postcode = addr.get("postcode")
                    display = item.get("display_name", "")

                    results.append(
                        GeoLocation(
                            name=name or display.split(",")[0],
                            display_name=display,
                            latitude=lat,
                            longitude=lon,
                            state=state,
                            district=district,
                            taluk_or_subdistrict=taluk,
                            postcode=postcode,
                            country="India",
                            grid_resolution_desc=GRID_RESOLUTION_DESC,
                            grid_resolution_min_km=GRID_RESOLUTION_MIN_KM,
                            grid_resolution_max_km=GRID_RESOLUTION_MAX_KM,
                        )
                    )

    except httpx.RequestError as exc:
        logger.warning(f"Geocoding request failed: {exc}")
    finally:
        if own_client:
            client.close()

    if use_cache and results:
        _GEO_SEARCH_CACHE[normalized_q] = (now_epoch, results)

    return results


def reverse_geocode(
    lat: float,
    lon: float,
    client: Optional[httpx.Client] = None,
    use_cache: bool = True,
) -> Optional[GeoLocation]:
    """
    Reverse geocode coordinates to the nearest Indian locality.
    """
    if not is_within_india(lat, lon):
        return None

    rounded_key = (round(lat, 3), round(lon, 3))
    now_epoch = datetime.now(timezone.utc).timestamp()
    if use_cache and rounded_key in _GEO_REVERSE_CACHE:
        cached_time, cached_res = _GEO_REVERSE_CACHE[rounded_key]
        if (now_epoch - cached_time) < GEO_CACHE_TTL_SECONDS:
            return cached_res

    own_client = False
    if client is None:
        client = httpx.Client(timeout=httpx.Timeout(5.0, connect=3.0))
        own_client = True

    location: Optional[GeoLocation] = None
    try:
        nom_params = {
            "lat": lat,
            "lon": lon,
            "format": "json",
            "addressdetails": "1",
        }
        headers = {"User-Agent": NOMINATIM_USER_AGENT}
        resp = client.get(NOMINATIM_REVERSE_URL, params=nom_params, headers=headers)
        if resp.status_code == 200:
            data = resp.json()
            addr = data.get("address", {})
            name = (
                addr.get("village")
                or addr.get("town")
                or addr.get("city")
                or addr.get("suburb")
                or addr.get("hamlet")
                or ""
            )
            location = GeoLocation(
                name=name or data.get("name") or f"Point ({lat:.3f}, {lon:.3f})",
                display_name=data.get("display_name", f"{lat:.3f}, {lon:.3f}"),
                latitude=lat,
                longitude=lon,
                state=addr.get("state"),
                district=addr.get("state_district") or addr.get("county"),
                taluk_or_subdistrict=addr.get("taluk") or addr.get("subdistrict"),
                postcode=addr.get("postcode"),
                country="India",
                grid_resolution_desc=GRID_RESOLUTION_DESC,
                grid_resolution_min_km=GRID_RESOLUTION_MIN_KM,
                grid_resolution_max_km=GRID_RESOLUTION_MAX_KM,
            )
    except httpx.RequestError as exc:
        logger.warning(f"Reverse geocoding request failed: {exc}")
    finally:
        if own_client:
            client.close()

    if use_cache:
        _GEO_REVERSE_CACHE[rounded_key] = (now_epoch, location)

    return location
