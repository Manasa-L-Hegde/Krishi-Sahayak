from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger(__name__)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CACHE_TTL_SECONDS = 3600  # 1 hour
# In India, Open-Meteo uses global NWP models (ECMWF IFS 0.25° and GFS 0.25°), giving ~15-28 km native resolution.
GRID_RESOLUTION_MIN_KM = 15.0
GRID_RESOLUTION_MAX_KM = 28.0
GRID_RESOLUTION_DESC = "15-28 km (ECMWF IFS 0.25° / GFS Global NWP grid for India)"


class WeatherServiceError(Exception):
    """Raised when weather data cannot be fetched or parsed."""
    pass


@dataclass
class HourlyDataPoint:
    time: str  # ISO string e.g. "2026-06-15T08:00"
    precipitation_mm: Optional[float] = None
    precipitation_probability: Optional[float] = None  # 0 to 100
    rain_mm: Optional[float] = None
    temperature_c: Optional[float] = None
    relative_humidity: Optional[float] = None
    wind_speed_kmh: Optional[float] = None
    wind_gusts_kmh: Optional[float] = None


@dataclass
class WeatherForecast:
    latitude: float
    longitude: float
    elevation: float
    timezone: str
    grid_resolution_desc: str
    grid_resolution_min_km: float
    grid_resolution_max_km: float
    current_time: str
    hourly: List[HourlyDataPoint]
    model_agreement_score: Optional[float] = None  # 0.0 to 1.0; None if secondary model unavailable or all-zero
    secondary_model_used: Optional[str] = None
    secondary_hourly_wind: Optional[List[Optional[float]]] = None
    past_24h_rain_mm: float = 0.0
    past_48h_rain_mm: float = 0.0
    past_7d_rain_mm: float = 0.0


# In-memory coordinate cache: key is (round(lat, 2), round(lon, 2)), value is (timestamp, WeatherForecast)
_WEATHER_CACHE: Dict[Tuple[float, float], Tuple[float, WeatherForecast]] = {}


def round_coordinates(lat: float, lon: float, decimals: int = 2) -> Tuple[float, float]:
    """Round coordinates to ~1.1 km grid to enable caching of adjacent plots."""
    return round(lat, decimals), round(lon, decimals)


def clear_weather_cache() -> None:
    """Utility to clear cache in tests or admin routines."""
    _WEATHER_CACHE.clear()


def _get_series(hourly_dict: Dict[str, List[Any]], base_name: str, model_pref: str = "ecmwf_ifs025") -> List[Any]:
    """Helper to retrieve hourly series handling single-model and multi-model key suffixes."""
    if base_name in hourly_dict:
        return hourly_dict[base_name]
    pref_key = f"{base_name}_{model_pref}"
    if pref_key in hourly_dict:
        return hourly_dict[pref_key]
    best_key = f"{base_name}_best_match"
    if best_key in hourly_dict:
        return hourly_dict[best_key]
    for k in hourly_dict:
        if k.startswith(f"{base_name}_"):
            return hourly_dict[k]
    return []


def parse_hourly_payload(hourly_dict: Dict[str, List[Any]], model_pref: str = "ecmwf_ifs025") -> List[HourlyDataPoint]:
    """
    Parse raw Open-Meteo hourly dictionary into typed HourlyDataPoint objects.
    Preserves None when values are missing (does not convert null to 0.0 or 25.0).
    Handles both default variable names and multi-model suffixed names.
    """
    times = hourly_dict.get("time", [])
    precip = _get_series(hourly_dict, "precipitation", model_pref)
    precip_prob = _get_series(hourly_dict, "precipitation_probability", model_pref)
    rain = _get_series(hourly_dict, "rain", model_pref)
    temp = _get_series(hourly_dict, "temperature_2m", model_pref)
    rh = _get_series(hourly_dict, "relative_humidity_2m", model_pref)
    wind = _get_series(hourly_dict, "wind_speed_10m", model_pref)
    gusts = _get_series(hourly_dict, "wind_gusts_10m", model_pref)

    points: List[HourlyDataPoint] = []
    n = len(times)
    for i in range(n):
        p_prob = float(precip_prob[i]) if (i < len(precip_prob) and precip_prob[i] is not None) else None
        p_val = float(precip[i]) if (i < len(precip) and precip[i] is not None) else None
        r_val = float(rain[i]) if (i < len(rain) and rain[i] is not None) else p_val
        t_val = float(temp[i]) if (i < len(temp) and temp[i] is not None) else None
        rh_val = float(rh[i]) if (i < len(rh) and rh[i] is not None) else None
        w_val = float(wind[i]) if (i < len(wind) and wind[i] is not None) else None
        g_val = float(gusts[i]) if (i < len(gusts) and gusts[i] is not None) else w_val

        points.append(
            HourlyDataPoint(
                time=times[i],
                precipitation_mm=max(0.0, p_val) if p_val is not None else None,
                precipitation_probability=max(0.0, min(100.0, p_prob)) if p_prob is not None else None,
                rain_mm=max(0.0, r_val) if r_val is not None else None,
                temperature_c=t_val,
                relative_humidity=max(0.0, min(100.0, rh_val)) if rh_val is not None else None,
                wind_speed_kmh=max(0.0, w_val) if w_val is not None else None,
                wind_gusts_kmh=max(0.0, g_val) if g_val is not None else None,
            )
        )
    return points


def compute_past_rain(hourly: List[HourlyDataPoint], current_index: int) -> Tuple[float, float, float]:
    """
    Compute cumulative past 24h, 48h, and 7d rainfall up to current_index.
    """
    if current_index <= 0:
        return 0.0, 0.0, 0.0

    idx_24 = max(0, current_index - 24)
    idx_48 = max(0, current_index - 48)
    idx_7d = max(0, current_index - 168)

    rain_24 = sum((pt.rain_mm or 0.0) for pt in hourly[idx_24:current_index] if pt.rain_mm is not None)
    rain_48 = sum((pt.rain_mm or 0.0) for pt in hourly[idx_48:current_index] if pt.rain_mm is not None)
    rain_7d = sum((pt.rain_mm or 0.0) for pt in hourly[idx_7d:current_index] if pt.rain_mm is not None)

    return round(rain_24, 2), round(rain_48, 2), round(rain_7d, 2)


def calculate_model_agreement(
    forecast_primary: List[HourlyDataPoint],
    forecast_secondary_rain: Optional[List[Optional[float]]],
    forecast_secondary_wind: Optional[List[Optional[float]]] = None,
    rain_threshold_mm: float = 0.5,
    wind_threshold_kmh: float = 15.0,
) -> Optional[float]:
    """
    Compute agreement score (0.0 to 1.0) between primary model (ECMWF) and secondary model (GFS)
    STRICTLY over future forecast hours (excluding past analysis/hindcast hours).

    Evaluation dimensions:
    1. Categorical hourly rain match: whether both models agree on spray-critical wet (>=rain_threshold_mm)
       vs dry (<rain_threshold_mm) hours. Sub-threshold trace rain (<0.5mm) is classified as dry.
    2. Rain volume agreement: relative agreement on total forecasted precipitation from significant hours.
       Rule for both-models-dry: If both forecast 0.0 mm significant rain, volume agreement is 1.0 (consistent dry forecast).
    3. Categorical hourly wind match (when secondary wind is available): whether both models agree on
       spray-safe wind (<= wind_threshold_kmh) vs excessive wind (> wind_threshold_kmh).

    Composite weights:
    - If wind is compared: 40% rain categorical + 30% rain volume + 30% wind categorical match.
    - If only rain is compared: 50% rain categorical + 50% rain volume match.
    """
    if not forecast_secondary_rain:
        return None

    length = min(len(forecast_primary), len(forecast_secondary_rain))
    if length == 0:
        return None

    p_rains = [(pt.rain_mm or 0.0) if pt.rain_mm is not None else 0.0 for pt in forecast_primary[:length]]
    s_rains = [(r or 0.0) if r is not None else 0.0 for r in forecast_secondary_rain[:length]]

    # 1. Hourly categorical wet/dry agreement using spray safety threshold (default 0.5 mm)
    wet_p = [r >= rain_threshold_mm for r in p_rains]
    wet_s = [r >= rain_threshold_mm for r in s_rains]

    agree_rain_hours = sum(1 for a, b in zip(wet_p, wet_s) if a == b)
    cat_rain_match = agree_rain_hours / length

    # 2. Total precipitation volume agreement for significant rainfall (>= rain_threshold_mm)
    p_sig = sum(r for r in p_rains if r >= rain_threshold_mm)
    s_sig = sum(r for r in s_rains if r >= rain_threshold_mm)

    if p_sig == 0.0 and s_sig == 0.0:
        vol_match = 1.0
    else:
        vol_diff = abs(p_sig - s_sig)
        vol_match = max(0.0, 1.0 - (vol_diff / max(p_sig, s_sig, 2.0)))

    # 3. Categorical wind agreement if secondary wind provided
    if forecast_secondary_wind:
        w_len = min(length, len(forecast_secondary_wind))
        p_winds = [(pt.wind_speed_kmh or 0.0) if pt.wind_speed_kmh is not None else 0.0 for pt in forecast_primary[:w_len]]
        s_winds = [(w or 0.0) if w is not None else 0.0 for w in forecast_secondary_wind[:w_len]]
        safe_p = [w <= wind_threshold_kmh for w in p_winds]
        safe_s = [w <= wind_threshold_kmh for w in s_winds]
        agree_wind_hours = sum(1 for a, b in zip(safe_p, safe_s) if a == b)
        wind_match = agree_wind_hours / w_len if w_len > 0 else 1.0

        score = 0.40 * cat_rain_match + 0.30 * vol_match + 0.30 * wind_match
    else:
        score = 0.50 * cat_rain_match + 0.50 * vol_match

    return round(score, 2)


def fetch_weather_forecast(
    lat: float,
    lon: float,
    past_days: int = 7,
    forecast_days: int = 7,
    client: Optional[httpx.Client] = None,
    use_cache: bool = True,
) -> WeatherForecast:
    """
    Fetch 7-day hourly forecast + past 7 days from Open-Meteo with caching and multi-model agreement check.
    Fails closed if the service fails: never guesses weather.
    """
    key = round_coordinates(lat, lon)
    now_epoch = datetime.now(timezone.utc).timestamp()

    if use_cache and key in _WEATHER_CACHE:
        cached_time, cached_data = _WEATHER_CACHE[key]
        if (now_epoch - cached_time) < WEATHER_CACHE_TTL_SECONDS:
            return cached_data

    params = {
        "latitude": key[0],
        "longitude": key[1],
        "hourly": "precipitation,precipitation_probability,rain,temperature_2m,relative_humidity_2m,wind_speed_10m,wind_gusts_10m",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "auto",
        "models": "ecmwf_ifs025,gfs_seamless",
    }

    own_client = False
    if client is None:
        client = httpx.Client(timeout=httpx.Timeout(6.0, connect=3.0))
        own_client = True

    try:
        response = client.get(OPEN_METEO_FORECAST_URL, params=params)
        if response.status_code != 200:
            params.pop("models")
            fallback_resp = client.get(OPEN_METEO_FORECAST_URL, params=params)
            if fallback_resp.status_code != 200:
                raise WeatherServiceError(
                    f"Open-Meteo returned status {fallback_resp.status_code}: {fallback_resp.text[:200]}"
                )
            data = fallback_resp.json()
            secondary_rain = None
            sec_model_name = None
        else:
            data = response.json()
            sec_model_name = "gfs_seamless"
            h = data.get("hourly", {})
            secondary_rain = h.get("rain_gfs_seamless") or h.get("precipitation_gfs_seamless")
            secondary_wind = h.get("wind_speed_10m_gfs_seamless")

    except httpx.RequestError as exc:
        logger.error(f"Network error communicating with Open-Meteo: {exc}")
        raise WeatherServiceError(f"Failed to connect to weather service: {exc}") from exc
    finally:
        if own_client:
            client.close()

    hourly_dict = data.get("hourly")
    if not hourly_dict or "time" not in hourly_dict:
        raise WeatherServiceError("Weather response missing hourly forecast data.")

    hourly_points = parse_hourly_payload(hourly_dict, model_pref="ecmwf_ifs025")
    if not hourly_points:
        raise WeatherServiceError("Parsed zero hourly weather datapoints.")

    current_index = None
    tz_name = data.get("timezone") or "Asia/Kolkata"
    try:
        import zoneinfo
        tz_obj = zoneinfo.ZoneInfo(tz_name)
        now_local = datetime.now(tz_obj)
        now_hour_iso = now_local.strftime("%Y-%m-%dT%H:00")
        for idx, pt in enumerate(hourly_points):
            if pt.time == now_hour_iso:
                current_index = idx
                break
    except Exception as exc:
        logger.debug(f"Could not compute local wall-clock hour index: {exc}")

    if current_index is None:
        # Fallback to past_days * 24 (e.g. for synthetic unit tests with fixed past dates)
        current_index = min(past_days * 24, len(hourly_points) - 1)
    rain_24, rain_48, rain_7d = compute_past_rain(hourly_points, current_index)

    # STRICT: Evaluate model agreement ONLY over future forecast hours (exclude past analysis/hindcast)
    future_hourly = hourly_points[current_index:]
    future_sec_rain = secondary_rain[current_index:] if secondary_rain else None
    future_sec_wind = secondary_wind[current_index:] if secondary_wind else None
    agreement_score = calculate_model_agreement(future_hourly, future_sec_rain, future_sec_wind)

    forecast = WeatherForecast(
        latitude=data.get("latitude", key[0]),
        longitude=data.get("longitude", key[1]),
        elevation=data.get("elevation", 0.0),
        timezone=data.get("timezone", "UTC"),
        grid_resolution_desc=GRID_RESOLUTION_DESC,
        grid_resolution_min_km=GRID_RESOLUTION_MIN_KM,
        grid_resolution_max_km=GRID_RESOLUTION_MAX_KM,
        current_time=hourly_points[current_index].time if current_index < len(hourly_points) else "",
        hourly=hourly_points,
        model_agreement_score=agreement_score,
        secondary_model_used=sec_model_name if secondary_rain is not None else None,
        secondary_hourly_wind=secondary_wind,
        past_24h_rain_mm=rain_24,
        past_48h_rain_mm=rain_48,
        past_7d_rain_mm=rain_7d,
    )

    if use_cache:
        _WEATHER_CACHE[key] = (now_epoch, forecast)

    return forecast
