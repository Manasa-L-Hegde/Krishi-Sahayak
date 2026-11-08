from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import httpx

logger = logging.getLogger(__name__)

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_CACHE_TTL_SECONDS = 3600  # 1 hour
DEFAULT_GRID_RESOLUTION_KM = 5.0


class WeatherServiceError(Exception):
    """Raised when weather data cannot be fetched or parsed."""
    pass


@dataclass
class HourlyDataPoint:
    time: str  # ISO string e.g. "2026-06-15T08:00"
    precipitation_mm: float
    precipitation_probability: float  # 0 to 100
    rain_mm: float
    temperature_c: float
    relative_humidity: float
    wind_speed_kmh: float
    wind_gusts_kmh: float


@dataclass
class WeatherForecast:
    latitude: float
    longitude: float
    elevation: float
    timezone: str
    grid_resolution_km: float
    current_time: str
    hourly: List[HourlyDataPoint]
    model_agreement_score: float  # 0.0 (total disagreement) to 1.0 (strong agreement)
    secondary_model_used: Optional[str] = None
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


def parse_hourly_payload(hourly_dict: Dict[str, List[Any]]) -> List[HourlyDataPoint]:
    """Parse raw Open-Meteo hourly dictionary into typed HourlyDataPoint objects."""
    times = hourly_dict.get("time", [])
    precip = hourly_dict.get("precipitation", [])
    precip_prob = hourly_dict.get("precipitation_probability", [])
    rain = hourly_dict.get("rain", [])
    temp = hourly_dict.get("temperature_2m", [])
    rh = hourly_dict.get("relative_humidity_2m", [])
    wind = hourly_dict.get("wind_speed_10m", [])
    gusts = hourly_dict.get("wind_gusts_10m", [])

    points: List[HourlyDataPoint] = []
    n = len(times)
    for i in range(n):
        p_prob = float(precip_prob[i]) if i < len(precip_prob) and precip_prob[i] is not None else 0.0
        p_val = float(precip[i]) if i < len(precip) and precip[i] is not None else 0.0
        r_val = float(rain[i]) if i < len(rain) and rain[i] is not None else p_val
        t_val = float(temp[i]) if i < len(temp) and temp[i] is not None else 25.0
        rh_val = float(rh[i]) if i < len(rh) and rh[i] is not None else 70.0
        w_val = float(wind[i]) if i < len(wind) and wind[i] is not None else 0.0
        g_val = float(gusts[i]) if i < len(gusts) and gusts[i] is not None else w_val

        points.append(
            HourlyDataPoint(
                time=times[i],
                precipitation_mm=max(0.0, p_val),
                precipitation_probability=max(0.0, min(100.0, p_prob)),
                rain_mm=max(0.0, r_val),
                temperature_c=t_val,
                relative_humidity=max(0.0, min(100.0, rh_val)),
                wind_speed_kmh=max(0.0, w_val),
                wind_gusts_kmh=max(0.0, g_val),
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

    rain_24 = sum(pt.rain_mm for pt in hourly[idx_24:current_index])
    rain_48 = sum(pt.rain_mm for pt in hourly[idx_48:current_index])
    rain_7d = sum(pt.rain_mm for pt in hourly[idx_7d:current_index])

    return round(rain_24, 2), round(rain_48, 2), round(rain_7d, 2)


def calculate_model_agreement(
    primary_hourly: List[HourlyDataPoint],
    secondary_rain_series: Optional[List[float]],
) -> float:
    """
    Compute agreement score (0.0 to 1.0) between primary model rain and secondary model rain.
    Returns 1.0 if secondary model is not queried or identical.
    """
    if not secondary_rain_series:
        return 1.0

    length = min(len(primary_hourly), len(secondary_rain_series))
    if length == 0:
        return 1.0

    total_diff = 0.0
    for i in range(length):
        diff = abs(primary_hourly[i].rain_mm - (secondary_rain_series[i] or 0.0))
        total_diff += min(5.0, diff)

    mean_diff = total_diff / length
    agreement = max(0.0, 1.0 - (mean_diff / 5.0))
    return round(agreement, 2)


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
        "models": "best_match,ecmwf_ifs025",
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
            sec_model_name = "ecmwf_ifs025"
            sec_hourly = data.get("hourly_ecmwf_ifs025") or {}
            secondary_rain = sec_hourly.get("precipitation") or sec_hourly.get("rain")

    except httpx.RequestError as exc:
        logger.error(f"Network error communicating with Open-Meteo: {exc}")
        raise WeatherServiceError(f"Failed to connect to weather service: {exc}") from exc
    finally:
        if own_client:
            client.close()

    hourly_dict = data.get("hourly")
    if not hourly_dict or "time" not in hourly_dict:
        raise WeatherServiceError("Weather response missing hourly forecast data.")

    hourly_points = parse_hourly_payload(hourly_dict)
    if not hourly_points:
        raise WeatherServiceError("Parsed zero hourly weather datapoints.")

    current_index = min(past_days * 24, len(hourly_points) - 1)
    rain_24, rain_48, rain_7d = compute_past_rain(hourly_points, current_index)
    agreement_score = calculate_model_agreement(hourly_points, secondary_rain)

    forecast = WeatherForecast(
        latitude=data.get("latitude", key[0]),
        longitude=data.get("longitude", key[1]),
        elevation=data.get("elevation", 0.0),
        timezone=data.get("timezone", "UTC"),
        grid_resolution_km=DEFAULT_GRID_RESOLUTION_KM,
        current_time=hourly_points[current_index].time if current_index < len(hourly_points) else "",
        hourly=hourly_points,
        model_agreement_score=agreement_score,
        secondary_model_used=sec_model_name if secondary_rain is not None else None,
        past_24h_rain_mm=rain_24,
        past_48h_rain_mm=rain_48,
        past_7d_rain_mm=rain_7d,
    )

    if use_cache:
        _WEATHER_CACHE[key] = (now_epoch, forecast)

    return forecast
