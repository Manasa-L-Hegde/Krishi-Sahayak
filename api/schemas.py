from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {"N": 90, "P": 42, "K": 43, "temperature": 25,
                                                   "humidity": 80, "ph": 6.5, "rainfall": 200}})

    N: float = Field(..., ge=0, le=200, examples=[90.0])
    P: float = Field(..., ge=0, le=200, examples=[42.0])
    K: float = Field(..., ge=0, le=250, examples=[43.0])
    temperature: float = Field(..., ge=-10, le=60, examples=[25.0])
    humidity: float = Field(..., ge=0, le=100, examples=[80.0])
    ph: float = Field(..., ge=0, le=14, examples=[6.5])
    rainfall: float = Field(..., ge=0, le=500, examples=[200.0])


class ShapContribution(BaseModel):
    feature: str
    value: float
    direction: str


class PredictionResponse(BaseModel):
    crop: str
    confidence: float
    risk_score: float
    shap_contributions: list[ShapContribution]


class ForecastRequest(BaseModel):
    commodity: str = Field(..., examples=["Onion"])
    recent_prices: list[float] = Field(..., min_length=30, examples=[[120.0] * 30])


class ForecastResponse(BaseModel):
    commodity: str
    horizon_days: int
    forecast_prices: list[float]
    model: str
