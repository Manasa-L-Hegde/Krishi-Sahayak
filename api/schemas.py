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


class AdvisoryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What is PM-KISAN?"])
    language: str | None = Field(default=None, pattern="(?i)^(english|kannada)$", examples=["English"])


class AdvisoryResponse(BaseModel):
    answer: str
    sources: list[str]
    grounded: bool
    fallback: bool


class ChatAdvisoryRequest(BaseModel):
    question: str = Field(..., min_length=2, examples=["How to prevent pests in rice?"])
    language: str = Field(default="English", examples=["English"])


class ChatAdvisoryResponse(BaseModel):
    answer: str
    sources: list[str]
    grounded: bool
    verified: bool
    fallback: bool


class SprayAdvisoryRequest(BaseModel):
    crop: str = Field(..., examples=["arecanut"])
    disease: str = Field(..., examples=["koleroga"])
    latitude: float = Field(..., ge=6.0, le=38.0, examples=[14.167])
    longitude: float = Field(..., ge=68.0, le=98.0, examples=[75.043])
    spray_intent: str = Field(default="first", pattern="(?i)^(first|repeat|unsure)$", examples=["first"])
    last_spray_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2026-06-01"])
    language: str = Field(default="en", pattern="(?i)^(en|kn)$", examples=["en"])


class SprayAdvisoryResponse(BaseModel):
    crop: str
    disease: str
    spray_intent: str
    decision: str
    decision_label: str
    reason: str
    confidence_score: float
    confidence_factors: dict
    best_window: dict | None
    agronomy: dict
    wash_off_alert: str | None
    seasonal_context_note: str | None
    disclaimer: str
    kvk_guidance: str
    citations: list[dict]


