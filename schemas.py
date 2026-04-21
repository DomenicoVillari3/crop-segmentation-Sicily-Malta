"""
schemas.py
==========
Modelli Pydantic per input/output API.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, model_validator
from config import SUPPORTED_YEARS

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

start_year = min(SUPPORTED_YEARS)
end_year = max(SUPPORTED_YEARS)

class PointRequest(BaseModel):
    lat:  float = Field(..., ge=-90,  le=90,  description="Latitudine WGS84")
    lon:  float = Field(..., ge=-180, le=180, description="Longitudine WGS84")
    year: int   = Field(2023, ge=start_year, le=end_year, description="Anno Sentinel-2")

    model_config = {"json_schema_extra": {
        "example": {"lat": 37.38, "lon": 14.91, "year": 2023}
    }}


class BBoxRequest(BaseModel):
    min_lon: float = Field(..., ge=-180, le=180)
    min_lat: float = Field(..., ge=-90,  le=90)
    max_lon: float = Field(..., ge=-180, le=180)
    max_lat: float = Field(..., ge=-90,  le=90)
    year:    int   = Field(2023, ge=start_year, le=end_year)

    @model_validator(mode="after")
    def bbox_valid(self):
        if self.max_lon <= self.min_lon:
            raise ValueError("max_lon deve essere > min_lon")
        if self.max_lat <= self.min_lat:
            raise ValueError("max_lat deve essere > min_lat")
        return self

    def to_list(self) -> list[float]:
        return [self.min_lon, self.min_lat, self.max_lon, self.max_lat]


# ---------------------------------------------------------------------------
# Output — componenti atomici
# ---------------------------------------------------------------------------

class ClassStat(BaseModel):
    class_id:   int
    name:       str
    color_hex:  str   = Field(description="Colore esadecimale es. #32ff32")
    hectares:   float = Field(description="Superficie in ettari")
    percentage: float = Field(description="% sul totale non-sfondo")


class NDVIPoint(BaseModel):
    season: Literal["winter", "spring", "summer", "autumn"]
    mean:   float
    std:    float
    min:    float
    max:    float


class LegendItem(BaseModel):
    class_id:  int
    name:      str
    color_hex: str
    hectares:  float | None = Field(None, description="Ettari (None se legenda statica)")
    percentage: float | None = None


# ---------------------------------------------------------------------------
# Output — risposta analisi
# ---------------------------------------------------------------------------

class AnalysisResponse(BaseModel):
    task_id:     str
    status:      Literal["pending", "running", "completed", "failed"]
    progress:    int  = Field(0, ge=0, le=100, description="Avanzamento 0-100%")
    source:      Literal["minio", "stac"] | None = None
    year_used:   int | None = None
    bbox:        list[float] | None = None    # [min_lon, min_lat, max_lon, max_lat]

    # Risultati (None finché non completed)
    class_stats: list[ClassStat] | None = None
    ndvi_series: list[NDVIPoint] | None = None
    total_ha:    float | None = None

    # Errore
    error:       str | None = None


# ---------------------------------------------------------------------------
# Output — endpoint dedicati
# ---------------------------------------------------------------------------

class NDVIResponse(BaseModel):
    task_id:    str
    year_used:  int
    bbox:       list[float]
    ndvi_series: list[NDVIPoint]


class LegendResponse(BaseModel):
    items:    list[LegendItem]
    total_ha: float | None = Field(None, description="Totale ettari analisi corrente")


class HealthResponse(BaseModel):
    status:       Literal["ok", "degraded", "error"]
    device:       str
    vram_used_mb: float
    minio:        Literal["ok", "error"]
    model:        Literal["ok", "error"]
    detail:       str | None = None