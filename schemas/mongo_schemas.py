from enum import Enum
from pydantic import BaseModel, Field, field_validator, validator
from datetime import datetime
from typing import Optional, List, Union
from fastapi import (
    HTTPException,
    status,
    Query,
)


class CollectionNum(str, Enum):
    # kWh_default = ""
    elmätare_1 = "elmätare_1"
    elmätare_2 = "elmätare_2"
    elmätare_3 = "elmätare_3"
    elmätare_4 = "elmätare_4"


class query_Series(BaseModel):
    DateTime: datetime
    # Actual: Optional[float] = "nan"
    Actual: Union[float, str]
    Forecast: Union[float, str]  # Allow both float and string


class LimitedQueryParams(BaseModel):

    collectionName: CollectionNum = Field(
        Query(
            "",
            description="Please pick ONLY one of the 4 available iSum collections",
            example="elmätare_2",
        )
    )

    start_dt: str = Field(
        Query(
            "2020-06-18 10",
            description="Start date for timeSeries",
            example="2021-10-21 14",
        )
    )
    end_dt: str = Field(
        Query(
            "2024-01-12 12",
            description="End date for timeSeries",
            example="2022-11-21 15",
        )
    )

    show_plot: bool = Field(
        Query(False, description="Returns plot image using fastAPI FileResponse")
    )

    @validator("start_dt")
    def validate_start_dt(cls, v):
        """Validates that start_dt is in the format yyyy-mm-dd hh."""
        try:
            dt = datetime.strptime(v, "%Y-%m-%d %H")
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"start_dt {v} has to be in the format of YYYY-MM-DD HH. For example: 2022-06-18 10",
            ) from exc

        if dt < datetime(2020, 6, 18) or dt > datetime(2024, 1, 12):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"start_dt cannot be earlier than 2020-06-18 or later than 2024-01-12. Received: {dt}",
            )

        return v

    @validator("end_dt")
    def validate_end_dt(cls, v):
        """Validates that end_dt is in the format yyyy-mm-dd hh."""
        try:
            dt = datetime.strptime(v, "%Y-%m-%d %H")
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"end_dt {v} has to be in the format of YYYY-MM-DD HH \n For example: 2022-06-18 10",
            ) from exc
        if dt < datetime(2020, 6, 18) or dt >= datetime(2024, 1, 13):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"end_dt cannot be earlier than 2020-06-18 or later than 2024-01-12. Received: {dt}",
            )

        return v

    @validator("end_dt")
    def validate_dt_order(cls, v, values):
        """Validates that end_dt is after start_dt."""
        start_dt = values.get("start_dt")
        end_dt = datetime.strptime(v, "%Y-%m-%d %H")
        start_dt = datetime.strptime(start_dt, "%Y-%m-%d %H")
        if end_dt <= start_dt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"end_dt: {end_dt} does not come after start_dt: {start_dt}",
            )
        return v

    @validator("collectionName")
    def validate_collection_name(cls, v):
        if not v:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="collection Name is required",
            )
        return v


# ? For insert many endpoint
class Series(BaseModel):
    DateTime: datetime
    # Actual: Optional[int] = "nan"
    Actual: Optional[float] = (
        "nan"  #! we might how to enforce float when we upload to mongoDB
    )
    Forecast: float


class insert_Series_Many(Series):
    series: List[Series]
