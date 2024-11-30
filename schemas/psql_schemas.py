from enum import Enum
from typing import List
from pydantic import BaseModel, Field, field_validator, validator
from datetime import datetime
from fastapi import (
    HTTPException,
    status,
    Query,
)


class SeriesNum(str, Enum):
    kWh_default = ""
    kwh1 = "KWhT1"
    kwh2 = "KWhT2"
    kwh3 = "KWhT3"
    kwh4 = "KWhT4"
    iSum1 = "iSum1"
    iSum2 = "iSum2"
    iSum3 = "iSum3"
    iSum4 = "iSum4"


class Series_iSum(str, Enum):
    kWh_default = ""
    iSum1 = "iSum1"
    iSum2 = "iSum2"
    iSum3 = "iSum3"
    iSum4 = "iSum4"


class Series_KWh_total(str, Enum):
    kWh_default = ""
    kwh1 = "KWhT1"
    kwh2 = "KWhT2"
    kwh3 = "KWhT3"
    kwh4 = "KWhT4"


class CollectionNum(str, Enum):
    kWh_default = ""
    elmätare_1 = "elmätare_1"
    elmätare_2 = "elmätare_2"
    elmätare_3 = "elmätare_3"
    elmätare_4 = "elmätare_4"


class CityEnum(str, Enum):
    city_default = ""
    Stockholm = "Stockholm"
    Karlstad = "Karlstad"


class GoogleEnum(str, Enum):
    google_default = ""
    el = "el"
    hyra = "hyra"
    skatt = "skatt"


class GoogleEnumList(str, Enum):
    google_default = ""
    el = "el"
    hyra = "hyra"
    skatt = "skatt"


class CommonQueryParams(BaseModel):
    seriesNum: SeriesNum = Field(
        Query(
            "",
            description="Please pick ONLY one of the 4 available KWh series or iSum series",
        )
    )
    city_name: CityEnum = Field(
        Query("", description="Please pick ONLY one of the 2 available cities")
    )

    google: List[GoogleEnumList] = Field(
        Query([], description="Please pick any of the 3 available google searches")
    )
    start_dt: str = Field(
        Query(
            # default="2021-06-18 10",
            description="Start date for timeSeries",
            example="2020-06-18 10",
        )
    )
    end_dt: str = Field(
        Query(
            # "2024-01-12 12",
            description="End date for timeSeries",
            example="2024-01-12 12",
        )
    )
    pred_horizon: int = Field(
        Query(
            7,  # 7 days
            gt=0,
            lt=32,
            description="How many timeSteps the model will forecast",
        )
    )
    show_image: bool = Field(
        Query(False, description="Returns image using fastAPI FileResponse")
    )
    stockmarket: bool = Field(
        Query(False, description="Use omxs30 timeSeries data as covariate")
    )

    @validator("seriesNum")
    def validate_series_num(cls, v):
        if not v:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="seriesNum is required",
            )
        return v

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


# ? Not used (cold be added to get searchSeries and get_KWh_series endpoints)
class onlySeriesParams(BaseModel):
    seriesNum: SeriesNum = Field(
        Query(
            "",
            description="Please pick ONLY one of the 4 available KWh series or iSum series",
        )
    )
    start_dt: str = Field(
        Query(
            "2020-06-18 10",
            description="Start date for timeSeries",
            example="2021-11-21 14",
        )
    )
    end_dt: str = Field(
        Query(
            "2024-01-12 12",
            description="End date for timeSeries",
            example="2022-11-21 14",
        )
    )

    @validator("seriesNum")
    def validate_series_num(cls, v):
        if not v:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="seriesNum is required",
            )
        return v

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


class stl_query_params(BaseModel):
    seriesNum: SeriesNum = Field(
        Query(
            "",
            description="Please pick ONLY one of the 4 available KWh series or iSum series",
        )
    )

    start_dt: str = Field(
        Query(
            "2022-03-05 14",
            description="Start date for timeSeries",
            example="2022-03-05 14",
        )
    )
    end_dt: str = Field(
        Query(
            "2023-09-01 12",
            description="End date for timeSeries",
            example="2023-09-01 12",
        )
    )

    show_plot: bool = Field(
        Query(False, description="Returns plot image using fastAPI FileResponse")
    )

    return_interval_sum: bool = Field(
        Query(
            True,
            description="If interval sum (iSum) or KWh_total should be imputed and returned. False returns KWh_total",
        )
    )

    @validator("seriesNum")
    def validate_series_num(cls, v):
        if not v:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="seriesNum is required",
            )
        return v

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


class forecast_query_params(BaseModel):
    series_iSum: Series_iSum = Field(
        Query(
            "",
            description="Please pick ONLY one of the 4 available iSum series.",
            example="iSum2",
        )
    )
    city_name: CityEnum = Field(
        Query("", description="Please pick ONLY one of the 2 available cities")
    )

    google: List[GoogleEnumList] = Field(
        Query([], description="Please pick any of the 3 available google searches")
    )
    start_dt: str = Field(
        Query(
            # default="2021-06-18 10",
            description="Start date for timeSeries",
            example="2020-06-18 10",
        )
    )
    end_dt: str = Field(
        Query(
            # "2024-01-12 12",
            description="End date for timeSeries",
            example="2024-01-12 12",
        )
    )
    pred_horizon: int = Field(
        Query(
            7,  # 7 days
            gt=0,
            lt=32,
            description="How many timeSteps the model will forecast",
        )
    )
    show_image: bool = Field(
        Query(False, description="Returns image using fastAPI FileResponse")
    )
    stockmarket: bool = Field(
        Query(False, description="Use omxs30 timeSeries data as covariate")
    )

    @validator("series_iSum")
    def validate_series_num(cls, v):
        if not v:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="series_iSum is required",
            )
        return v

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
