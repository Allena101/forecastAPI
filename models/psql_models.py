from typing import List, Optional
from decimal import Decimal
from sqlmodel import Field, SQLModel
from datetime import datetime


class iSum(SQLModel, table=True):
    __tablename__ = "iSum"
    id: Optional[int] = Field(default=None, primary_key=True)
    dt: datetime = Field(nullable=False)
    iSum1: Decimal = Field(default=0, max_digits=5, decimal_places=3)
    iSum2: Decimal = Field(default=0, max_digits=5, decimal_places=3)
    iSum3: Decimal = Field(default=0, max_digits=5, decimal_places=3)
    iSum4: Decimal = Field(default=0, max_digits=5, decimal_places=3)
    # iSum4: float = Field() Using float type is also possible
    # dt: datetime = Field(sa_column=sqlmodel.Column(sqlmodel.DateTime(timezone=True), nullable=False)) # If you want to enforce timezone
    # ? In retrospect it was not a good idea to use Decimal. Had to convert the data to float to use it


class KWh(SQLModel, table=True):
    __tablename__ = "eKWh"
    id: Optional[int] = Field(default=None, primary_key=True)
    dt: datetime = Field(nullable=False)
    KWhT1: Decimal = Field(default=0, max_digits=5, decimal_places=3)
    KWhT2: Decimal = Field(default=0, max_digits=5, decimal_places=3)
    KWhT3: Decimal = Field(default=0, max_digits=5, decimal_places=3)
    KWhT4: Decimal = Field(default=0, max_digits=5, decimal_places=3)


class Weather(SQLModel, table=True):
    __tablename__ = "weather"
    id: Optional[int] = Field(default=None, primary_key=True)
    dt: datetime = Field(nullable=False)
    Ksd_temp: Decimal = Field(nullable=False)
    Ksd_precip: Decimal = Field(nullable=False)
    Sthlm_temp: Decimal = Field(nullable=False)
    Sthlm_precip: Decimal = Field(nullable=False)


class gTrend(SQLModel, table=True):
    __tablename__ = "gTrend"
    id: Optional[int] = Field(default=None, primary_key=True)
    dt: datetime = Field(nullable=False)
    skatt: str = Field(nullable=False)
    el: str = Field(nullable=False)
    hyra: str = Field(nullable=False)


class omxs30(SQLModel, table=True):
    __tablename__ = "omxs30"
    id: Optional[int] = Field(default=None, primary_key=True)
    dt: datetime = Field(nullable=False)
    open: Decimal = Field(nullable=False)
    close: Decimal = Field(nullable=False)


seriesDict = {
    "iSum1": iSum.iSum1,
    "iSum2": iSum.iSum2,
    "iSum3": iSum.iSum3,
    "iSum4": iSum.iSum4,
    "KWhT1": KWh.KWhT1,
    "KWhT2": KWh.KWhT2,
    "KWhT3": KWh.KWhT3,
    "KWhT4": KWh.KWhT4,
}

weatherDict = {
    "Ksd_temp": Weather.Ksd_temp,
    "Ksd_precip": Weather.Ksd_precip,
    "Sthlm_temp": Weather.Sthlm_temp,
    "Sthlm_precip": Weather.Sthlm_precip,
}

gTrendDict = {
    "skatt": gTrend.skatt,
    "el": gTrend.el,
    "hyra": gTrend.hyra,
}

google_search_list_full = ["skatt", "el", "hyra"]
city_name_list = ["Stockholm", "Karlstad"]
