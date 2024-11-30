# Standard Library Imports
from typing import List
from datetime import datetime, timedelta

# Third-Party Library Imports
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import FileResponse
from sqlmodel import select
from bson import ObjectId
import pandas as pd
import matplotlib.pyplot as plt

# Local Application Imports
# Database modules
from database.mongo_database import (
    get_mongo_client,
    individual_series,
)
from database.psql_database import get_session, Session

# Schema definitions
from schemas.mongo_schemas import (
    query_Series,
    LimitedQueryParams,
)
from schemas.psql_schemas import (
    forecast_query_params,
    stl_query_params,
    onlySeriesParams,
)

# Models
from models.psql_models import (
    iSum,
    KWh,
    Weather,
    gTrend,
    omxs30,
    seriesDict,
    weatherDict,
    gTrendDict,
    google_search_list_full,
    city_name_list,
)

# Utility functions
from utils.utility_functions import (
    timezone_adjust_func,
    return_plot_func,
    return_KWh_total_series,
)

# Forecasting and STL Imputation
from DARTS.model_TCN import darts_forecast
from stl_impute.stl_impute_script import stl_impute_func


router = APIRouter()


# for pinging the server
@router.get("/")
def read_root():
    return {"fastAPI status": "Running"}


# for creating a limited forecasting model
@router.get("/forecast")
def forecast(
    db: Session = Depends(get_session),
    query_params: forecast_query_params = Depends(),
):
    # For holding table/column names that should be merged
    merge_list = []

    # Create datetime objects from the query parameter strings
    # print("start_dt =", query_params.start_dt)
    # print("end_dt =", query_params.end_dt)
    start_dt_str = query_params.start_dt
    end_dt_str = query_params.end_dt
    start_dt_obj = datetime.strptime(start_dt_str, "%Y-%m-%d %H")
    end_dt_obj = datetime.strptime(end_dt_str, "%Y-%m-%d %H")

    # function that adjusts the datetime based on the current timezones in Sweden
    # The function uses dst() to determine how many timedelta hours to add
    start_dt_tz_adjusted, end_dt_tz_adjusted = timezone_adjust_func(
        start_dt_obj, end_dt_obj, print_help=True
    )

    if not query_params.series_iSum:
        raise HTTPException(
            status_code=404,
            detail="seriesNum is required. Please query only for one of these four series: iSum1, iSum2, iSum3, or iSum4",
        )

    #! START iSum block
    # print("ISUM ISUM ISUM ISUM ISUM")
    # Gets value of chosen series from the query parameters
    iSum_string = query_params.series_iSum
    statement = (
        select(iSum.dt, seriesDict[iSum_string])
        .where(iSum.dt >= start_dt_tz_adjusted)  # start_dt_with_tz
        .where(iSum.dt <= end_dt_tz_adjusted)  # end_dt_with_tz
        # .limit(5)  # remove limit later
    )
    iSum_list = []
    results = db.exec(statement)
    # Convert each result object to a dictionary (otherwise sqlalchemy objects)
    for result in results:
        # print(result)
        result_dict = result._asdict()
        iSum_list.append(result_dict)

    data = {}
    # Converts the data from the database
    for item in iSum_list:
        # Get the keys of the current item
        keys = list(item.keys())
        # Iterate over the keys and add them to the data dictionary
        for key in keys:
            if key not in data:
                data[key] = []
            data[key].append(item[key])

    # Converts data into a dataFrmae data type and set dt to dateTime
    iSum_df = pd.DataFrame(data)
    iSum_df["dt"] = pd.to_datetime(iSum_df["dt"], utc=True)
    iSum_df["dt"] = iSum_df["dt"].dt.tz_localize(None)
    iSum_df.rename(columns={"dt": "DT"}, inplace=True)
    # print(iSum_df.tail())
    #! END iSum block

    # df_head = iSum_df.head(10)
    # df_dict = iSum_df.to_dict()
    # df_dict = iSum_df.to_dict()
    print(iSum_df.head())
    print(iSum_df.tail())

    #! START weather block
    weather_df = None
    # check if value of city name query paramater.
    # If user dont select a city and empty string ("") becomes the default value
    city_string = query_params.city_name
    print("string value = ", city_string)

    # If the default empty string value is present then the if condition will not be met since an empty string evaluates to false in python
    # If the if condition is not met that means that the user did not select any weather query parameters
    if city_string:
        if query_params.city_name.value not in city_name_list:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"City name: {query_params.city_name.value} is not in the database. \nCurrently there is only data for Stockholm and Karlstad",
            )

        # if query paramater of "Stockholm" is selected then the relevant field names will be added to the weather_list
        if query_params.city_name == "Stockholm":
            weather_list = ["Sthlm_temp", "Sthlm_precip"]
        if query_params.city_name == "Karlstad":
            weather_list = ["Ksd_temp", "Ksd_precip"]

        # based on the column names the appropriate class.attributes are appended to the list from the dictionary. e.g. "Sthlm_temp" : Weather.Sthlm_temp
        weather_string_list = []
        for i in weather_list:
            weather_string_list.append(weatherDict[i])

        # Using SQLModel to select first the selected columns and then filter (i.e. where) on the start_dt and end_dt time interval
        statement = (
            select(Weather.dt, *weather_string_list)
            .where(Weather.dt >= start_dt_tz_adjusted)
            .where(Weather.dt <= end_dt_tz_adjusted)
            # .limit(5)  # remove limit later
        )
        weather_data_list = []
        results = db.exec(statement)
        # Convert each result object to a dictionary
        for result in results:
            # print(result)
            result_dict = result._asdict()
            weather_data_list.append(result_dict)

        # Unpacks the the dictionaries so that they are in a format that can be converted to a dataFrame
        weather_dataFrame_data = {}
        for item in weather_data_list:  # weather_data_list
            # Get the keys of the current item
            keys = list(item.keys())
            # Iterate over the keys and add them to the data dictionary
            for key in keys:
                if key not in weather_dataFrame_data:
                    weather_dataFrame_data[key] = []
                weather_dataFrame_data[key].append(item[key])

        # Converts the weather data into a dataFrame
        # Converts "dt" column to dateTime
        weather_df = pd.DataFrame(weather_dataFrame_data)
        weather_df["dt"] = pd.to_datetime(weather_df["dt"], utc=False)
        # Adds "weather" to the merge_list so it can be added to the merge_df later
        merge_list.append("weather")
        # print(weather_df.tail())
    #! END weather block

    #! START gTrend block
    # Checks if there is a google query parameter
    google_string = query_params.google

    # default value of google query parameter is a list with an empty string
    print(f"{google_string=}")
    if google_string == []:
        google_string = [False]
        print(f"{google_string=}")

    if google_string[0]:
        # print(f"{query_params.google=}")
        # Unpacks the google list so that it just contains strings of the selected search terms
        google_search_list = [
            google_search.value for google_search in query_params.google
        ]

        # An extra check if the user selected search terms that are not in the database
        # only relevant if the user is not using the fastAPI swagger docs
        for i in google_search_list:
            if i not in google_search_list_full or i == "":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Google search: {query_params.google} is not in the database",
                )

        google_search_list_cols = []
        for i in google_search_list:
            google_search_list_cols.append(gTrendDict[i])

        statement = (
            select(gTrend.dt, *google_search_list_cols)
            .where(gTrend.dt >= start_dt_tz_adjusted)
            .where(gTrend.dt <= end_dt_tz_adjusted)
            # .limit(5)  # remove limit later
        )

        google_data_list = []
        results = db.exec(statement)
        for result in results:
            # print(result)
            result_dict = result._asdict()
            google_data_list.append(result_dict)

        google_DataFrame_data = {}
        for item in google_data_list:
            keys = list(item.keys())
            for key in keys:
                if key not in google_DataFrame_data:
                    google_DataFrame_data[key] = []
                google_DataFrame_data[key].append(item[key])

        gTrend_df = pd.DataFrame(google_DataFrame_data)
        gTrend_df["dt"] = pd.to_datetime(gTrend_df["dt"], utc=False)
        print(gTrend_df.head())
        merge_list.append("iot")
    #! END gTrend block

    #! START omxs30 block

    # query_params.stockmarket is a bool
    if query_params.stockmarket:

        statement = (
            select(omxs30.dt, omxs30.open, omxs30.close)
            .where(omxs30.dt >= start_dt_tz_adjusted)
            .where(omxs30.dt <= end_dt_tz_adjusted)
            # .limit(5)  # remove limit later
        )
        results = db.exec(statement)

        stockmarket_data_list = []
        for result in results:
            # print(result)
            result_dict = result._asdict()
            stockmarket_data_list.append(result_dict)

        stockmarket_dataFrame_data = {}
        for item in stockmarket_data_list:
            keys = list(item.keys())
            for key in keys:
                if key not in stockmarket_dataFrame_data:
                    stockmarket_dataFrame_data[key] = []
                stockmarket_dataFrame_data[key].append(item[key])

        stockmarket_df = pd.DataFrame(stockmarket_dataFrame_data)
        stockmarket_df["dt"] = pd.to_datetime(stockmarket_df["dt"], utc=False)
        merge_list.append("omxs30")
        # print(stockmarket_df.tail())
    #! END omxs30 block

    #! Merge_df block START

    # Start df_merge my copying iSum_df since iSum is the target feature and it will it is a required query parameter
    df_merge = iSum_df

    # Checks merge_list if the weather feature has been selected by looking into the merge list for the "weather" string
    if "weather" in merge_list:
        # print("weather in merge list")
        # Merge on the hour attribute
        df_merge = iSum_df.merge(
            weather_df,
            left_on=iSum_df["DT"].dt.strftime("%Y-%m-%d %H"),
            right_on=weather_df["dt"].dt.strftime("%Y-%m-%d %H"),
            how="left",
        ).drop("key_0", axis=1)
        # drops "dt" columns since all tables in the database has that field
        df_merge.drop("dt", axis=1, inplace=True)
        # The rows that did not match in the merge will be set to NaN and those values can be imputated. Here the back fill method is used
        df_merge.bfill(inplace=True)
        # print(df_merge.tail())

    if "omxs30" in merge_list:
        # print("omxs30 in merge_list")
        df_merge = df_merge.merge(
            stockmarket_df,
            # matching on the day dateTime attribute
            left_on=df_merge["DT"].dt.strftime("%Y-%m-%d"),
            right_on=stockmarket_df["dt"].dt.strftime("%Y-%m-%d"),
            how="left",
        ).drop("key_0", axis=1)

        df_merge.drop("dt", axis=1, inplace=True)
        df_merge = df_merge.infer_objects(copy=False)  # Analyze data types
        # Merging on day will results in lots of nan values since iSum has the frequency of 15m and stock market has less than daily frequency
        df_merge.ffill(inplace=True)
        df_merge.bfill(inplace=True)

    if "iot" in merge_list:
        print("iot in merge list")
        df_merge = df_merge.merge(
            gTrend_df,
            left_on=df_merge["DT"].dt.strftime("%Y-%m-%d"),
            right_on=gTrend_df["dt"].dt.strftime("%Y-%m-%d"),
            how="left",
        ).drop("key_0", axis=1)

        df_merge.drop("dt", axis=1, inplace=True)
        df_merge.bfill(inplace=True)
        df_merge.ffill(inplace=True)

    print("Finished merging tables:")
    print(df_merge.tail())

    # Checks if there are any nan values in df_merge
    has_nan = df_merge.isna().any()
    if has_nan.any():
        print("There are NaN values in the DataFrame")
    else:
        print("There are no NaN values in the DataFrame")
    # ! Merge_df block END

    # ! START Darts block

    # Function that doe the darts model training on the selected psql database data
    # df_merge and some the relevant query parameters are the function parameters
    print("Running darts_forecast_func_uni script")
    image_path, preds_inverse_list = darts_forecast(
        df_merge,
        query_params.series_iSum.value,
        start_dt_obj,
        end_dt_obj,
        query_params.pred_horizon,
    )
    # print(preds_inverse_list)

    if query_params.show_image:
        return FileResponse(
            image_path, headers={"Predictions": str(preds_inverse_list)}
        )

    if not query_params.show_image:
        return {"Predictions": str(preds_inverse_list)}


# for imputing missing values based on known future accumulated data
@router.get("/stl_imputation/")
async def stl_imputation(
    db: Session = Depends(get_session),
    query_params: stl_query_params = Depends(),
):

    start_dt_stl = query_params.start_dt
    end_dt_stl = query_params.end_dt

    secTZ: str = ":00:00+02"
    start_dt_full = query_params.start_dt + secTZ
    end_dt_full = query_params.end_dt + secTZ
    start_dt_with_tz = datetime.strptime(start_dt_full, "%Y-%m-%d %H:%M:%S+02")
    end_dt_with_tz = datetime.strptime(end_dt_full, "%Y-%m-%d %H:%M:%S+02")

    # Subtract one year
    # start_dt_with_tz_minus_one_year = start_dt_with_tz - timedelta(days=365)
    start_dt_with_tz_minus_one_year = start_dt_with_tz - timedelta(days=365)
    end_dt_with_tz_minus_one_year = end_dt_with_tz - timedelta(days=365)

    # Convert the string to a datetime object
    start_dt_obj = datetime.strptime(start_dt_stl, "%Y-%m-%d %H")
    end_dt_obj = datetime.strptime(end_dt_stl, "%Y-%m-%d %H")

    # Subtract one year
    # start_dt_minus_one_year_obj = start_dt_obj - timedelta(days=365)
    start_dt_minus_one_year_obj = start_dt_obj - timedelta(days=365)
    end_dt_minus_one_year_obj = end_dt_obj - timedelta(days=365)

    if not query_params.seriesNum:
        raise HTTPException(
            status_code=404,
            detail="seriesNum is required. Please query only for one of these four series: iSum1, iSum2, iSum3, or iSum4, KWhT1, KWhT2, KWhT3, KWhT4",
        )

    # Use the query parameter to know which pSQL table to query the data from
    iSum_string = query_params.seriesNum
    # print(f"{iSum_string=}")

    if iSum_string in ["KWhT1", "KWhT2", "KWhT3", "KWhT4"]:
        print("Querying KWh-total table")
        # Queries the value for one year prior to start_dt which is needed for stl imputation
        statement = select(KWh.dt, seriesDict[iSum_string]).where(
            KWh.dt == start_dt_with_tz_minus_one_year
        )
        results = db.exec(statement)
        any_result = results.first()

        if any_result is not None:
            print("Found a result for start_dt minus one year:")
            print(f"{any_result=}")

        if any_result is None:
            print(
                f"There is not data one year prior to {start_dt_stl} which is needed for STL imputation"
            )
            raise HTTPException(
                status_code=404,
                detail="Found a result for start_dt minus one year:",
            )

        # Gets value of chosen series from the query parameters
        iSum_string = query_params.seriesNum
        statement = (
            select(KWh.dt, seriesDict[iSum_string])
            # .where(KWh.dt >= start_dt_with_tz) #? Would be more proper to check here already if there is data for at least one year back
            .where(KWh.dt <= end_dt_with_tz)
        )

        iSum_list = []
        results = db.exec(statement)
        # Convert each result object to a dictionary (otherwise sqlalchemy objects)
        for result in results:
            # print(result)
            result_dict = result._asdict()
            iSum_list.append(result_dict)

        data = {}
        # Converts the data from the database
        for item in iSum_list:
            # Get the keys of the current item
            keys = list(item.keys())
            # Iterate over the keys and add them to the data dictionary
            for key in keys:
                if key not in data:
                    data[key] = []
                data[key].append(item[key])

        # Converts data into a dataFrmae data type and set dt to dateTime
        iSum_df = pd.DataFrame(data)
        iSum_df["dt"] = pd.to_datetime(iSum_df["dt"], utc=True)
        iSum_df["dt"] = iSum_df["dt"].dt.tz_localize(None)
        iSum_df.rename(columns={"dt": "DT"}, inplace=True)
        # print(iSum_df)

        iSum_string = query_params.seriesNum
        statement = (
            select(KWh.dt, seriesDict[iSum_string])
            # .where(KWh.dt >= start_dt_with_tz_minus_one_year)
            .where(KWh.dt <= end_dt_with_tz_minus_one_year)
        )

        iSum_list = []
        results = db.exec(statement)
        # Convert each result object to a dictionary (otherwise sqlalchemy objects)
        for result in results:
            # print(result)
            result_dict = result._asdict()
            iSum_list.append(result_dict)

        data = {}
        # Converts the data from the database
        for item in iSum_list:
            # Get the keys of the current item
            keys = list(item.keys())
            # Iterate over the keys and add them to the data dictionary
            for key in keys:
                if key not in data:
                    data[key] = []
                data[key].append(item[key])

        # Converts data into a dataFrmae data type and set dt to dateTime
        df_past_year = pd.DataFrame(data)
        df_past_year["dt"] = pd.to_datetime(df_past_year["dt"], utc=True)
        df_past_year["dt"] = df_past_year["dt"].dt.tz_localize(None)
        df_past_year.rename(columns={"dt": "DT"}, inplace=True)
        # print(df_past_year)

    if iSum_string in ["iSum1", "iSum2", "iSum3", "iSum4"]:
        print("Querying iSum table")

        # Queries the value for one year prior to start_dt which is needed for stl imputation
        statement = (
            select(iSum.dt, seriesDict[iSum_string]).where(
                iSum.dt == start_dt_with_tz_minus_one_year
            )
            # .limit(5)  # remove limit later
        )
        results = db.exec(statement)
        any_result = results.first()

        if any_result is not None:
            print("Found a result for start_dt minus one year:")
            print(f"{any_result=}")

        if any_result is None:
            print(
                f"There is not data one year prior to {start_dt_stl} which is needed for STL imputation"
            )
            raise HTTPException(
                status_code=404,
                detail="Found a result for start_dt minus one year:",
            )

        # Gets value of chosen series from the query parameters
        iSum_string = query_params.seriesNum
        statement = (
            select(iSum.dt, seriesDict[iSum_string])
            # .where(iSum.dt >= start_dt_with_tz)
            .where(iSum.dt <= end_dt_with_tz)
        )
        # .limit(5)  # for faster testing

        iSum_list = []
        results = db.exec(statement)
        # Convert each result object to a dictionary (otherwise sqlalchemy objects)
        for result in results:
            # print(result)
            result_dict = result._asdict()
            iSum_list.append(result_dict)

        data = {}
        # Converts the data from the database
        for item in iSum_list:
            # Get the keys of the current item
            keys = list(item.keys())
            # Iterate over the keys and add them to the data dictionary
            for key in keys:
                if key not in data:
                    data[key] = []
                data[key].append(item[key])

        # Converts data into a dataFrame data type and set dt to dateTime
        iSum_df = pd.DataFrame(data)
        iSum_df["dt"] = pd.to_datetime(iSum_df["dt"], utc=True)
        iSum_df["dt"] = iSum_df["dt"].dt.tz_localize(None)
        iSum_df.rename(columns={"dt": "DT"}, inplace=True)
        # print(iSum_df)

        iSum_string = query_params.seriesNum
        statement = (
            select(iSum.dt, seriesDict[iSum_string])
            .where(iSum.dt >= start_dt_with_tz_minus_one_year)
            .where(iSum.dt <= end_dt_with_tz_minus_one_year)
        )

        iSum_list = []
        results = db.exec(statement)
        # Convert each result object to a dictionary (otherwise sqlalchemy objects)
        for result in results:
            # print(result)
            result_dict = result._asdict()
            iSum_list.append(result_dict)

        data = {}
        # Converts the data from the database
        for item in iSum_list:
            # Get the keys of the current item
            keys = list(item.keys())
            # Iterate over the keys and add them to the data dictionary
            for key in keys:
                if key not in data:
                    data[key] = []
                data[key].append(item[key])

        # Converts data into a dataFrmae data type and set dt to dateTime
        df_past_year = pd.DataFrame(data)
        df_past_year["dt"] = pd.to_datetime(df_past_year["dt"], utc=True)
        df_past_year["dt"] = df_past_year["dt"].dt.tz_localize(None)
        df_past_year.rename(columns={"dt": "DT"}, inplace=True)
        # print(df.tail())
        print(df_past_year)

    return_interval_sum = query_params.return_interval_sum
    # print(f"{return_interval_sum=}")

    imupationPlot_path, imputation_df = stl_impute_func(
        iSum_df,
        iSum_string,
        start_dt_obj,
        end_dt_obj,
        start_dt_minus_one_year_obj,
        end_dt_minus_one_year_obj,
        return_interval_sum,
    )

    if query_params.show_plot:
        print("Imputation Plot Returned")
        return FileResponse(imupationPlot_path)

    if not query_params.show_plot:
        print("return STL-imputation as json dictionary")
        # ? Useful if you want to also query and compare with acutals
        df_dict = imputation_df.to_dict()

        return df_dict


# for querying and returning iSum and KWh_total columns from pSQL server
@router.get("/get_KWh_series/")
async def get_KWh_series(
    db: Session = Depends(get_session),
    query_params: onlySeriesParams = Depends(),
    # query_params: CommonQueryParams = Depends(),
):
    print("start_dt =", query_params.start_dt)
    print("end_dt =", query_params.end_dt)

    secTZ: str = ":00:00+02"
    start_dt_full = query_params.start_dt + secTZ
    end_dt_full = query_params.end_dt + secTZ

    start_dt_with_tz = datetime.strptime(start_dt_full, "%Y-%m-%d %H:%M:%S+02")
    end_dt_with_tz = datetime.strptime(end_dt_full, "%Y-%m-%d %H:%M:%S+02")

    iSum_string = query_params.seriesNum
    statement = (
        select(iSum.dt, seriesDict[iSum_string])
        .where(iSum.dt >= start_dt_with_tz)
        .where(iSum.dt <= end_dt_with_tz)
        .limit(10)  # remove limit later #? query the whole series take some time
    )
    result_list = []
    results = db.exec(statement)
    # Convert each result object to a dictionary
    for result in results:
        print(result)
        result_dict = result._asdict()
        result_list.append(result_dict)

    return result_list


# for querying and returning iSum series from mongoDB (includes forecasts)
@router.get("/getSeries")
async def get_series(
    client=Depends(get_mongo_client),
    series_name: str = Query(..., description="Name of collection to query"),
):

    elmatare = client[series_name]

    series = []
    # ? Takes > 1 minute
    # counter = 0
    for time_step in elmatare.find().sort({"DateTime": 1}):
        series.append(individual_series(time_step))
        # counter += 1
        # if counter > 10:
        #     break
    return series


# For querying and returning mongoDB collections between a dateTime range
@router.get("/get_search_series", response_model=List[query_Series])
async def get_search_series(
    client=Depends(get_mongo_client),
    query_params: LimitedQueryParams = Depends(),
):

    # Get MongoDB collection from query param
    elmätare_collection = query_params.collectionName.value
    elmatare = client[elmätare_collection]

    # Define the format according to the string
    date_format = "%Y-%m-%d %H"
    # Convert query param start_dt and end_dt into dateTime
    start_dt_datetime = datetime.strptime(query_params.start_dt, date_format)
    end_dt_datetime = datetime.strptime(query_params.end_dt, date_format)

    # Find documents with DateTime between start and end (inclusive)
    date_filter = {"DateTime": {"$gte": start_dt_datetime, "$lte": end_dt_datetime}}
    results = elmatare.find(date_filter).sort({"DateTime": 1})

    resultList = list(results)

    myList = []
    for i in resultList:
        myList.append(individual_series(i))

    # If show_plot = True then return a plot using pandas dataFrame
    if query_params.show_plot:
        print("Plot == True")
        # print(myList)
        # print(type(myList))
        for dictionary in myList:
            del dictionary["id"]

        image_path = return_plot_func(myList, ["Actual", "Forecast"])
        return FileResponse(image_path)

    # if show plot = False return list of objects/dictionaries from mongoDB
    if not query_params.show_plot:
        # print("Plot == False")
        print("Returning collection from mongoDB as a dictionary")
        return myList
    # ? delete image plot image on shutdown?


# For ONLY updating a mongoDB Forecast field with random numbers
@router.put("/updateSeries")
async def updateSeries(
    client=Depends(get_mongo_client),
    query_params: LimitedQueryParams = Depends(),
):

    from random import random

    # Get MongoDB collection from query param
    elmätare_collection = query_params.collectionName.value
    # print(f"{elmätare_collection=}")
    elmatare = client[elmätare_collection]

    # Just to check the query parameters
    # print("start_dt =", query_params.start_dt)
    # print("end_dt =", query_params.end_dt)
    # test_start_dt = "2020-06-18 08"
    # test_end_dt = "2020-06-18 10"

    # Define the format according to the string
    date_format = "%Y-%m-%d %H"
    # Convert query param start_dt and end_dt into dateTime
    start_dt_datetime = datetime.strptime(query_params.start_dt, date_format)
    end_dt_datetime = datetime.strptime(query_params.end_dt, date_format)

    # Find documents with DateTime between START and END (inclusive)
    date_filter = {"DateTime": {"$gte": start_dt_datetime, "$lte": end_dt_datetime}}
    results = elmatare.find(date_filter).sort({"DateTime": 1})

    resultList = list(results)

    date_filter = {"DateTime": {"$gte": start_dt_datetime, "$lte": end_dt_datetime}}
    results = elmatare.find(date_filter).sort({"DateTime": 1})

    resultList = list(results)

    print(resultList)

    myList = []
    for i in resultList:
        myList.append(individual_series(i))

    random_number_list = []
    for i in range(len(resultList)):
        x = random()
        random_number_list.append(x)

    update_values = random_number_list

    for list_item, update_value in zip(myList, update_values):
        _id = list_item["id"]
        update_operation = {"$set": {"Forecast": update_value}}
        elmatare.update_one({"_id": ObjectId(_id)}, update_operation)
        # if result.modified_count == 1:
        #     print(f"Document with ID {_id} updated successfully.")
        # else:
        #     print(f"Document with ID {_id} not found or not updated.")

    print(f"MongodDB collection {elmätare_collection} has been updated")

    if query_params.show_plot:
        # print("Plot == True")
        df = pd.DataFrame(myList)
        df["Random Forecast"] = random_number_list
        df.set_index("DateTime", inplace=True)
        # df[["Actual", "Forecast"]].plot()
        df[["Actual", "Random Forecast"]].plot()
        plt.savefig("plots/myPlot.png")
        image_path = "plots/myPlot.png"
        # print(df.head())
        return FileResponse(image_path)

    # if show plot = False return list of objects/dictionaries from mongoDB
    if not query_params.show_plot:
        # print("Plot == False")
        for number, ddict in zip(random_number_list, myList):
            ddict["Forecast"] = number

        print(
            "Returning list with dictionaries that contains the actual and randomly generated series"
        )
        return myList


# Just a simple route to upload series data from a saved parquet file
# ? Names of iSum columns = iSum1 , iSum2 , iSum3 , iSum4. Parquet file has to be in root project folder
@router.post("/parquet_to_MongoDB/")
async def pSQL_to_MongoDB(
    client=Depends(get_mongo_client),
    collection_name: str = Query(..., description="Name of MongoDB collection"),
    series_name: str = Query(
        ..., description="Name of pSQL Series to query (e.g. iSum2)"
    ),
):
    elmatare = client[collection_name]

    iSum = pd.read_parquet("iSum.parquet")
    iSum["dt"] = iSum["dt"].dt.tz_localize(None)
    iSum.rename(columns={"dt": "DT"}, inplace=True)
    iSum["idx"] = iSum.index
    iSum.index = iSum["DT"]

    # original timeSeries data has frequency 15m
    iSum_hour = iSum.resample("H")
    iSum_hourly_mean = iSum_hour.mean(numeric_only=True)
    iSum_hourly_mean["DT"] = iSum_hourly_mean.index

    for index, row in iSum_hourly_mean.iterrows():
        # print(index)  # Access the row index
        # print(row)  # Access the row as a Series
        # print(row[series_name])  # Access a specific column
        float_iSum = float(row[series_name])
        new_item = {"DateTime": row["DT"], "Actual": float_iSum, "Forecast": "nan"}
        elmatare.insert_one(new_item)
        # if row["idx"] > 250:
        #     break

    print("Parquet file was successfully uploaded to MongoDB")
    return {"success": "Parquet file contents are now stored as a MongoDB collection"}


# for uploading local parquet filed to mongoDB (that are compatible with the model)
@router.get("/trained_model/")
async def trained_model(
    compare_forecast: bool = Query(
        True,
        description="Set to True to compare the forecast values with the actual values from mongoDB",
    ),
    replace_forecast: bool = Query(
        False,
        description="Set to True to replace the forecast values in the mongoDB collection with the new forecast values",
    ),
    return_plot: bool = Query(
        True,
        description="Set to True to return a plot using fastAPI FileResponse",
    ),
    return_KWh_total: bool = Query(
        True,
        description="Returns the data as KWh_total instead of iSum (makes plotting clearer)",
    ),
    client=Depends(get_mongo_client),
):
    # May be better to load the Darts package before (fastAPI Lifespan)
    # Though not all endpoints use the model
    from darts.models import TCNModel
    from datetime import datetime, timedelta

    # ? relative path
    model_path = r"DARTS\saved_models\TCNModel_2024-10-17_10_21_45.pt"

    # ? # ultimate path
    # model_path = r"C:\Users\Magnus\Desktop\prognosVENV\DARTS\saved_models\TCNModel_2024-10-17_10_21_45.pt"

    # Load previously trained and saved models in the saved_models folder
    loaded_model = TCNModel.load(model_path)

    # This model has been trained to predict electricity usage for November for one smart meter
    nov_preds = loaded_model.predict(720, num_samples=1)  # 24 x 30 (hours in November)

    # Converts Darts Timeseries object to python list
    preds_pd_series = nov_preds.pd_series()
    preds_list = preds_pd_series.to_list()

    # Since likelihood is used there is a slight chance that the models predicts negative values which is not possible. Those values are adjusted to zero/0
    for idx, value in enumerate(preds_list):
        if value < 0:
            preds_list[idx] = 0

    # Creates a list of datetime objects that start one time step after the training period
    datetime_list = []
    last_time_step = datetime(2023, 11, 1, 0, 0, 0)  # 2023-11-01 00:00:00

    for _ in range(len(preds_list)):
        # Add one hour to the starting datetime for each iteration
        # print(f"{last_time_step=}")
        datetime_list.append(last_time_step)
        last_time_step = last_time_step + timedelta(hours=1)
    last_time_step = last_time_step - timedelta(
        hours=1
    )  # In case i want to know the last time step

    # Cerates a dictionary with the datetime forecast/pred lists
    predictions_with_dateTime = {
        "DateTime": datetime_list,
        "Forecast": preds_list,
    }

    print("predictions_with_dateTime dictionary created")
    # print(predictions_with_dateTime)

    # * Compare , replace , plot
    # All flag combination should work

    # Could have made a more elegant solution handling all the flags/bools
    if not compare_forecast:
        print("Compare == False")

        if replace_forecast:
            print("Replace == True")

            # Since this demo model is only trained to predict November we query those that month in the corresponding mongDB collection
            # If this was not a demo model. The corresponding dateTime would have to be saved in a database
            # elmätare_collection = "elmätare_test"
            elmätare_collection = "elmätare_4"
            elmatare = client[elmätare_collection]
            start_dt_datetime = datetime(2023, 11, 1, 0, 0, 0)
            end_dt_datetime = datetime(2023, 11, 30, 23, 0, 0)

            date_filter = {
                "DateTime": {"$gte": start_dt_datetime, "$lte": end_dt_datetime}
            }
            results = elmatare.find(date_filter).sort({"DateTime": 1})
            resultList = list(results)
            myList = []
            for i in resultList:
                myList.append(individual_series(i))
            # print(type(myList)) # myList is a list with dictionaries
            # print(myList)

            # Could have just used preds_list directly
            update_values = preds_list

            for list_item, update_value in zip(myList, update_values):
                _id = list_item["id"]
                update_operation = {"$set": {"Forecast": update_value}}
                elmatare.update_one({"_id": ObjectId(_id)}, update_operation)
                # if result.modified_count == 1:
                #     print(f"Document with ID {_id} updated successfully.")
                # else:
                #     print(f"Document with ID {_id} not found or not updated.")

            print(
                f"Updated the Forecast field for the collection {elmätare_collection}"
            )
            if not return_plot:
                print("Plot == False")
                return predictions_with_dateTime

            # return_plot returns a dataFrame.plot(). Could return a matplotlib plot instead with prettier styling
            # ? Might be a good idea to delete plot after returning it (now it is just over written each time)
            if return_plot:
                print("Plot == True")

                if return_KWh_total:
                    print("KWh_total == True")
                    KWh_total_list = return_KWh_total_series(preds_list)
                    predictions_with_dateTime["Forecast_Kwh"] = KWh_total_list

                    image_path = return_plot_func(
                        predictions_with_dateTime, ["Forecast_Kwh"]
                    )
                    print(
                        "Returned forecast plot using FileResponse. Only plotting forecast converted to KWh_total"
                    )
                    return FileResponse(image_path)

                image_path = return_plot_func(myList, ["Forecast"])
                print(
                    "Returned forecast plot using FileResponse. Only plotting forecast"
                )
                return FileResponse(image_path)

        if not replace_forecast:
            print("Replace == False")

            if not return_plot:
                print("Return forecast as a json")
                return predictions_with_dateTime

            if return_plot:

                if return_KWh_total:
                    print("KWh_total == True")
                    KWh_total_list = return_KWh_total_series(preds_list)
                    predictions_with_dateTime["Forecast_Kwh"] = KWh_total_list
                    image_path = return_plot_func(KWh_total_list, ["Forecast_Kwh"])
                    print(
                        "Returned forecast plot using FileResponse. Only plotting forecast converted to KWh_total"
                    )
                    return FileResponse(image_path)

                image_path = return_plot_func(predictions_with_dateTime, ["Forecast"])
                print(
                    "Returned forecast plot using FileResponse. Only plotting forecast"
                )
                return FileResponse(image_path)

    if compare_forecast and not replace_forecast:
        print("Compare == True")
        print("Replace == False")

        elmätare_collection = "elmätare_4"
        elmatare = client[elmätare_collection]
        start_dt_datetime = datetime(2023, 11, 1, 0, 0, 0)
        end_dt_datetime = datetime(2023, 11, 30, 23, 0, 0)
        date_filter = {"DateTime": {"$gte": start_dt_datetime, "$lte": end_dt_datetime}}
        results = elmatare.find(date_filter).sort({"DateTime": 1})
        resultList = list(results)
        myList = []
        for i in resultList:
            myList.append(individual_series(i))

        actual_values_list = []
        for i in myList:
            # print(i)
            actual_value = i.get("Actual")
            actual_values_list.append(actual_value)

        predictions_with_dateTime["Actual"] = actual_values_list

        if not return_plot:
            # print("Plot == False")
            print("Return forecast and actual series as a json")
            return predictions_with_dateTime

        # actual_values_list
        if return_plot:
            print("Plot == True")

            if return_KWh_total:
                print("KWh_total == True")
                forecast_KWh_total_list = return_KWh_total_series(preds_list)
                actual_KWh_total_list = return_KWh_total_series(actual_values_list)
                predictions_with_dateTime["Forecast_Kwh"] = forecast_KWh_total_list
                predictions_with_dateTime["Actual_Kwh"] = actual_KWh_total_list
                image_path = return_plot_func(
                    predictions_with_dateTime, ["Actual_Kwh", "Forecast_Kwh"]
                )
                print(
                    "Returned forecast and actual plot using FileResponse. Actual and forecast values converted to KWh_total"
                )
                return FileResponse(image_path)

            predictions_with_dateTime["Actual"] = actual_values_list
            image_path = return_plot_func(
                predictions_with_dateTime, ["Actual", "Forecast"]
            )
            print(
                "Returned forecast plot using FileResponse. Plotting forecast and actual values"
            )
            return FileResponse(image_path)

    if compare_forecast and replace_forecast:
        print("Compare == True")
        print("Replace == True")

        # elmätare_collection = "elmätare_test"
        elmätare_collection = "elmätare_4"
        elmatare = client[elmätare_collection]
        start_dt_datetime = datetime(2023, 11, 1, 0, 0, 0)
        end_dt_datetime = datetime(2023, 11, 30, 23, 0, 0)

        date_filter = {"DateTime": {"$gte": start_dt_datetime, "$lte": end_dt_datetime}}
        results = elmatare.find(date_filter).sort({"DateTime": 1})
        resultList = list(results)
        myList = []
        for i in resultList:
            myList.append(individual_series(i))

        actual_values_list = []
        for i in myList:
            # print(i)
            actual_value = i.get("Actual")
            actual_values_list.append(actual_value)

        predictions_with_dateTime["Actual"] = actual_values_list

        update_values = preds_list

        for list_item, update_value in zip(myList, update_values):
            _id = list_item["id"]
            update_operation = {"$set": {"Forecast": update_value}}
            elmatare.update_one({"_id": ObjectId(_id)}, update_operation)
            # if result.modified_count == 1:
            #     print(f"Document with ID {_id} updated successfully.")
            # else:
            #     print(f"Document with ID {_id} not found or not updated.")

        print(f"Updated the Forecast field for the collection {elmätare_collection}")
        if not return_plot:
            # print("Plot == False")
            print("Return forecast and actual series as a json")
            return predictions_with_dateTime

        if return_plot:
            print("Plot == True")

            if return_KWh_total:
                print("KWh_total == True")
                forecast_KWh_total_list = return_KWh_total_series(preds_list)
                actual_KWh_total_list = return_KWh_total_series(actual_values_list)
                predictions_with_dateTime["Forecast_Kwh"] = forecast_KWh_total_list
                predictions_with_dateTime["Actual_Kwh"] = actual_KWh_total_list
                image_path = return_plot_func(
                    predictions_with_dateTime, ["Actual_Kwh", "Forecast_Kwh"]
                )
                print(
                    "Returned forecast and actual plot using FileResponse. Actual and forecast values converted to KWh_total"
                )
                df = pd.DataFrame(predictions_with_dateTime)
                print(df[["Actual_Kwh", "Forecast_Kwh"]].head())
                print(df[["Actual_Kwh", "Forecast_Kwh"]].tail())
                return FileResponse(image_path)

            predictions_with_dateTime["Actual"] = actual_values_list
            image_path = return_plot_func(
                predictions_with_dateTime, ["Actual", "Forecast"]
            )
            print(
                "Returned forecast plot using FileResponse. Plotting forecast and actual values"
            )
            return FileResponse(image_path)
