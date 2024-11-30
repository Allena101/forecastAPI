from darts.models import TCNModel
from darts import TimeSeries, concatenate
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr
import matplotlib.pyplot as plt
from darts.dataprocessing.transformers import Scaler
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple
from pathlib import Path


def darts_forecast(
    edf: pd.DataFrame, target: str, start_dt: str, end_dt: str, pred_horizon: int
) -> Tuple[str, list]:

    # creates date range for the same period between start_date and end_date
    date_range_past = pd.date_range(start=start_dt, end=end_dt, freq="1h")

    # uses start_date and end_date query parameters to create a ts object with timeSeries exogenous variables
    ts_time_and_holiday_past_covariates = concatenate(
        [
            dt_attr(date_range_past, "month", dtype=np.float32) / 12,
            (dt_attr(date_range_past, "year", dtype=np.float32) - 1948) / 12,
            dt_attr(date_range_past, "day", dtype=np.float32) / 52,
        ],
        axis="component",
    )

    # Adds a holiday feature
    ts_time_and_holiday_past_covariates = (
        ts_time_and_holiday_past_covariates.add_holidays("SE")
    )

    # Feature engineers two rolling mean features
    edf["rolling_mean_24h"] = edf[target].rolling(window=96, center=False).mean()
    edf["EWMA_24h"] = edf[target].ewm(span=96).mean()
    # Can result in some nan values that has to be imputed
    edf.rolling_mean_24h.fillna(method="bfill", inplace=True)
    edf.EWMA_24h.fillna(method="bfill", inplace=True)

    # Separates the target features and the other features so that they can be scaled separately
    edf_cols_without_DT = list(edf.columns)
    edf_cols_without_DT.remove("DT")
    cols_without_target_and_DT = [col for col in edf_cols_without_DT if col != target]
    # print(f"{edf_cols_without_DT=}")
    # print(f"{cols_without_target_and_DT=}")

    # Converts the dataFrame into a darts timeSeries object
    ts = TimeSeries.from_dataframe(edf, "DT", edf_cols_without_DT)
    ts = ts.resample(freq="1h")  #  offset=pd.Timedelta("30min")

    # Scales the target feature and the other features seperately so that the target feature can be reversed/inversed
    target_scaler = Scaler()
    covariate_scaler = Scaler()
    train_scaled = target_scaler.fit_transform(ts[target])
    covariates_scaled = covariate_scaler.fit_transform(ts[cols_without_target_and_DT])

    # TCNModel proved to be effective and not to resource heavy
    TCN_Model = TCNModel(
        input_chunk_length=25,  # 170
        output_chunk_length=24,  # 168
        n_epochs=3,  # 15
        num_filters=15,  # 15
        kernel_size=3,  # 5
    )

    # Combines the scaled features and the time/calendar features
    covariates_concat = covariates_scaled.concatenate(
        ts_time_and_holiday_past_covariates,
        axis=1,
        ignore_time_axis=True,
    )

    # Train/Fits the model. TCN_Model only support past_covariates
    print("Starting model training")
    TCN_Model.fit(
        train_scaled,
        past_covariates=covariates_concat,
    )
    print("Model training complete")

    # generates predictions from the model. The query parameter pred_horizon is used to set the amount of time steps that will be forecasted
    preds = TCN_Model.predict(pred_horizon)

    # print("The preds:")
    # print(type(preds))

    # use the scaler to inverse the predictions to its original scale
    preds_inverse = target_scaler.inverse_transform(preds)
    # converts the darts timeSeries prediction object into a list that can be returned as json
    preds_inversed_series = preds_inverse.pd_series()
    preds_inversed_list = preds_inversed_series.to_list()
    print(f"{preds_inversed_list=}")

    last_time_step = ts[-1].end_time()
    print(last_time_step)
    forecast_list = []
    # creates a list of dateTime that starts one time step after the last value in the training data.
    last_time_step = last_time_step + timedelta(hours=1)
    for i in range(len(preds_inversed_list)):
        # Add one hour to the starting datetime for each iteration
        last_time_step = last_time_step + timedelta(hours=i)
        last_time_step_DT = last_time_step.to_pydatetime()
        forecast_list.append(last_time_step_DT)
    print(forecast_list)

    # plots the predictions and saves the plot in the plots folder.
    pltt = preds_inverse.plot()
    pltt.get_figure().savefig("plots/myForecastPlot.png")
    pltt.cla()
    # get the path of the saved plot image
    image_path = Path("plots/myForecastPlot.png")

    predictions_with_dateTime = {
        "dateTime": forecast_list,
        "forecast": preds_inversed_list,
    }

    # Sends plot image without using fileresponse
    # with open(image_path, "rb") as image_file:
    #     binary_data = image_file.read()
    # base64_data = base64.b64encode(binary_data).decode("utf-8")

    # returns the image path to the saved plot as well as the forecast
    print("Darts script has completed")
    return image_path, predictions_with_dateTime
