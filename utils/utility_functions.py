from datetime import datetime, timedelta
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import pytz


def timezone_adjust_func(
    start_dt: datetime, end_dt: datetime, print_help: bool = False
) -> tuple[datetime, datetime]:

    # Define Sweden's time zone (CET/CEST)
    sweden_tz = pytz.timezone("Europe/Stockholm")

    # Localize the datetime object to Sweden's timezone
    start_dt_obj_with_tz = sweden_tz.localize(start_dt)
    end_dt_obj_with_tz = sweden_tz.localize(end_dt)

    if print_help:
        print(f"{start_dt=}")
        print(f"{end_dt=}")
        print(f"{start_dt_obj_with_tz=}")
        print(f"{end_dt_obj_with_tz=}")

    #! Start_dt
    # Check if the datetime is in daylight saving time (CEST, UTC+2) or not (CET, UTC+1)
    if start_dt_obj_with_tz.dst() != timedelta(0):  # DST is not zero for summer (CEST)
        if print_help:
            print("START_DT START_DT START_DT START_DT START_DT")
            print("It is in summer time fro start_dt (CEST, UTC+2)")
        start_dt_obj_with_tz_adjusted = start_dt_obj_with_tz + timedelta(hours=2)
    else:  # Winter time (CET, UTC+1)
        if print_help:
            print("START_DT START_DT START_DT START_DT START_DT")
            print("It is in winter time for start_dt (CET, UTC+1)")
        start_dt_obj_with_tz_adjusted = start_dt_obj_with_tz + timedelta(hours=1)

    if print_help:
        print(f"Original time: {start_dt_obj_with_tz}")
        print(f"Time after adding 1 hour: {start_dt_obj_with_tz_adjusted}")

    #! end_dt
    # Check if the datetime is in daylight saving time (CEST, UTC+2) or not (CET, UTC+1)
    if end_dt_obj_with_tz.dst() != timedelta(0):  # DST is not zero for summer (CEST)
        if print_help:
            print("END_DT END_DT END_DT END_DT END_DT")
            print("It is in summer time for end_dt (CEST, UTC+2)")
        end_dt_obj_with_tz_adjusted = end_dt_obj_with_tz + timedelta(hours=2)
    else:  # Winter time (CET, UTC+1)
        if print_help:
            print("END_DT END_DT END_DT END_DT END_DT")
            print("It is in winter time for end_dt (CET, UTC+1)")
        end_dt_obj_with_tz_adjusted = end_dt_obj_with_tz + timedelta(hours=1)

    if print_help:
        print(f"Original time: {end_dt_obj_with_tz}")
        print(f"Time after adding 2 hour: {end_dt_obj_with_tz_adjusted}")

    return start_dt_obj_with_tz_adjusted, end_dt_obj_with_tz_adjusted


def return_plot_func(df: pd.DataFrame, column: List[str]) -> str:
    """
    Plots specified columns (either iSum or KWh_total from any of the Forecast or Actual series) from a DataFrame and saves the plot as an image.

    Parameters:
    - df: A pandas DataFrame containing the data.
    - column: A list of column names to plot from the DataFrame.

    Returns:
    - The file path to the saved plot image.
    """
    df = pd.DataFrame(df)
    # print(df.head())
    # print(df.tail())
    df.set_index("DateTime", inplace=True)

    ax = df[column].plot()

    ax.legend(column, loc="best", fontsize=15, facecolor="white", edgecolor="blue")

    if len(column) == 1:
        ax.set_title(column[0], fontsize=18, color="green", pad=15)
        ax.get_legend().remove()

    if len(column) == 2:
        if "Forecast" in df.columns:
            # print("The 'Forecast' column exists.")
            if df["Forecast"].isna().all():
                print(
                    "Forecast field has only nan values and will thus not be plotted."
                )
                ax.get_legend().get_texts()[0].set_color("black")
        else:
            ax.get_legend().get_texts()[0].set_color("black")
            ax.get_legend().get_texts()[1].set_color("blue")

    plt.savefig("plots/myPlot.png")
    image_path = "plots/myPlot.png"
    print("Returning mongoDB collection as a plot")
    return image_path


def return_KWh_total_series(ml: List[float]) -> List[float]:
    """
    Calculate the cumulative sum (kWh total) of a series of energy values.

    Args:
        ml (List[float]): A list of energy values in kWh (e.g., hourly consumption data).

    Returns:
        List[float]: A list where each element is the cumulative sum of the input list up to that index.
    """

    iSum = []

    for ix, i in enumerate(ml):
        if ix == 0:
            iSum.append(i)
            continue
        iSum_value = iSum[-1] + i
        iSum.append(iSum_value)

    return iSum
