import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from datetime import datetime, timedelta
from pathlib import Path


def stl_impute_func(
    df: pd.DataFrame,
    target: str,
    start_dt_obj: datetime,
    end_dt_obj: datetime,
    start_dt_minus_one_year_obj: datetime,
    end_dt_minus_one_year_obj: datetime,
    return_interval_sum: bool,
) -> str:

    print("Start of STL-imputation script")

    # Format the input so its easer to work with
    df = df.copy()
    df.set_index("DT", inplace=True)
    df = df[target].to_frame()
    # interpolation does not work if the data type is Decimal for some reason ...
    # so have to cast Decimal to float
    df[target] = df[target].astype(float)
    # Save an rename original data (from the database)
    df["original"] = df[target]

    # used since there is a +2 hour time zone difference in the database
    # Create a timedelta object representing 2 hours
    two_hours = timedelta(hours=2)
    # Subtract the timedelta from the datetime object
    start_dt_obj = start_dt_obj - two_hours
    end_dt_obj = end_dt_obj - two_hours
    start_dt_minus_one_year_obj = start_dt_minus_one_year_obj - two_hours
    end_dt_minus_one_year_obj = end_dt_minus_one_year_obj - two_hours

    # Makes sure the last value (which is used to for the original linear interpolation) is not converted to a nan value
    one_hour = timedelta(hours=1)
    end_dt_obj_with_value = end_dt_obj - one_hour
    # creates a mask between start_dt and end_dt that is then used to change the values in the dataFrame into nan values.
    nan_mask = (df.index >= start_dt_obj) & (df.index <= end_dt_obj_with_value)
    df.loc[nan_mask, target] = np.nan

    # The created nan period is linearly interpolated to simulate missing data.
    # df[target].interpolate(method="linear", limit_direction="both", inplace=True)
    df[target] = df[target].interpolate(method="linear", limit_direction="both")

    # safety check if there are any nan values for some reason
    nan_count = df[target].isna().sum()
    if nan_count:
        print(f"Fount nan values! Amount detected = {nan_count}")

    # uses STL to get season, trend, and residual decomposition components
    period = 4  # period has to do with the frequency of the data
    stl = STL(df[target], period=period)
    result = stl.fit()
    seasonal, trend, resid = result.seasonal, result.trend, result.resid
    # Create new dataFrame so we dont risk overwriting anything
    stl_df = pd.DataFrame(df[target])

    # Add the seasonal, trend, and resid columns to the new DataFrame
    # Also adds the original and linearly interpolated columns
    stl_df["seasonal"] = seasonal
    stl_df["trend"] = trend
    stl_df["resid"] = resid
    stl_df["linear"] = df[target]
    stl_df["original"] = df["original"]

    # ? Bug detecting code to check the dateTime objects
    # print(f"{start_dt_obj=}")
    # print(f"{end_dt_obj=}")
    # print(f"{start_dt_minus_one_year_obj=}")
    # print(f"{end_dt_minus_one_year_obj=}")
    # print(f"{end_dt_obj_with_value=}")

    # needed for later Differencing of the original series
    start_dt_obj_minus_one_hour = start_dt_obj - one_hour
    # print(f"{start_dt_obj_minus_one_hour=}")

    # Uses the datetime object of the current year and the past year so that the seasonality from the past year can be used to shape the interpolated current year
    # Can be issues with leap year
    nan_val = stl_df[start_dt_obj:end_dt_obj]
    past_year = stl_df[start_dt_minus_one_year_obj:end_dt_minus_one_year_obj]

    # ? Bug detecting code to check that the years have have the same length
    # print(len(nan_val))
    # print(len(past_year))

    # selects only the seasonal feature from the selected date range
    past_year_season = past_year["seasonal"]
    seasonal_past_year_list = past_year_season.values
    # not flattening the list will raise shape mismatch error
    seasonal_past_year_list = seasonal_past_year_list.flatten()

    # Does the actual replacement
    # the seasonal values from the selected period are replaced with the seasonal values form the previous year
    stl_df.loc[nan_val.index, "seasonal"] = seasonal_past_year_list
    # stl_impute column is creating by adding back together the time decomposition components (where the seasonal component is the one from the past year)
    # df and stl_df need to have the same index type (i.e. datetime or range index for this to work)
    stl_df["stl_impute"] = stl_df.seasonal + stl_df.trend + stl_df.resid
    # print("Imputation from previous year seasonality has completed"

    # Again selects only the current year period (that has the old seasonal values)
    imputation_period_df = stl_df[start_dt_obj:end_dt_obj].copy()

    # The goal sum is used to check that the new stl_imputed has the same sum value as the old original column
    goal_sum = imputation_period_df.original.sum()
    # print("Goal sum:")
    # print(f"{goal_sum=}")

    # current_sum is compared with the goal/original sum
    current_sum = imputation_period_df.stl_impute.sum()
    # print("Sum before leveling:")
    # print(f"{current_sum=}")

    # Difference is calculated
    starting_difference = goal_sum - current_sum
    # print(f"{starting_difference=}")

    # Calculates how much needs to be added or subtracted to each row of the stl_impute column IF there is a difference in sum value
    distribute_value = starting_difference / len(imputation_period_df)
    # print(f"{distribute_value=}")

    # The if conditions checks if the starting difference is positive, negative or zero and levels the stl_impute column accordingly
    if starting_difference == 0:
        print("There is no sum difference")

    if starting_difference > 0:
        print("Value need to be added to STL imputed series")
        imputation_period_df["stl_impute"] += distribute_value

    if starting_difference < 0:
        print("Value need to be subtracted from STL imputed series")
        imputation_period_df["stl_impute"] += distribute_value

    # Does another column sum calculation and comparison
    current_sum = imputation_period_df.stl_impute.sum()
    # print("Sum after leveling:")
    # print(f"{current_sum=}")

    # creates a negative mask for all values that are negative
    negative_mask = imputation_period_df["stl_impute"] < 0
    # counts have many values are negative
    sum_negatives = negative_mask.sum()

    # Checks if any values are negative and sets the bool to False if there are any
    if sum_negatives <= 0:
        print("no negatives")
        negative_values = False

    # if there are any negative values after the first leveling negative_values will be set to True which activates another leveling while loop
    if sum_negatives > 0:
        print(f"there are {sum_negatives} negative values")
        negative_values = True

        # df_len is used to know how many values the sum difference can be spread between
        df_len = len(imputation_period_df)

        # since there is a small risk of the difference that is being added each loop is very small, a max loop value is set
        loop_count = 1

        # the threshold value that is used to judge when the sum leveling is close enough AND there are no zero values
        min_threshold = 1

        # creates a new negative mask
        negative_mask = imputation_period_df["stl_impute"] < 0
        imputation_period_df.loc[negative_mask, "stl_impute"] = 0

    while negative_values:
        # since there is a small risk of the difference that is being added each loop is very small, a max loop value is set
        # Also there could be a risk of the leveling overshooting the threshold value (needs further testing)
        if loop_count > 10:
            print(f"Loop count has exceeded limit at {loop_count}")
            break

        stl_impute_sum = imputation_period_df["stl_impute"].sum()
        current_diff = goal_sum - stl_impute_sum
        print(f"{current_diff=} at loop {loop_count}")

        # calculated the value that should be subtracted from each value in the time period / dataFrame
        distribute_value = current_diff / df_len
        # convert it to a positive value because it made the code more readable
        distribute_value_plus = abs(distribute_value)
        # print(f"{distribute_value=}")
        # print(f"{distribute_value_plus=}")

        # create mask for values that can be subtracted the distributed value amount without becoming negative
        ltz_mask = (imputation_period_df["stl_impute"] - distribute_value_plus) <= 0
        print(imputation_period_df.head())
        # does the actual leveling
        imputation_period_df.loc[~ltz_mask, "stl_impute"] -= distribute_value_plus
        print("Leveling Leveling Leveling Leveling Leveling")
        # print(imputation_period_df.head())

        # calculates the sum difference again after the leveling so that it can be compared to the min_threshold value
        stl_impute_sum = imputation_period_df["stl_impute"].sum()
        current_diff = goal_sum - stl_impute_sum
        print(f"after loop: {loop_count}")
        print(f"{current_diff=}")
        current_diff_abs = abs(current_diff)

        # Checks if the sum difference after leveling is small enough to be below the min_threshold
        # If the difference is small enough - break out of the while loop
        if current_diff_abs < min_threshold:
            print("break loop")
            print(f"Current loop count: {loop_count}")
            print(current_diff_abs)
            print(min_threshold)
            break

        # iterate the loop count so that the loop does not become an infinite loop
        loop_count += 1

    if return_interval_sum:
        # print(f"{return_interval_sum=}")

        # ? new_interval creation START
        # ? new_interval creation START

        # Does the same procedure that was run fro stl_impute is now done on new_interval. Because negative values can easily be created when differencing the stl_impute series
        imputation_period_df["new_interval"] = imputation_period_df[
            "stl_impute"
        ] - imputation_period_df["stl_impute"].shift(1)
        current_year_first_value = df.loc[start_dt_obj, "original"]
        # print(f"{current_year_first_value=}")
        past_year_last_value = df.loc[start_dt_obj_minus_one_hour, "original"]
        # print(f"{past_year_last_value=}")
        first_replacement_value = current_year_first_value - past_year_last_value
        imputation_period_df["new_interval"].iloc[0] = first_replacement_value

        # create new goal_sum
        fifteen_minutes = timedelta(minutes=15)
        # Subtract the timedelta from the datetime object
        start_dt_obj_minus_15m = start_dt_obj - fifteen_minutes

        # A new dataFrame is created primerily so that we can get the series sum values
        stl_df_plus_15m = stl_df.loc[
            start_dt_obj_minus_15m:end_dt_obj,
            [
                "original",
                "stl_impute",
            ],
        ].copy()
        # creates a new iSum series by differencing the original series
        stl_df_plus_15m["plus15m_iSum"] = stl_df_plus_15m["original"] - stl_df_plus_15m[
            "original"
        ].shift(1)

        # drops the first value so that both new_interval and plus15m_iSum has the same length. Needed for calculating sum
        stl_df_plus_15m = stl_df_plus_15m.drop(stl_df_plus_15m.index[0])

        # Finally the sum values can be calculated
        iSum_sum = stl_df_plus_15m.plus15m_iSum.sum()
        new_interval_sum = imputation_period_df.new_interval.sum()

        # ? new_interval creation END
        # ? new_interval creation END

        # ? new_interval loop START
        # ? new_interval loop START

        # Extra check if there are any negative values in new_interval
        negative_count = (imputation_period_df["new_interval"] < 0).sum()
        print(f"Column 'new_interval' has {negative_count} negative value(s).")

        # new goal sum is the from the plus15m_iSum since it will be most similar to the original interval sum (df.column.shift()) from the linear KWh_total
        goal_sum = iSum_sum

        # Does another column sum calculation and comparison
        current_sum = imputation_period_df.new_interval.sum()
        # print("Sum after leveling:")
        # print(f"{current_sum=}")

        # creates a negative mask for all values that are negative
        negative_mask = imputation_period_df["new_interval"] < 0
        # counts have many values are negative
        sum_negatives = negative_mask.sum()

        # Checks if any values are negative and sets the bool to False if there are any
        if sum_negatives <= 0:
            print("no negatives")
            negative_values = False

        # if there are any negative values after the first leveling negative_values will be set to True which activates another leveling while loop
        if sum_negatives > 0:
            print(f"there are {sum_negatives} negative values")
            negative_values = True

            # df_len is used to know how many values the sum difference can be spread between
            df_len = len(imputation_period_df)

            # since there is a small risk of the difference that is being added each loop is very small, a max loop value is set
            loop_count = 1

            # the threshold value that is used to judge when the sum leveling is close enough AND there are no zero values.
            min_threshold = 1

            # creates a new negative mask
            negative_mask = imputation_period_df["new_interval"] < 0
            # changes the values that are ltz to 0
            imputation_period_df.loc[negative_mask, "new_interval"] = 0

        while negative_values:
            # since there is a small risk of the difference that is being added each loop is very small, a max loop value is set
            # Also there could be a risk of the leveling overshooting the threshold value (needs further testing)
            if loop_count > 10:
                print(f"Loop count has exceeded limit at {loop_count}")
                break

            new_interval_sum = imputation_period_df["new_interval"].sum()
            current_diff = goal_sum - new_interval_sum
            print(f"{current_diff=} at loop {loop_count}")

            # calculated the value that should be subtracted from each value in the time period / dataFrame
            distribute_value = current_diff / df_len
            # convert it to a positive value because it made the code more readable
            distribute_value_plus = abs(distribute_value)
            # print(f"{distribute_value=}")
            # print(f"{distribute_value_plus=}")

            # create mask for values that can be subtracted the distributed value amount without becoming negative
            ltz_mask = (
                imputation_period_df["new_interval"] - distribute_value_plus
            ) <= 0
            imputation_period_df.loc[~ltz_mask, "new_interval"] -= distribute_value_plus
            print("Leveling Leveling Leveling Leveling Leveling")

            # calculates the sum difference again after the leveling so that it can be compared to the min_threshold value
            new_interval_sum = imputation_period_df["new_interval"].sum()
            current_diff = goal_sum - new_interval_sum
            print(f"after loop: {loop_count}")
            print(f"{current_diff=}")
            current_diff_abs = abs(current_diff)

            # Checks if the sum difference after leveling is small enough to be below the min_threshold
            # If the difference is small enough - break out of the while loop
            if current_diff_abs < min_threshold:
                print("break loop")
                print(f"Current loop count: {loop_count}")
                print(f"{current_diff_abs}")
                print(f"{min_threshold}")
                break

            # iterate the loop count so that the loop does not become an infinite loop
            print(f"new_interval loop - loop count: {loop_count}")
            print(f"{loop_count=}")
            loop_count += 1

        # ? new_interval loop END
        # ? new_interval loop END

    # plots the predictions and saves the plot in the plots folder.
    if return_interval_sum:
        imupationPlot = imputation_period_df[["new_interval"]].plot()
        # print(imputation_period_df[["new_interval"]].head(30))
        # print(imputation_period_df[["new_interval"]].tail(30))
    # depending on query_param bool return the right plot
    if not return_interval_sum:
        imupationPlot = imputation_period_df[["linear", "stl_impute"]].plot()
        # print(imputation_period_df[["linear", "stl_impute"]].head(30))
        # print(imputation_period_df[["linear", "stl_impute"]].tail(30))
    imupationPlot.get_figure().savefig("plots/imputationPlot.png")
    imupationPlot.cla()
    # get the path of the saved plot image
    imupationPlot_path = Path("plots/imputationPlot.png")
    # print(imputation_period_df.columns)

    # the path for the newly saved plot is returned by the function
    return imupationPlot_path, imputation_period_df[["linear", "stl_impute"]]
