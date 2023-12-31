# This file contains the data cleaning pipeline for the project.
# It takes in the raw data and outputs the cleaned data.
# It combines all the separate data cleaning steps into one pipeline.
import os
from glob import glob
import re
from pathlib import Path
import time
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# Import libraries
import pandas as pd
import pyarrow.parquet as pq

import logging

# Define constants
# Path to raw data
DATA_PATH = Path(".", "data")
FILE_PATTERN = 'yellow_tripdata_2022-*.parquet'
OUTPUT_PATH = Path(".", "data", "cleaned_data")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

REQUIRED_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance", "PULocationID", "DOLocationID"]

DATA_TYPE_MAP = {
    "tpep_pickup_datetime": "datetime64[ns]",
    "tpep_dropoff_datetime": "datetime64[ns]",
    "trip_distance": "float32",
    "PULocationID": "int32",
    "DOLocationID": "int32"
}

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def timeit_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logging.info(f"Function {func.__name__} took {time.time() - start} seconds")
        return result
    return wrapper

def _preset_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype(DATA_TYPE_MAP)
    return df

def _filter_pickup_dropoff(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    df = df[df["tpep_pickup_datetime"].dt.year == int(year)]
    df = df[df["tpep_pickup_datetime"].dt.month == int(month)]
    # For dropoff time, we need to include the next year as well, but only january of next year
    df = df[(df["tpep_dropoff_datetime"].dt.year == int(year)) | ((df["tpep_dropoff_datetime"].dt.year + 1 == int(year)) & ((df["tpep_dropoff_datetime"].dt.month == 1)))]
    # For the dropoff time, we need to include the next month as well
    df = df[(df["tpep_dropoff_datetime"].dt.month == int(month)) | (df["tpep_dropoff_datetime"].dt.month + 1 == int(month))]
    return df


def _filter_trip_distance(df: pd.DataFrame, min_distance: int = 0, max_distance: int = 70) -> pd.DataFrame:
    df = df[(df["trip_distance"] > min_distance) & (df["trip_distance"] < max_distance)]
    return df


def _filter_valid_locations(df: pd.DataFrame, max_location_id: int = 263) -> pd.DataFrame:
    # Ensure that DO and PU locations are within the range of 1-263
    df = df[(df["PULocationID"] >= 1) & (df["PULocationID"] <= max_location_id)]
    df = df[(df["DOLocationID"] >= 1) & (df["DOLocationID"] <= max_location_id)]
    return df


def _filter_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    return df


def _feature_holidays(df: pd.DataFrame) -> pd.DataFrame:
    american_holidays = calendar().holidays()

    df["pickup_is_holiday"] = df["tpep_pickup_datetime"].dt.date.isin(american_holidays)
    df["dropoff_is_holiday"] = df["tpep_dropoff_datetime"].dt.date.isin(american_holidays)
    return df


def _feature_timecols(df: pd.DataFrame, prefix: str, column_name: str) -> pd.DataFrame:
    df[f"{prefix}_year"] = df[column_name].dt.year
    df[f"{prefix}_month"] = df[column_name].dt.month
    df[f"{prefix}_day"] = df[column_name].dt.day
    df[f"{prefix}_hour"] = df[column_name].dt.hour
    df[f"{prefix}_day_of_week"] = df[column_name].dt.dayofweek
    df[f"{prefix}_week"] = df[column_name].dt.isocalendar().week
    df[f"{prefix}_is_weekend"] = df[column_name].dt.weekday.isin([5, 6])
    df[f"{prefix}_day_of_year"] = df[column_name].dt.dayofyear
    return df


def _transform_distance_km(df: pd.DataFrame) -> pd.DataFrame:
    df["trip_distance"] = df["trip_distance"] * 1.609344
    return df


def _feature_duration(df: pd.DataFrame) -> pd.DataFrame:
    df["trip_duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.seconds

    # Filter out trips that are too short or too long
    # filter out trips with negative duration
    df = df.drop(df[df['trip_duration'] <= 0].index, axis=0)
    # @todo 2 minutes is a bit arbitrary, explain why we chose this number
    df = df.drop(df[df['trip_duration'] < 120].index, axis=0)
    # @todo explain why we chose this number
    df = df.drop(df[df['trip_duration'] >= df['trip_duration'].quantile(0.998)].index, axis=0)
    return df


def _feature_velocity(df: pd.DataFrame) -> pd.DataFrame:
    df["velocity"] = df["trip_distance"] / (
                (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.seconds / 3600)
    return df

@timeit_decorator
def _preprocessing_pipeline(df: pd.DataFrame, year, month) -> pd.DataFrame:
    df = _preset_types(df)

    df = _filter_pickup_dropoff(df, year, month)
    df = _filter_trip_distance(df)
    df = _filter_valid_locations(df)
    df = _filter_duplicates(df)

    df = _transform_distance_km(df)

    df = _feature_holidays(df)
    df = _feature_timecols(df, "pickup", "tpep_pickup_datetime")
    df = _feature_timecols(df, "dropoff", "tpep_dropoff_datetime")
    df = _feature_velocity(df)
    df = _feature_duration(df)
    return df




def transform_data():
    parquet_files = glob(str(DATA_PATH / FILE_PATTERN))

    loaded_data = [pq.read_table(file, columns=REQUIRED_COLUMNS) for file in parquet_files]

    cleaned_files = []
    # get total rows
    total_rows_before = 0
    total_rows_after = 0

    # Run the preprocessing pipeline for each month seperatly
    for i, df in enumerate(loaded_data):
        year, month = re.findall(r'\d+', parquet_files[i])[-2:]
        year = int(year)
        month = int(month)

        df = df.to_pandas()
        logging.debug(f"Processing {year}-{month}, Length before: {len(df)}")
        total_rows_before += len(df)
        df = _preprocessing_pipeline(df, year=year, month=month)
        total_rows_after += len(df)
        logging.debug(f"Processing {year}-{month}, Length after: {len(df)}")
        df = df.astype(DATA_TYPE_MAP)

        file_name = f"yellow_tripdata_{year}-{month}.parquet"

        df.to_parquet(OUTPUT_PATH / file_name, index=False)
        cleaned_files.append(df)

    logging.info(f"Total rows before: {total_rows_before}")
    logging.info(f"Total rows after: {total_rows_after}")
    # print difference in percentage
    logging.info(f"Percentage of rows kept: {round(total_rows_after / total_rows_before * 100, 2)}%")

    # Combine all the dataframes into one
    df = pd.concat(cleaned_files)
    df.to_parquet(OUTPUT_PATH / f"cleaned_data_all.parquet", index=False)






def preprocessing_pipeline():
    pass


if __name__ == '__main__':
    transform_data()