from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle


DATA_PATH = Path(".", "data", "cleaned_data")
FILE_PATH = DATA_PATH / "cleaned_data_all.parquet"


def import_cleaned_data():
    # Import cleaned data
    data = pq.read_table(FILE_PATH)

    # Convert to pandas dataframe
    return data.to_pandas()


def model_data(df: pd.DataFrame):
    df = df.sample(frac=0.1, random_state=42)
    print("AFTER SHUFFLE")
    # split data into train and test set using sklearn
    y = df["trip_duration"]
    X = df.drop(["trip_duration", "tpep_pickup_datetime", "tpep_dropoff_datetime"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("AFTER SPLIT")

    # train model
    model = RandomForestRegressor(verbose=1)
    model.fit(X_train, y_train)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    # evaluate model
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean squared error test: {mse:.2f}")
    print(f"R2 score test: {r2:.2f}")

    mset = mean_squared_error(y_train, y_train_pred)
    r2t = r2_score(y_train, y_train_pred)
    print(f"Mean squared error train: {mset:.2f}")
    print(f"R2 score train: {r2t:.2f}")



if __name__ == "__main__":
    df = import_cleaned_data()

    #df.iloc[:200].to_csv("cleaned_data_all.csv")
    #exit(0)

    print(f"Dataframe shape: {df.shape}")
    model_data(df)
