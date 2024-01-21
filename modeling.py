from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import tree
import pickle
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# set preferred model
RANDOM_FOREST = "randomforest"
DECISION_TREE = "decisiontree"
MODEL = DECISION_TREE
# define whether to load model or to train a new one
LOAD_MODEL = False



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

    #@todo validation set

    if MODEL == RANDOM_FOREST:
        print("RANDOM FOREST")
        if LOAD_MODEL:
            # load the model from disk
            with open("model_randomforest.pkl", "rb") as f:
                model = pickle.load(f)
        else:
            # train model Random Forest
            model = RandomForestRegressor(verbose=2, oob_score=True)
            model.fit(X_train, y_train)

            # save model
            with open("model_randomforest.pkl", "wb") as f:
                pickle.dump(model, f)
    # otherwise decision tree
    else:
        print("DECISION TREE")
        if LOAD_MODEL:
            # load the model from disk
            with open("model_decisiontree.pkl", "rb") as f:
                model = pickle.load(f)
        else:
            # train model Decision Tree
            # @todo max_depth
            model = tree.DecisionTreeRegressor()
            model.fit(X_train, y_train)

            # save model decision tree
            with open("model_decisiontree.pkl", "wb") as f:
                pickle.dump(model, f)

    # return model
    return model, X_train, y_train, X_test, y_test


def evaluate_model(model, X_train, y_train, X_test, y_test):
    # evaluate model
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean absolute error train: {mae:.2f}")
    print(f"Mean squared error test: {mse:.2f}")
    print(f"R2 score test: {r2:.2f}")

    mset = mean_squared_error(y_train, y_train_pred)
    r2t = r2_score(y_train, y_train_pred)
    maet = mean_absolute_error(y_train, y_train_pred)
    print(f"Mean squared error train: {mset:.2f}")
    print(f"Mean absolute error train: {maet:.2f}")
    print(f"R2 score train: {r2t:.2f}")



if __name__ == "__main__":
    df = import_cleaned_data()

    #df.iloc[:200].to_csv("cleaned_data_all.csv")
    #exit(0)

    print(f"Dataframe shape: {df.shape}")
    model, X_train, y_train, X_test, y_test = model_data(df)

    # evaluate model
    evaluate_model(model, X_train, y_train, X_test, y_test)
