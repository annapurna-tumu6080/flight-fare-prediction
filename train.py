import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.config import AIRLINE_TO_CODE, DEST_TO_CODE, TOTAL_STOPS_MAP


def extract_duration(duration_str):
    if pd.isna(duration_str):
        return 0, 0
    hours = 0
    mins = 0
    parts = duration_str.split()
    for p in parts:
        if 'h' in p:
            hours = int(p.replace('h', ''))
        elif 'm' in p:
            mins = int(p.replace('m', ''))
    return hours, mins


def main():
    print("1. Loading 'flightdata.csv'...")
    try:
        df = pd.read_csv("flightdata.csv")
    except FileNotFoundError:
        print("Error: flightdata.csv not found in the current directory.")
        return

    df.dropna(inplace=True)

    print("2. Engineering date and time features...")
    # Date of Journey
    journey_dates = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["Date_of_Journey_day"] = journey_dates.dt.day
    df["Date_of_Journey_month"] = journey_dates.dt.month

    # Departure Time
    df["Dep_time_hour"] = df["Dep_Time"].str.split(':').str[0].astype(int)
    df["Dep_time_minutes"] = df["Dep_Time"].str.split(':').str[1].str.split().str[0].astype(int)

    # Arrival Time
    df["Arrival_Time_hour"] = df["Arrival_Time"].str.split(':').str[0].astype(int)
    df["Arrival_Time_minutes"] = df["Arrival_Time"].str.split(':').str[1].str.split().str[0].astype(int)

    # Duration
    dur_hours = []
    dur_mins = []
    for d in df["Duration"]:
        h, m = extract_duration(d)
        dur_hours.append(h)
        dur_mins.append(m)
    df["Duration_hours"] = dur_hours
    df["Duration_minutes"] = dur_mins

    # Total Stops
    df["Total_Stops"] = df["Total_Stops"].map(TOTAL_STOPS_MAP)
    df["Total_Stops"] = df["Total_Stops"].fillna(0).astype(int)

    print("3. Encoding categorical features...")
    # Airline and Destination Label Encoding (so they exactly match what Streamlit expects)
    df["Airline"] = df["Airline"].map(AIRLINE_TO_CODE).fillna(0).astype(int)
    df["Destination"] = df["Destination"].map(DEST_TO_CODE).fillna(0).astype(int)

    # One-hot encode Source
    source_dummies = pd.get_dummies(df["Source"], prefix="Source", dtype=int)
    df = pd.concat([df, source_dummies], axis=1)

    # Required columns in the same specific order as `build_feature_row` in predict logic
    required_cols = [
        "Airline",
        "Destination",
        "Total_Stops",
        "Date_of_Journey_day",
        "Date_of_Journey_month",
        "Dep_time_hour",
        "Dep_time_minutes",
        "Arrival_Time_hour",
        "Arrival_Time_minutes",
        "Duration_hours",
        "Duration_minutes",
        "Source_Banglore",
        "Source_Chennai",
        "Source_Delhi",
        "Source_Kolkata",
        "Source_Mumbai",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0

    X = df[required_cols]
    y = df["Price"]

    print("4. Training Random Forest model (this may take a little while)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X.values, y.values)  # `.values` prevents feature name mismatch warnings later on

    # Actually assign `feature_names_in_` so the streamlit script is happy
    model.feature_names_in_ = X.columns.to_numpy()

    print("5. Saving model to models/rd_random.pkl...")
    os.makedirs("models", exist_ok=True)
    with open("models/rd_random.pkl", "wb") as f:
        pickle.dump(model, f)

    print("--- SUCCESS ---")
    print("Model created successfully! You can now start the Streamlit app.")

if __name__ == "__main__":
    main()
