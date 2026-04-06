import numpy as np
import pandas as pd

def build_feature_row(
    *,
    model,
    airline_code: int,
    destination_code: int,
    total_stops: int,
    date_day: int,
    date_month: int,
    dep_hour: int,
    dep_min: int,
    arr_hour: int,
    arr_min: int,
    duration_hours: int,
    duration_minutes: int,
    source: str,
) -> pd.DataFrame:
    
    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is None:
        feature_names = np.array(
            [
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
            ],
            dtype=object,
        )

    row = {col: 0 for col in feature_names}
    
    for k, v in {
        "Airline": airline_code,
        "Destination": destination_code,
        "Total_Stops": total_stops,
        "Date_of_Journey_day": date_day,
        "Date_of_Journey_month": date_month,
        "Dep_time_hour": dep_hour,
        "Dep_time_minutes": dep_min,
        "Arrival_Time_hour": arr_hour,
        "Arrival_Time_minutes": arr_min,
        "Duration_hours": duration_hours,
        "Duration_minutes": duration_minutes,
    }.items():
        if k in row:
            row[k] = v

    source_col = f"Source_{source}"
    if source_col in row:
        row[source_col] = 1
    else:
        pass

    return pd.DataFrame([row], columns=feature_names)
