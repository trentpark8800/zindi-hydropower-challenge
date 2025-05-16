"""
In this script my intention was to treat each of the series that needed to be forecast independently using models from the
darts package in python.

## Lessons Learned:
- This approach did not take into consideration the different lengths of the timeseries which lead to heavy biasing.
- There is not sufficient data per series to justify treating each independently.
- This code is scrappy and there are a lot of improvements that could be made to achieve better parallelization.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import duckdb as ddb
from darts import TimeSeries
from darts.models import AutoARIMA
from darts.metrics import rmse

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


con = ddb.connect("../kalam_hydropower.db", read_only=True)


def ensure_correct_date_range(df, min_date, max_date):
    full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")
    df = df.reindex(full_date_range)
    return df


def forecast_routine(df, min_date, max_date, forecast_horizon=31):
    extended_df = ensure_correct_date_range(df, min_date, max_date)

    cleaned_df = extended_df.interpolate(method="linear", limit_direction="both")

    ts = TimeSeries.from_dataframe(cleaned_df)

    train_ts, test_ts = ts.split_after(len(ts) - forecast_horizon)
    model = AutoARIMA()
    model.fit(train_ts)
    test_forecast = model.predict(forecast_horizon)
    rmse_score = rmse(test_ts, test_forecast)

    full_model = AutoARIMA()
    full_model.fit(ts)
    full_forecast = full_model.predict(forecast_horizon)

    full_forecast_df = full_forecast.to_dataframe()

    return full_forecast_df, rmse_score


def forecast_for_source(source, selected_df, min_date, max_date, forecast_horizon):
    try:
        selected_source_df = selected_df[selected_df["source"] == source].copy()
        selected_source_df.set_index("date", inplace=True)

        forecast_df, rmse_score = forecast_routine(
            selected_source_df[["kwh"]],
            min_date=min_date,
            max_date=max_date,
            forecast_horizon=forecast_horizon
        )

        forecast_df["source"] = source
        return forecast_df, source, rmse_score
    except Exception as e:
        print(f"Error processing source {source}: {e}")
        return None, source, None


def main():

    ss_df = con.sql("select * from raw.sample_submission").to_df()
    ss_df[["date", "source"]] = ss_df["ID"].str.split("_", n=1, expand=True)

    df = con.sql("""
        select
            *
        from prepared.daily_hydropower_production
        where source in (select source from ss_df)
    """).to_df()

    con.close()

    selected_df = df[["date", "source", "kwh"]]

    min_date = pd.Timestamp("2023-06-24")
    max_date = pd.Timestamp("2024-09-23")

    forecast_horizon = 31
    sources = ss_df["source"].unique()

    forecasts_df = pd.DataFrame()
    results = {
        "source": [],
        "rmse_score": []
    }

    print(f"Running parallel forecasts for {len(sources)} sources...\n")

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                forecast_for_source,
                source,
                selected_df,
                min_date,
                max_date,
                forecast_horizon
            ): source for source in sources
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            forecast_df, source, rmse_score = future.result()

            if forecast_df is not None:
                forecasts_df = pd.concat([forecasts_df, forecast_df])
                results["source"].append(source)
                results["rmse_score"].append(rmse_score)

    forecasts_df["ID"] = forecasts_df.index.astype(str) + "_" + forecasts_df["source"]

    id_difference = set(forecasts_df["ID"]) - set(ss_df["ID"])
    if len(id_difference) > 0:
        print("\nSubmission is incomplete!")

    forecasts_df.sort_values(by=["source", "date"], inplace=True)

    # Save results
    forecasts_df[["ID", "kwh"]].to_csv("./submissions/arima_forecast.csv", index=False)


if __name__ == "__main__":
    main()
