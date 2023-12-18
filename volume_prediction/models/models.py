import numpy as np
import pandas as pd
from typing import Tuple
from help_functions.help_functions import load_config
from models.prophet_helpers import generate_holiday_df, generate_training_data_prophet
from pmdarima import auto_arima
from prophet import Prophet


def create_training_data(df: pd.DataFrame, prediction_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a training DataFrame and a target date DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - prediction_date (str): The target prediction date.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training DataFrame and the target date DataFrame.
    """
    # Create a training dataframe (3 weeks after the target date)
    training_end_date = pd.to_datetime(prediction_date) + pd.DateOffset(days=-1)
    training_start_date = pd.to_datetime(prediction_date) - pd.Timedelta(weeks=3)
    training_df = df.loc[training_start_date:training_end_date]

    # Create a target date DataFrame
    target_date = pd.DataFrame(index=[pd.to_datetime(prediction_date)])

    return training_df, target_date


def is_covid_season(ds: str) -> bool:
    """
    Check if the given date is within the COVID season.

    Parameters:
    - ds (str): The date string.

    Returns:
    - bool: True if the date is within the COVID season, False otherwise.
    """

    # Load configuration from file
    model_config = load_config()
    config = model_config['model_config']['prophet_config']

    date = pd.to_datetime(ds)
    return pd.to_datetime(config['covid_start_date']) <= date <= pd.to_datetime(config['covid_start_date'])


def baseline_model_predict(df: pd.DataFrame, prediction_date: str) -> pd.DataFrame:
    """
    Baseline model prediction.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - prediction_date (str): The target prediction date.

    Returns:
    - pd.DataFrame: The predicted DataFrame.
    """
    training_df, target_date = create_training_data(df, prediction_date)

    output_df = pd.DataFrame(columns=["forecast"])
    list_days = [i for i in range(0, 7, 1)]
    for day in list_days:
        day_df = training_df.copy().loc[training_df.index.dayofweek == day]
        output_df.loc[day, "forecast"] = np.rint(np.median(day_df.tail(21)))

    output_df["forecast"] = output_df["forecast"].astype(int, copy=False)

    target_date["day_of_week"] = target_date.index.dayofweek
    output_df = pd.merge(
        target_date,
        output_df.reset_index().rename(columns={"index": "day_of_week"}),
        on="day_of_week",
    )
    output_df = output_df.set_index(target_date.index)

    return output_df[['forecast']]


def rolling_average_model_predict(df: pd.DataFrame, prediction_date: str) -> pd.DataFrame:
    """
    Rolling average model prediction.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - prediction_date (str): The target prediction date.

    Returns:
    - pd.DataFrame: The predicted DataFrame.
    """
    training_df, target_date = create_training_data(df, prediction_date)

    rolling_avg = training_df.rolling(window=7, min_periods=5).mean()
    target_date['forecast'] = rolling_avg.iloc[-1].values

    return target_date


def prophet_model_predict(df: pd.DataFrame, prediction_date: str) -> pd.DataFrame:
    """
    Prophet model prediction.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - prediction_date (str): The target prediction date.

    Returns:
    - pd.DataFrame: The predicted DataFrame.
    """
    holidays = generate_holiday_df()
    temp_df = generate_training_data_prophet(df, prediction_date)

    prophet_df = temp_df.reset_index().rename(columns={'index': 'ds', 'VOLUME': 'y'})
    model = Prophet(holidays=holidays)
    model.fit(prophet_df)
    future_df = model.make_future_dataframe(periods=1).iloc[[-1]]
    output_df = model.predict(future_df)
    output_df.set_index('ds', inplace=True)

    return output_df[['yhat']]


def prophet_model_covid_predict(df: pd.DataFrame, prediction_date: str) -> pd.DataFrame:
    """
    Prophet model with COVID correction prediction.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - prediction_date (str): The target prediction date.

    Returns:
    - pd.DataFrame: The predicted DataFrame.
    """
    holidays = generate_holiday_df()
    temp_df = generate_training_data_prophet(df, prediction_date)

    prophet_df = temp_df.reset_index().rename(columns={'index': 'ds', 'VOLUME': 'y'})
    prophet_df["corona_effect"] = prophet_df['ds'].apply(is_covid_season)
    prophet_df["no_corona_effect"] = ~prophet_df['ds'].apply(is_covid_season)

    model = Prophet(holidays=holidays, yearly_seasonality=False)
    model.add_seasonality(name='corona_correction', period=365.25, fourier_order=4, condition_name='corona_effect')
    model.add_seasonality(name='no_corona_correction', period=365.25, fourier_order=4, condition_name='no_corona_effect')
    model.fit(prophet_df)
    future_df = model.make_future_dataframe(periods=1).iloc[[-1]]
    future_df['corona_effect'] = future_df['ds'].apply(is_covid_season)
    future_df['no_corona_effect'] = ~future_df['ds'].apply(is_covid_season)
    output_df = model.predict(future_df)

    output_df.set_index('ds', inplace=True)

    return output_df[['yhat']]


def auto_arima_predict(df: pd.DataFrame, prediction_date: str) -> pd.DataFrame:
    """
    AutoARIMA model prediction.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - prediction_date (str): The target prediction date.

    Returns:
    - pd.DataFrame: The predicted DataFrame.
    """
    temp_df = generate_training_data_prophet(df, prediction_date)

    model = auto_arima(temp_df, suppress_warnings=True, seasonal=False, stepwise=True)
    forecast_steps = 1
    forecast, _ = model.predict(n_periods=forecast_steps, return_conf_int=True)

    target_date = pd.DataFrame(index=[pd.to_datetime(prediction_date)])
    target_date['forecast'] = forecast.values

    return target_date
