import pandas as pd

def generate_holiday_df(custom_holidays=None):
    """
    Generate a DataFrame containing holidays and weekends.

    Parameters:
    - start_date: str, start date of the desired range in 'YYYY-MM-DD' format
    - end_date: str, end date of the desired range in 'YYYY-MM-DD' format
    - custom_holidays: list of str, optional, custom holidays in 'YYYY-MM-DD' format

    Returns:
    - pd.DataFrame, DataFrame containing holidays and weekends
    """
    custom_holidays = ['2016-12-25', '2017-12-25', '2018-12-25', '2019-12-25', '2020-12-25', '2021-12-25', '2022-12-25',
                            '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01',
                            '2016-12-26', '2023-12-25', '2024-12-25', '2023-01-01', '2024-01-01']
    
    start_date = '2016-01-01'
    end_date = '2024-12-31'
    date_range = pd.date_range(start=start_date, end=end_date)
    weekends = date_range[date_range.day_name().isin(['Saturday', 'Sunday'])]

    weekend_df = pd.DataFrame({
        'holiday': 'weekend',
        'ds': weekends,
        'lower_window': 0,
        'upper_window': 0,
        'prior_scale': 10,  
    })

    if custom_holidays is not None:
        custom_holidays_df = pd.DataFrame({
            'holiday': 'custom_holidays',
            'ds': pd.to_datetime(custom_holidays),
            'lower_window': 0,
            'upper_window': 0,
            'prior_scale': 10, 
        })

        all_holidays_df = pd.concat([weekend_df, custom_holidays_df])
    else:
        all_holidays_df = weekend_df

    return all_holidays_df



def generate_training_data_prophet(df, prediction_date, max_training_horizon_years=5):
    """
    Generate training data based on the given prediction date and maximum training horizon.

    Parameters:
    - df: pd.DataFrame, the original DataFrame containing the time series data
    - prediction_date: pd.Timestamp, the date for which predictions are made
    - max_training_horizon_years: int, the maximum training horizon in years

    Returns:
    - pd.DataFrame, a copy of the DataFrame containing training data
    """
    max_training_horizon = max_training_horizon_years * 365  # Convert years to days
    training_date_start = prediction_date - pd.Timedelta(days=max_training_horizon)
    training_date_end = prediction_date - pd.Timedelta(days=1)
    training_data = df.loc[training_date_start:training_date_end].copy()
    return training_data

