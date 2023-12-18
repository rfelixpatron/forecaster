import pandas as pd
from scipy.stats import zscore
import json
import numpy as np
import random
from typing import Tuple, List

def load_config():
    with open('volume_prediction/config.json', 'r') as config_file:
        config = json.load(config_file)

    return config

def read_csv_with_date(file_name: str, fill_method: str) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
    """
    Read a CSV file with date information and handle missing values.

    Parameters:
    - file_name (str): The name of the CSV file.
    - fill_method (str): The method to fill missing values, either 'zeros' or 'bfill'.

    Returns:
    - Tuple[pd.DataFrame, List[pd.Timestamp]]: A tuple containing the DataFrame and a list of missing dates.
    """
    # Read CSV
    df = pd.read_csv(f'volume_prediction/data/{file_name}')

    # Find the first column containing the word "date"
    date_column = next((col for col in df.columns if 'date' in col.lower()), None)

    # If no date column is found, raise an exception or handle it as needed
    if date_column is None:
        raise ValueError("No valid date column found in the DataFrame.")

    # Convert the date column to datetime
    df[date_column] = pd.to_datetime(df[date_column])

    # Sort the DataFrame by the date column
    df.sort_values(by=date_column, inplace=True)

    # Create a full date range
    date_range = pd.date_range(start=df[date_column].min(), end=df[date_column].max())

    # Find missing dates
    missing_dates = date_range.difference(df[date_column])

    # Reindex the DataFrame with the full date range
    df = df.set_index(date_column).reindex(date_range)

    try:
        # Fill missing values in the specified value column using backward fill
        if fill_method == 'zeros':
            df.fillna(0, inplace=True)
        elif fill_method == 'bfill':
            df.fillna(method='bfill', inplace=True)
        else:
            raise ValueError(f"Invalid fill method: {fill_method}. Using default method 'zeros'.")
    except ValueError as e:
        print(e)
        df.fillna(0, inplace=True)  # Use default method 'zeros'

    df.loc[df.values < 0] = 0

    return df, missing_dates

def handle_outliers(df: pd.DataFrame, method: str = 'remove', z_score_threshold: float = 2.0) -> pd.DataFrame:
    """
    Handle outliers in a DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - method (str, optional): The method for handling outliers, either 'remove' or 'replace'. Defaults to 'remove'.
    - z_score_threshold (float, optional): The threshold for z-score identification of outliers. Defaults to 2.0.

    Returns:
    - pd.DataFrame: The DataFrame with outliers handled.
    """
    # Get the values of the DataFrame
    data = df.values.flatten()

    # Calculate z-scores
    z_scores = zscore(data)

    # Identify outliers based on the threshold
    outliers = np.where(np.abs(z_scores) > z_score_threshold)[0]

    if method == 'remove':
        # Create a new DataFrame without outliers
        df_no_outliers = df.drop(df.index[outliers])
        return df_no_outliers
    elif method == 'replace':
        # Calculate the average of all data points
        average_value = np.mean(data)

        # Replace outliers with the average value
        data[outliers] = average_value

        # Update the DataFrame with the modified column
        df.iloc[:, 0] = data
        return df
    else:
        raise ValueError("Invalid method. Choose 'remove' or 'replace'.")

def generate_random_dates(start_date_1: str, end_date_1: str, start_date_2: str, end_date_2: str, num_dates: int) -> List[pd.Timestamp]:
    """
    Generate random dates within specified date ranges.

    Parameters:
    - start_date_1 (str): Start date of the first range.
    - end_date_1 (str): End date of the first range.
    - start_date_2 (str): Start date of the second range.
    - end_date_2 (str): End date of the second range.
    - num_dates (int): Number of random dates to generate.

    Returns:
    - List[pd.Timestamp]: A list of randomly generated dates.
    """
    # Generate date ranges
    date_range_1 = pd.date_range(start=start_date_1, end=end_date_1, freq='D')
    date_range_2 = pd.date_range(start=start_date_2, end=end_date_2, freq='D')

    # Combine the date ranges
    evaluation_dates = date_range_1.union(date_range_2)

    # Set a seed for reproducibility
    random.seed(42)

    # Generate non-repeated random dates
    random_dates = random.sample(evaluation_dates.tolist(), num_dates)

    return random_dates

from typing import List

def generate_dates_before_date(date: str, num_dates: int) -> List[pd.Timestamp]:
    """
    Generate 30 dates including and before end_date_2.

    Parameters:
    - date (str): End date.
    - num_dates (int): Number of dates to generate.

    Returns:
    - List[pd.Timestamp]: A list of dates.
    """
    date = pd.to_datetime(date)
    start_date = date - pd.DateOffset(days=num_dates - 1)
    date_range = pd.date_range(start=start_date, end=date, freq='D')

    return date_range.tolist()

def create_results_dataframe(results_list):
    """
    Create a DataFrame from a list of lists containing dates, actual values, and forecast values.

    Parameters:
    - results_list (List[List]): List of lists containing [date, actual, forecast].

    Returns:
    - pd.DataFrame: DataFrame with 'Date' as the index, 'Actual' column for actual values, and 'Forecast' column for forecasted values.
    """
    dates = [entry[0] for entry in results_list]
    actual_values = [entry[1] for entry in results_list]
    forecast_values = [entry[2] for entry in results_list]

    df_results = pd.DataFrame({
        'Date': dates,
        'Actual': actual_values,
        'Forecast': forecast_values
    })

    # Set the 'Date' column as the index
    df_results.set_index('Date', inplace=True)

    return df_results
