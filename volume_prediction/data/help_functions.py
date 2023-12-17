import pandas as pd
from scipy.stats import zscore
import numpy as np
import random

def read_csv_with_date(file_name, FILL_METHOD):
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

    # Fill missing values in the specified value column using backward fill
    if FILL_METHOD == 'zeros':
        df.fillna(0, inplace=True)
    elif FILL_METHOD == 'bfill':
        df.fillna(method='bfill', inplace=True)

    df.loc[df.values < 0] = 0
    

    return df, missing_dates

def handle_outliers(df, method='remove', z_score_threshold=2):
    """
    Handle outliers in a DataFrame.

    Parameters:
    - df: DataFrame
    - method: str, either 'remove' or 'replace'
    - z_score_threshold: float, threshold for z-score identification of outliers

    Returns:
    - DataFrame
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
    


def generate_random_dates(start_date_1, end_date_1, start_date_2, end_date_2, num_dates):
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



