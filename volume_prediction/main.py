# Standard library imports
import logging
import sys
from datetime import datetime
import json
from typing import List, Callable

# Third-party library imports
import numpy as np
import pandas as pd

# Local module imports
from data.help_functions import (
    read_csv_with_date,
    handle_outliers,
    generate_random_dates,
)
from models.models import (
    baseline_model_predict,
    rolling_average_model_predict,
    prophet_model_predict,
    prophet_model_covid_predict,
    auto_arima_predict,
)
from models.tft_model import tft_model
from models.evaluation import MAPE

# Setting up logging
_logger = logging.getLogger("training_logger")
_logger.setLevel(logging.DEBUG)


def load_config():
    with open('volume_prediction/config.json', 'r') as config_file:
        config = json.load(config_file)

    return config


def calculate_mape_for_dates(df: pd.DataFrame, 
                              random_dates_for_evaluation: List[datetime], 
                              model_function: Callable[[pd.DataFrame, datetime], pd.Series]) -> List[float]:
    """
    Calculate MAPE for a given model on specified random dates.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing 'VOLUME' (target prediction).
    - random_dates_for_evaluation (List[datetime]): List of dates for evaluation.
    - model_function (Callable[[pd.DataFrame, datetime], pd.Series]): The function for making model predictions.

    Returns:
    - List[float]: List of MAPE values.
    """
    mape_list = []

    for date in random_dates_for_evaluation:
        # Check for holidays and weekends
        if (
            datetime(date.year, 12, 25) == date or
            datetime(date.year + 1, 1, 1) == date or
            (datetime(date.year, 12, 26) == date and date.year == 2016) or
            date.weekday() in [5, 6]  # Saturday and Sunday
        ):
            continue

        # Model prediction
        forecast = model_function(df[['VOLUME']], date)

        # Assuming that the forecast and test are one day forecasts, you may need to adjust this based on your use case
        test = df.loc[forecast.index][['VOLUME']]

        # Calculate MAPE and append to the list
        mape_list.append(MAPE(test.iloc[0].iloc[0], forecast.iloc[0].iloc[0]))

    return mape_list


def main():
    """
    Main execution function.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    _logger.debug("Starting main")

    # Load configuration from file
    config = load_config()

    # LOAD DATA
    df, _ = read_csv_with_date(config['main_input_csv'], config['filling_method'])
    df = handle_outliers(df, method=config['outlier_method'])

    cross_validation_windows = config['cross_validation_windows']

    random_dates_for_evaluation = generate_random_dates(
        config['start_date_1'],
        config['end_date_1'],
        config['start_date_2'],
        config['end_date_2'],
        cross_validation_windows
    )

    models = {
        'Baseline': baseline_model_predict,
        'Rolling Average': rolling_average_model_predict,
        'Prophet': prophet_model_predict,
        'Prophet-COVID': prophet_model_covid_predict,
        'Auto ARIMA': auto_arima_predict,
        'TFT': tft_model,
    }

    results = {}

    mape_data = pd.DataFrame()

    for model_name, model_function in models.items():
        mape_results = calculate_mape_for_dates(df, random_dates_for_evaluation, model_function)
        avg_mape = np.mean(mape_results)
        results[model_name] = avg_mape

        # Save mape_results to the DataFrame
        mape_data[model_name] = mape_results

    # Display results in descending order
    results_sorted = sorted(results.items(), key=lambda x: x[1], reverse=False)
    for model_name, avg_mape in results_sorted:
        print(f'{model_name}: {avg_mape:.1f}%')

    # Save mape_data to a CSV file
    mape_data.to_csv(f'volume_prediction/data/{config["output_csv"]}', index=False)

    # Display the best model
    best_model = results_sorted[0][0]
    print(f'\nThe best model is: {best_model} (MAPE: {results[best_model]:.1f}%)')


if __name__ == "__main__":
    main()


