# Standard library imports
import logging
import sys
from datetime import datetime
from typing import List, Callable, Tuple, Dict

# Third-party library imports
import numpy as np
import pandas as pd

# Local module imports
from help_functions.help_functions import (
    create_results_dataframe,
    generate_dates_before_date,
    generate_random_dates,
    handle_outliers,
    load_config,
    read_csv_with_date,
)
from models.models import (
    auto_arima_predict,
    baseline_model_predict,
    prophet_model_covid_predict,
    prophet_model_predict,
    rolling_average_model_predict,
)
from models.evaluation import calculate_mape
from models.tft_model import tft_model


# Setting up logging
_logger = logging.getLogger("training_logger")
_logger.setLevel(logging.DEBUG)


def calculate_mape_for_dates(df: pd.DataFrame, 
                              random_dates_for_evaluation: List[datetime], 
                              model_function: Callable[[pd.DataFrame, datetime], pd.Series]) -> Tuple[List[float], List[Tuple[datetime, float, float]]]:
    """
    Calculate MAPE for a given model on specified random dates.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing 'VOLUME' (target prediction).
    - random_dates_for_evaluation (List[datetime]): List of dates for evaluation.
    - model_function (Callable[[pd.DataFrame, datetime], pd.Series]): The function for making model predictions.

    Returns:
    - Tuple[List[float], List[Tuple[datetime, float, float]]]: List of MAPE values and a list of tuples containing test index, test value, and forecast value.
    """
    mape_list = []
    values_list = []

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
        mape_list.append(calculate_mape(test.iloc[0].iloc[0], forecast.iloc[0].iloc[0]))
        values_list.append((test.index[0], test.iloc[0].iloc[0], forecast.iloc[0].iloc[0]))

    return mape_list, values_list


def evaluate_models(models: Dict[str, Callable], 
                    df: pd.DataFrame, 
                    random_dates_for_evaluation: List[datetime], 
                    config: Dict) -> Tuple[Dict[str, float], pd.DataFrame, Callable]:
    """
    Evaluate different models using MAPE for a given DataFrame and a list of random dates.

    Parameters:
    - models (Dict[str, Callable]): Dictionary containing model names as keys and model functions as values.
    - df (pd.DataFrame): DataFrame containing the data.
    - random_dates_for_evaluation (List[datetime]): List of dates for evaluation.
    - config (Dict): Configuration dictionary.

    Returns:
    - Tuple[Dict[str, float], pd.DataFrame, Callable]: 
        - Results dictionary containing model names as keys and average MAPE values as values.
        - DataFrame containing MAPE results for each model.
        - Best model function.
    """
    results = {}
    mape_data = pd.DataFrame()

    for model_name, model_function in models.items():
        mape_results, _ = calculate_mape_for_dates(df, random_dates_for_evaluation, model_function)
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

    return results_sorted


def main():
    """
    Main execution function.
    """
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    _logger.debug("Starting main")

    # Load configuration from file
    main_config = load_config()
    config = main_config['main_config']

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

    # DEFINE MODELS TO BE TESTED

    models = {
        'Baseline': baseline_model_predict,
        'Rolling Average': rolling_average_model_predict,
        'Prophet': prophet_model_predict,
        'Prophet-COVID': prophet_model_covid_predict,
        'Auto ARIMA': auto_arima_predict,
        'TFT': tft_model,
    }

    # EVALUATE THE POTENTIAL MODELS AND SELECT THE BEST MODEL

    results = evaluate_models(models, df, random_dates_for_evaluation, config)

    best_model = results[0][0]
    best_results = results[0][1]
    best_model_function = models[best_model]

    print(f'\nThe best model using {cross_validation_windows} cross validation windows is:  '
      f'{best_model} (MAPE: {best_results:.1f}%)')

    # GENERATE FINAL RESULTS WITH THE BEST MODEL

    days_before = config['days_before']
    final_prediction_dates = generate_dates_before_date(config['end_date_2'], days_before)
    
    _, results_list = calculate_mape_for_dates(df, final_prediction_dates, best_model_function)

    results_df = create_results_dataframe(results_list)
    # Save mape_data to a CSV file
    results_df.to_csv(f'volume_prediction/data/{config["final_output_csv"]}', index=True)


if __name__ == "__main__":
    main()


