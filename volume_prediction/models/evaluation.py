import numpy as np

def calculate_mape(evaluation_values, forecast_values):
    """
    Calculate Mean Absolute Percentage Error (MAPE) between evaluation and forecast values.

    Parameters:
    - evaluation_values (numpy.ndarray or list): Actual or observed values.
    - forecast_values (numpy.ndarray or list): Predicted or forecasted values.

    Returns:
    float: Mean Absolute Percentage Error (MAPE) as a percentage.
    """
    absolute_percentage_errors = np.abs((evaluation_values - forecast_values) / evaluation_values)
    mean_absolute_percentage_error = 100 * np.mean(absolute_percentage_errors)

    return mean_absolute_percentage_error
