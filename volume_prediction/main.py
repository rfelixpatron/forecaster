import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import logging, sys
from data.help_functions import read_csv_with_date, handle_outliers, generate_random_dates
from models.models import baseline_model_predict, rolling_average_model_predict, prophet_model_predict, prophet_model_covid_predict, auto_arima_predict
from prophet import Prophet

from models.evaluation import MAPE

_logger = logging.getLogger("training_logger")
_logger.setLevel(logging.DEBUG) 

def calculate_mape_for_dates(df, random_dates_for_evaluation, model_function):
    mape_list = []

    for date in random_dates_for_evaluation:
        # Check for holidays and weekends
        if date.day == 25 and date.month == 12 or \
                date.day == 1 and date.month == 1 or \
                date.day == 26 and date.month == 12 and date.year == 2016 or \
                date.weekday() in [5, 6]:  # Saturday and Sunday
            continue

        # Model prediction
        forecast = model_function(df[['VOLUME']], date)

        # Assuming that the forecast and test are one day forecasts, you may need to adjust this based on your use case
        test = df.loc[forecast.index][['VOLUME']]

        # Calculate MAPE and append to the list
        mape_list.append(MAPE(test.iloc[0].iloc[0], forecast.iloc[0].iloc[0]))


    return mape_list




def main():
    _logger.debug("Starting main")

    # LOAD DATA
    df, _ = read_csv_with_date('VOLUMES.csv', 'zeros')
    df = handle_outliers(df, method = 'replace')

    cross_validation_windows = 20

    random_dates_for_evaluation = generate_random_dates('2016-05-01', '2019-11-30', '2020-07-01', '2021-02-16', cross_validation_windows)

    mape_results_baseline = calculate_mape_for_dates(df, random_dates_for_evaluation, model_function=baseline_model_predict)
    mape_results_rolling = calculate_mape_for_dates(df, random_dates_for_evaluation, model_function=rolling_average_model_predict)
    mape_results_prophet = calculate_mape_for_dates(df, random_dates_for_evaluation, model_function=prophet_model_predict)
    mape_results_prophet_covid = calculate_mape_for_dates(df, random_dates_for_evaluation, model_function=prophet_model_covid_predict)
    mape_results_auto_arima = calculate_mape_for_dates(df, random_dates_for_evaluation, model_function=auto_arima_predict)
    

    print(f'The Baseline model MAPE results for {cross_validation_windows} cross validation windows is: {np.mean(mape_results_baseline):.1f}%')
    print(f'The Rolling average MAPE results for {cross_validation_windows} cross validation windows is: {np.mean(mape_results_rolling):.1f}%')
    print(f'The Prophet MAPE results for {cross_validation_windows} cross validation windows is: {np.mean(mape_results_prophet):.1f}%')
    print(f'The Prophe-COVID MAPE results for {cross_validation_windows} cross validation windows is: {np.mean(mape_results_prophet_covid):.1f}%')
    print(f'The Auto Arima average MAPE results for {cross_validation_windows} cross validation windows is: {np.mean(mape_results_auto_arima):.1f}%')










    #file_names = ['10YBONDYIELDS', 'EURUSD', 'GOLD', 'SP500', 'VIX']
    #for file in file_names:
    #    df_temp, _ = read_csv_with_date(f'{file}.csv', 'zeros')
    #    df_temp = df_temp.add_prefix(f'{file}_')
    #    df = pd.merge(df, df_temp, left_index=True, right_index=True, how='left')

    



  
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout) 
    main()


