import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.tft_model import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from help_functions.help_functions import load_config, read_csv_with_date
from models.prophet_helpers import generate_training_data_prophet


def add_rolling_average_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new row with the rolling average for the last 15 days (excluding zero values) to a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with time series data.

    Returns:
    - pd.DataFrame: Updated DataFrame with the new row.
    """
    # Create a new row with the next date
    next_date = df.index[-1] + pd.DateOffset(days=1)
    new_row = pd.Series(index=df.columns, name=next_date)

    # Calculate the rolling average for the last 15 days excluding zero values
    for column in df.columns:
        last_15_days = df[column].rolling(window=15).apply(lambda x: x[x != 0].mean(), raw=True).iloc[-1]
        new_row[column] = last_15_days if not pd.isna(last_15_days) else df[column].iloc[-1]

    # Create a new DataFrame with only the new row
    new_df = pd.DataFrame([new_row], index=[next_date])

    # Concatenate the new DataFrame to the original DataFrame
    result_df = pd.concat([df, new_df])

    return result_df


def prepare_data_for_darts(df: pd.DataFrame, covid_config: dict, target_column: str = "VOLUME") -> (TimeSeries, TimeSeries):
    """
    Prepares data for Darts library.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with time series data.
    - covid_config (dict): Configuration dictionary for COVID-related parameters.
    - target_column (str): Name of the column to use for creating the TimeSeries. Default is "VOLUME".

    Returns:
    - TimeSeries: Darts TimeSeries object for the specified column.
    - TimeSeries: Darts TimeSeries object for covariates.
    """
    # Copy the DataFrame to avoid modifying the original
    df_ds_idx = df.copy()

    # Create a binary column indicating COVID period
    df_ds_idx['is_covid'] = ((df_ds_idx.index >= covid_config['covid_start_date']) & (df_ds_idx.index < covid_config['covid_end_date'])).astype(int)

    # Add rolling average row
    df_ds_idx = add_rolling_average_row(df_ds_idx)

    # Convert the index to datetime period and then back to timestamp
    df_ds_idx.index = df_ds_idx.index.to_period(freq="D").to_timestamp()

    # Convert the specified column to a float and create a TimeSeries object
    # Remove the row containing the rolling average
    ts_target = TimeSeries.from_series(df_ds_idx[target_column][:-1].astype(float))

    # Create covariates TimeSeries
    ts_covariates = TimeSeries.from_series(pd.get_dummies(df_ds_idx).drop([target_column, 'GOLD_Adj Close', 'GOLD_Volume'], axis=1))
    ts_covariates = ts_covariates.stack(datetime_attribute_timeseries(ts_covariates.time_index, attribute="day"))
    ts_covariates = ts_covariates.stack(datetime_attribute_timeseries(ts_covariates.time_index, attribute="day_of_week"))
    ts_covariates = ts_covariates.stack(datetime_attribute_timeseries(ts_covariates.time_index, attribute="month"))
    ts_covariates = ts_covariates.stack(datetime_attribute_timeseries(ts_covariates.time_index, attribute="year"))

    return ts_target, ts_covariates


def prepare_data_for_training(ts_train: TimeSeries, ts_covariates: TimeSeries) -> (TimeSeries, TimeSeries, Scaler, TimeSeries):
    """
    Prepares data for training.

    Parameters:
    - ts_train (TimeSeries): Target time series for training.
    - ts_covariates (TimeSeries): Covariates time series.

    Returns:
    - TimeSeries: Scaled and transformed target time series for training.
    - TimeSeries: Scaled and transformed covariates for future predictions.
    - Scaler: Scaler used for the target time series.
    """
    # Scaled and transformed target time series for training
    scaler_target = Scaler()
    scaler_target.fit_transform(ts_train)
    ts_ttrain = scaler_target.transform(ts_train).astype(np.float32)

    # Rescale the covariates: fitting on the training set
    scaler_date = Scaler()
    scaler_date.fit(ts_covariates)
    ts_covariates_t = scaler_date.transform(ts_covariates).astype(np.float32)

    # Concatenate the scaled covariates for training and future predictions
    #ts_cov_all_t = ts_covariates.concatenate(covariates_beyond_tdate.slice_intersect(ts_covariates), axis=1)

    return ts_ttrain, ts_covariates_t, scaler_target




    
def tft_model(df, prediction_date):


    # Load files with features
    file_names = ['10YBONDYIELDS', 'EURUSD', 'GOLD', 'SP500', 'VIX']
    for file in file_names:
        df_temp, _ = read_csv_with_date(f'{file}.csv', 'zeros')
        df_temp = df_temp.add_prefix(f'{file}_')
        df = pd.merge(df, df_temp, left_index=True, right_index=True, how='left')


    temp_df = generate_training_data_prophet(df, prediction_date)

    # Load configuration from file
    config = load_config()
    covid_config = config['model_config']['prophet_config']
    tft_config = config['model_config']['tft_config']

    ts_train, ts_covariates = prepare_data_for_darts(temp_df, covid_config)

    ts_ttrain, ts_cov_all_t, scaler_target = prepare_data_for_training(ts_train, ts_covariates)


    model_tft_volume_forecast = TFTModel(   
        input_chunk_length=tft_config["INLEN"], # input size
                        output_chunk_length=tft_config["N_FC"], # output size
                        hidden_size=tft_config["HIDDEN"], # hidden layers    
                        lstm_layers=tft_config["LSTMLAYERS"], # recurrent layers
                        num_attention_heads=tft_config["ATTH"], # attention heads
                        dropout=tft_config["DROPOUT"], # dropout rate
                        batch_size=tft_config["BATCH"], # batch size
                        n_epochs=tft_config["EPOCHS"],                        
                        nr_epochs_val_period=tft_config["VALWAIT"], # epochs to wait before evaluating the loss on the test/validation set
                        likelihood=QuantileRegression(tft_config["QUANTILES"]), 
                        optimizer_kwargs={"lr": tft_config["LEARN"]}, # learning rate
                        model_name="VolumeForecaster",
                        log_tensorboard=True,
                        random_state=tft_config["RAND"], # random seed
                        force_reset=True,
                        save_checkpoints=True
                    )


    model_tft_volume_forecast.fit(  series=ts_ttrain, 
                    future_covariates=ts_cov_all_t.astype(np.float32), #It only takes the values from the dates required for each train/test
                    val_future_covariates=ts_cov_all_t.astype(np.float32), 
                    verbose=True)#

    ts_pred_t = model_tft_volume_forecast.predict(   n=1, 
                                num_samples=tft_config["N_SAMPLES"], # number of times a prediction is sampled from a probabilistic model
                                n_jobs=tft_config["N_JOBS"], # parallel processors to use;  -1 = all processors
                                verbose=True)

    
    ts_q = scaler_target.inverse_transform(ts_pred_t.quantile_timeseries(0.5))
    s = TimeSeries.pd_series(ts_q)
    return s.to_frame()


