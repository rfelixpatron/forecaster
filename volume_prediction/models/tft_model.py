from darts import TimeSeries
from darts.models.forecasting.tft_model import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from sklearn.preprocessing import StandardScaler
import pandas as pd
from darts.utils.likelihood_models import QuantileRegression
from darts import TimeSeries, concatenate
from darts.metrics import mape, rmse
from models.prophet_helpers import generate_training_data_prophet
import numpy as np
from data.help_functions import read_csv_with_date

import pandas as pd

def add_rolling_average_row(df):
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


    
def tft_model(df, prediction_date):


    file_names = ['10YBONDYIELDS', 'EURUSD', 'GOLD', 'SP500', 'VIX']
    for file in file_names:
        df_temp, _ = read_csv_with_date(f'{file}.csv', 'zeros')
        df_temp = df_temp.add_prefix(f'{file}_')
        df = pd.merge(df, df_temp, left_index=True, right_index=True, how='left')

    print('TIESTO')
    print(df.head())

    temp_df = generate_training_data_prophet(df, prediction_date)

    # Preparing data to be used with Darts. It needs to be float!
    df_ds_idx = temp_df.copy()
    df_ds_idx['is_covid'] = ((df_ds_idx.index >= '2020-03-01') & (df_ds_idx.index < '2021-02-01')).astype(int)
    df_ds_idx = add_rolling_average_row(df_ds_idx)

    df_ds_idx.index = df_ds_idx.index.to_period(freq="D")
    df_ds_idx.index = df_ds_idx.index.to_timestamp()
    ts_volume_df = TimeSeries.from_series(df_ds_idx["VOLUME"][:-1].copy().astype(float)) # Not using the last row

 

    ts_covariates = TimeSeries.from_series(pd.get_dummies(df_ds_idx).drop(['VOLUME', 'GOLD_Adj Close', 'GOLD_Volume'], axis=1))
    ts_covariates = ts_covariates.stack(datetime_attribute_timeseries(ts_covariates.time_index, attribute = "day"))
    ts_covariates = ts_covariates.stack(datetime_attribute_timeseries(ts_covariates.time_index, attribute = "day_of_week"))
    ts_covariates = ts_covariates.stack(datetime_attribute_timeseries(ts_covariates.time_index, attribute = "month"))
    ts_covariates = ts_covariates.stack(datetime_attribute_timeseries(ts_covariates.time_index, attribute = "year"))    

    # Split for the train/test

    ts_train = ts_volume_df
    scaler_order = Scaler()
    scaler_order.fit_transform(ts_train)
    ts_ttrain = scaler_order.transform(ts_train) 
    ts_torderdf = scaler_order.transform(ts_volume_df)

    # make sure data are of type float
    ts_ttrain = ts_ttrain.astype(np.float32)
    ts_torderdf = ts_torderdf.astype(np.float32)

    # Transforming the covariates. The tt convention is for transformed train (ttrain)
    covariates_train = ts_covariates
    scaler_covariates = Scaler()
    scaler_covariates.fit_transform(covariates_train)
    covariates_ttrain = scaler_covariates.transform(covariates_train) 
    ts_covariates_t = scaler_covariates.transform(ts_covariates)  
    # make sure data are of type float
    covariates_ttrain = covariates_ttrain.astype(np.float32)
    # Creating the dataset for future predictions (data not extracted)
    covariates_beyond_date = datetime_attribute_timeseries( ts_volume_df.time_index, attribute="day", add_length=30)   
    covariates_beyond_date = covariates_beyond_date.stack(  datetime_attribute_timeseries(covariates_beyond_date.time_index, attribute="day_of_week")  )
    covariates_beyond_date = covariates_beyond_date.stack(  datetime_attribute_timeseries(covariates_beyond_date.time_index, attribute="month")  )
    covariates_beyond_date = covariates_beyond_date.stack(  datetime_attribute_timeseries(covariates_beyond_date.time_index, attribute="year")  )

    covariates_beyond_date_train = covariates_beyond_date


    # rescale the covariates: fitting on the training set
    scaler_date = Scaler()
    scaler_date.fit(covariates_beyond_date_train)
    covariates_beyond_date_ttrain = scaler_date.transform(covariates_beyond_date_train)
    covariates_beyond_tdate = scaler_date.transform(covariates_beyond_date)
    covariates_beyond_tdate = covariates_beyond_tdate.astype(np.float32)


    ts_cov_all = ts_covariates.concatenate( covariates_beyond_date.slice_intersect(ts_covariates), axis=1 )                     # unscaled F+T
    ts_cov_all_t = ts_covariates_t.concatenate( covariates_beyond_tdate.slice_intersect(ts_covariates_t), axis=1 )              # scaled F+T
    ts_cov_train_t = covariates_ttrain.concatenate( covariates_beyond_date_ttrain.slice_intersect(covariates_ttrain), axis=1 )  # scaled F+T training set



    # Standard values as shown in the example. 200 Epochs will take about 10-15 minutes?
    EPOCHS = 1
    INLEN = 32          # input size
    HIDDEN = 64         # hidden layers    
    LSTMLAYERS = 2      # recurrent layers
    ATTH = 4            # attention heads
    BATCH = 32          # batch size
    LEARN = 1e-3        # learning rate
    DROPOUT = 0.1       # dropout rate
    VALWAIT = 1         # epochs to wait before evaluating the loss on the test/validation set
    N_FC = 1            # output size
    RAND = 42           # random seed
    N_SAMPLES = 100     # number of times a prediction is sampled from a probabilistic model
    N_JOBS = 3          # parallel processors to use;  -1 = all processors

    # default quantiles for QuantileRegression
    # This turned out to be very important for the results. Check QuantileRegression.
    QUANTILES = [0.01, 0.1, 0.5, 0.9, 0.99]

    FIGSIZE = (9, 6)

    qL1, qL2 = 0.01, 0.10        # percentiles of predictions: lower bounds
    qU1, qU2 = 1-qL1, 1-qL2,     # upper bounds derived from lower bounds
    label_q1 = f'{int(qU1 * 100)} / {int(qL1 * 100)} percentile band'
    label_q2 = f'{int(qU2 * 100)} / {int(qL2 * 100)} percentile band'

    model_tft_volume_forecast = TFTModel(   input_chunk_length=INLEN,
                        output_chunk_length=N_FC,
                        hidden_size=HIDDEN,
                        lstm_layers=LSTMLAYERS,
                        num_attention_heads=ATTH,
                        dropout=DROPOUT,
                        batch_size=BATCH,
                        n_epochs=EPOCHS,                        
                        nr_epochs_val_period=VALWAIT, 
                        likelihood=QuantileRegression(QUANTILES), 
                        optimizer_kwargs={"lr": LEARN}, 
                        model_name="VolumeForecaster",
                        log_tensorboard=True,
                        random_state=RAND,
                        force_reset=True,
                        save_checkpoints=True
                    )


    model_tft_volume_forecast.fit(  series=ts_ttrain, 
                    future_covariates=ts_cov_all_t.astype(np.float32), #It only takes the values from the dates required for each train/test
                    val_future_covariates=ts_cov_all_t.astype(np.float32), 
                    verbose=True)#

    ts_pred_t = model_tft_volume_forecast.predict(   n=1, 
                                num_samples=N_SAMPLES,   
                                n_jobs=N_JOBS, 
                                verbose=True)

    
    ts_q = scaler_order.inverse_transform(ts_pred_t.quantile_timeseries(0.5))
    s = TimeSeries.pd_series(ts_q)
    return s.to_frame()


