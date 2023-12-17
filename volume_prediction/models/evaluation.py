import numpy as np
def MAPE(evaluation_values, forecast_values):
  MAPE = 100*np.mean(abs((evaluation_values - forecast_values)/evaluation_values))

  return MAPE