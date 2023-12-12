import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.impute import KNNImputer
import numpy as np
import sqlite3

class NCARS:
   def __init__(self, train_size=90, test_size=20, columns=[], period=None, extra_data=None, stockfpath="", eventdbpath="", eventtable="", lag=1):
      self.columns = columns
      self.extra_data = extra_data
      self.stockdata = self.read_stockdata(stockfpath, extra_data)
      self.eventdata = self.read_eventdata(eventdbpath, eventtable)
      self.train_size = train_size
      self.test_size = test_size
      self.lag = lag
      self.period = period

   def read_stockdata(self, fpath, extra_data):
      stock_data = pd.read_csv(fpath, compression='zip', header=[0, 1], index_col=0, parse_dates=True)
      stock_data.index = stock_data.index.date
      if extra_data is not None:
         extra_name = extra_data.name
         stock_data = pd.merge(stock_data, extra_data, left_index=True, right_index=True, how='inner')
         stock_data = stock_data.rename(columns={extra_name: ('EXTERNAL', extra_name)})
         stock_data.columns = pd.MultiIndex.from_tuples(stock_data.columns)
      return stock_data
   
   def read_eventdata(self, dbpath, table):
      conn = sqlite3.connect(dbpath)
      query = 'SELECT * FROM ' + table
      events_df = pd.read_sql(query, conn)
      conn.close()
      events_df['date'] = pd.to_datetime(events_df['date'])
      return events_df
   
   def find_closest_index(self, date, available_dates):
      closest_index = min(available_dates, key=lambda x: abs(pd.Timestamp(x) - pd.Timestamp(date)))
      return (available_dates == closest_index).idxmax()

   def prepare_data(self, stock_df, forecast_dates):
      samples = {}

      available_dates = stock_df.iloc[self.lag:, stock_df.columns.get_loc('date')]

      if not isinstance(forecast_dates, list):
         forecast_dates = [forecast_dates]

      for date in forecast_dates:
         closest_index = self.find_closest_index(date, available_dates)

         # ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
         X = stock_df[stock_df.columns.difference(['date'])].shift(self.lag).iloc[self.lag:, :].ffill()
         y = stock_df.iloc[self.lag:, stock_df.columns.get_loc('close')].ffill()

         train_end = closest_index
         train_start = train_end - self.train_size
         test_start = closest_index
         test_end = test_start + self.test_size

         X_train = X.iloc[train_start:train_end, :]
         y_train = y.iloc[train_start:train_end]
         X_test = X.iloc[test_start:test_end, :]
         y_test = y.iloc[test_start:test_end]

         # If only one date is provided, return a single-item dictionary
         if len(forecast_dates) == 1:
            return (X_train, y_train, X_test, y_test)

         samples[date] = (X_train, y_train, X_test, y_test)

      return samples
   
   def plot_ncar(self, preds, true_vals, ticker, prev_val=None):
      ncar = self.get_ncar(preds, true_vals, prev_val)
      # Plot both series with different colors
      plt.plot(true_vals.reset_index(drop=True), label='True Values', color='black')
      plt.plot(preds.reset_index(drop=True), label='NBI Index', color='red')

      # # Annotate a specific point on the plot in the corner
      # annotate_index = 2
      # annotate_value = series1[annotate_index]

      plt.annotate(f'NCAR: {ncar}', xy=(0.05, 0.05), xycoords='axes fraction',
                  xytext=(0.05, 0.05), textcoords='axes fraction')

      # Add labels, title, and legend
      plt.xlabel('Time')
      plt.ylabel('Value')
      plt.title(ticker)
      plt.legend()

      # Show the plot
      plt.show()
   
   def get_ncar(self, preds, true_vals, prev_val=None):
      if prev_val is not None:
         prev_series = pd.Series([prev_val])
         preds = pd.concat(objs=[prev_series, preds])
         true_vals = pd.concat(objs=[prev_series, true_vals])
      pred_ret = preds.pct_change()
      pred_ret.dropna(inplace=True)
      pred_ret = pred_ret.reset_index(drop=True)
      true_ret = true_vals.pct_change()
      true_ret.dropna(inplace=True)
      true_ret = true_ret.reset_index(drop=True)

      abn_ret = true_ret - pred_ret
      car = abn_ret.sum()
      ncar = car / pred_ret.sum()
      return ncar

   def get_event_preds(self, model, index):
      if self.extra_data is None:
         stock_data = self.stockdata[self.eventdata.iloc[index]['ticker']]
      else:
         stock_data = self.stockdata[[self.eventdata.iloc[index]['ticker'], 'EXTERNAL']]
         stock_data = stock_data.droplevel(level=0, axis=1)
      stock_data.reset_index(inplace=True)
      stock_data = stock_data.rename(columns={'index': 'date'})
      stock_data['date'] = pd.to_datetime(stock_data['date'])
      event_date = self.eventdata.iloc[index]['date']
      X_train, y_train, X_test, y_test = self.prepare_data(stock_data, event_date)
      model.fit(X_train, y_train)
      predictions = model.predict(X_test)
      return predictions, X_train, y_train, X_test, y_test
   
   def get_event_data(self, index, period, normalize=False):
      close_data = self.stockdata[self.eventdata.iloc[index]['ticker']]['close']
      biotech_index = self.stockdata['EXTERNAL']['NBI']
      available_dates = pd.Series(close_data.index)
      idx = self.find_closest_index(self.eventdata.iloc[index]['date'], available_dates)

      # Getting the period start and end
      period_end = idx + period
      period_start = idx

      close_data_sub = close_data.iloc[period_start:period_end]
      biotech_index_sub = biotech_index.iloc[period_start:period_end]

      if normalize:
         close_data_sub = close_data_sub/close_data_sub[0]
         biotech_index_sub = biotech_index_sub/biotech_index_sub[0]
      return close_data_sub, biotech_index_sub
      

   def get_ncars(self, model):
      ncars = []

      for index, row in self.eventdata.iterrows():
         try:
            predictions, _, y_train, _, y_test = self.get_event_preds(model, index)
            prev_val = y_train.iloc[-1]
            ncar = self.get_ncar(predictions, y_test, prev_val)
         except Exception as e:
            print(f'Error: {e}')
            print(f"Ticker: {row['ticker']}")
            print(f"Date: {row['date']}")
            print("")
            ncar = np.nan
         ncars.append(ncar)

      return pd.Series(ncars)
   
   def get_index_ncar(self, index, period, normalize=False):
      biotech_index_sub, close_data_sub = self.get_event_data(index, period, normalize)
      ncar = self.get_ncar(biotech_index_sub, close_data_sub)
      return ncar
   
   def get_index_ncars(self, normalize=False):
      ncars = []

      for index, row in self.eventdata.iterrows():
         try:
            ncar = self.get_index_ncar(index, self.period, normalize)
         except Exception as e:
            print(f'Error: {e}')
            print(f"Ticker: {row['ticker']}")
            print(f"Date: {row['date']}")
            print("")
            ncar = np.nan
         ncars.append(ncar)

      return pd.Series(ncars)



   



class GBRegressor:
   def __init__(self, n_estimators=100, learning_rate=0.1):
      self.model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)

   def fit(self, X_train, y_train):
      self.model.fit(X_train, y_train)

   def predict(self, X_test):
      y_pred = self.model.predict(X_test)
      return pd.Series(y_pred)
   
class RFRegressor:
   def __init__(self, n_estimators=100, max_depth=None):
      self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

   def fit(self, X_train, y_train):
      self.model.fit(X_train, y_train)

   def predict(self, X_test):
      y_pred = self.model.predict(X_test)
      return pd.Series(y_pred) 

# class LSTMRegressor:
#     def __init__(self, input_shape, lstm_units=50, dense_units=1):
#         self.model = Sequential()
#         self.model.add(LSTM(units=lstm_units, input_shape=(None, input_shape), activation='relu'))
#         self.model.add(Dense(units=dense_units))

#         self.model.compile(optimizer='adam', loss='mean_squared_error')

#     def fit(self, X_train, y_train, epochs=50, batch_size=32):
#         self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

#     def predict(self, X_test):
#         return self.model.predict(X_test)

# # Example usage:
# # Assume X_train and y_train are your training data and labels, respectively.
# # X_train should have the shape (number of samples, number of time steps, number of features).

# # Instantiate the model
# input_shape = 10  # number of features
# lstm_regressor = LSTMRegressor(input_shape=input_shape)

# # Train the model
# # X_train and y_train are assumed to be your training data and labels
# # Make sure to shape your data appropriately
# lstm_regressor.train(X_train, y_train, epochs=50, batch_size=32)

# # Make predictions
# # X_test is assumed to be your test data
# predictions = lstm_regressor.predict(X_test)
