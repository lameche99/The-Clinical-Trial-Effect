import pandas as pd
import numpy as np

class Trading_Strategy:
   def __init__(self):
      pass

   def find_closest_index(self, date, available_dates):
      closest_index = min(available_dates, key=lambda x: abs(pd.Timestamp(x) - pd.Timestamp(date)))
      return (available_dates == closest_index).idxmax()

   def get_period_return(self, close, date, period):
      available_dates = pd.Series(close.index)
      idx = self.find_closest_index(date, available_dates)

      return (close.iloc[idx+period] - close.iloc[idx])/close.iloc[idx]

   def buy_short(self, signal_col, test_event_df, stock_df, period):
      pos = 1
      portfolio = []
      portfolio.append(pos)
      for index, row in test_event_df.iterrows():
         try:
            signal = row[signal_col]
            close_data = stock_df[row['ticker']]['close']
            ret = self.get_period_return(close_data, row['date'], period)
            if signal != 0:
               pos = (1+ (ret * signal)) * pos
         except:
            print(f'Error at index: {index}')

         portfolio.append(pos)

      return portfolio
         