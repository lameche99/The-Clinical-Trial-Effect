import csv
from polygon import RESTClient
import pandas as pd
from datetime import datetime, timedelta
import pytz

def read_csv_to_list(file_path):
   """
   Read a CSV file and store its content as a list of strings.

   Args:
   file_path (str): The path to the CSV file.

   Returns:
   list: A list of strings read from the CSV file.
   """
   string_list = []

   try:
      with open(file_path, 'r', newline='') as csv_file:
         csv_reader = csv.reader(csv_file)
         for row in csv_reader:
               for item in row:
                  string_list.append(item)

         csv_file.close()

      return string_list

   except FileNotFoundError:
      print(f"File not found: {file_path}")
      return None

# Function to check for API errors and handle them
def get_data(client, ticker, start_date, end_date):
   cols = pd.MultiIndex.from_product([[ticker], ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']])
   
   df = pd.DataFrame(columns=cols)
   time_stamps = []
   while start_date < end_date:
      try:
         current_period = start_date.strftime("%Y-%m-%d")
         next_period = (start_date + timedelta(days=1000)).strftime("%Y-%m-%d")
         for a in client.list_aggs(
               ticker,
               15,
               "minute",
               current_period,
               next_period,
               limit=50000,
         ):
               time_stamps.append(a.timestamp)
               data = {
                  (ticker,'open'): [a.open],
                  (ticker,'high'): [a.high], 
                  (ticker,'low'): [a.low],
                  (ticker,'close'): [a.close],
                  (ticker,'volume'): [a.volume],
                  (ticker,'vwap'): [a.vwap],
                  (ticker,'transactions'): [a.transactions]
               }
               new_row = pd.DataFrame(columns=cols, data=data)
               df = pd.concat([df, new_row], ignore_index=True)
      except:
         print("{" + ticker + "}" + " No Data for " + current_period + " to " + next_period + ".")
      start_date = start_date + timedelta(days=1001)

   # Set the timezone to UTC
   utc_time = pd.to_datetime(time_stamps, unit='ms', utc=True)
   # Convert to Eastern Time
   eastern = pytz.timezone('US/Eastern')
   eastern_time = utc_time.tz_convert(eastern)
   df.index = eastern_time

   return df #pd.Series(close_prices, index=pd.to_datetime(time_stamps, unit='s'), name=ticker

# Define your API key here
api_key = 'FuGEoCgmKhdpXJVy7pNqWD_TlARMHuMa'

# Initialize the RESTClient with the API key
client = RESTClient(api_key)

# Replace with the path to your CSV file containing tickers
tickers_csv_path = "tickers.csv"
ticker_list = read_csv_to_list(tickers_csv_path)
# ticker_list = ['AADI']

combined_df = pd.DataFrame()

# Iterate through tickers and retrieve data
start_date = datetime(2015, 1, 1)
end_date = datetime(2023, 10, 1)
for ticker in ticker_list:
    df = get_data(client, ticker, start_date, end_date)
    if not df.empty:
        combined_df = pd.concat([combined_df, df], axis=1)

# Sort the DataFrame by its datetime index
combined_df.sort_index(inplace=True)

# Save the DataFrame to a CSV file
combined_df.to_csv("stock_data.csv")