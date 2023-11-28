import pandas as pd
import sqlite3
from util.util_data import cleanCat

def main():
    data_raw = pd.read_excel(BIOPHARM, sheet_name='catalysts', header=None)
    data_clean = cleanCat(raw=data_raw)
    data_clean.to_sql('catalysts', ENGINE, if_exists='replace', index=False)


if __name__ == '__main__':
    ENGINE = sqlite3.connect('./data.db')
    BIOPHARM = './bio-data.xlsm'
    main()