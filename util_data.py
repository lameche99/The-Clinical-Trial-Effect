import pandas as pd
import sqlite3

def cleanCat(raw: pd.DataFrame):
    """
    This function cleans a raw excel file
    of clinical trial announcements
    :param raw: pd.DataFrame - raw input table
    :return: pd.DataFrame - clean catalyst data
    """
    clean = raw.loc[:, raw.columns[:5]].copy()
    clean.columns = ['ticker', 'disease', 'stage', 'date', 'catalyst']
    clean['ticker'] = clean.ticker.str.split('Add', expand=True)[0]
    clean['date'] = pd.to_datetime(clean.date.str.removesuffix(' ET'),
                                            format='%d/%m/%Y')
    return clean


def main():
    data_raw = pd.read_excel(BIOPHARM, sheet_name='catalysts', header=None)
    data_clean = cleanCat(raw=data_raw)
    data_clean.to_sql('catalysts', ENGINE, if_exists='replace', index=False)



if __name__ == '__main__':
    ENGINE = sqlite3.connect('./src/data.db')
    BIOPHARM = './src/bio-data.xlsm'
    main()