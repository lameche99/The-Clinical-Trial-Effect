import pandas as pd
import re, random

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
    clean.dropna(axis=0, inplace=True)
    df = clean.copy(deep=True).sort_values(by=['date']).reset_index(drop=True)
    return df


def special_encode(x: str, bull_words: list, bear_words: list):
    """
    This function labels a piece of text according to the keywords
    passed in.
    :param x: str - text
    :param bull_words: list - list of keywords for dovish sentiment
    :param bear_words: list - list of keywords for hawkish sentiment
    :return: int - encoding label for positive, negative or neutral sentiment
    """
    tmp_x = x.lower()
    # BULLISH
    if re.findall(pattern="|".join(bull_words), string=tmp_x):
        return 1
    # BEARISH
    elif re.findall(pattern="|".join(bear_words), string=tmp_x):
        return -1
    # NEUTRAL
    else:
        return 0


def makeRegex(wrds: list, neg: bool = False):
    """
    This function returns a regular expression to search for any of the
    given words in text
    :param wrds: list - bag of words
    :param neg: bool - whether to search for the negative of the word
    i.e. adding no/not before the word. Default is false
    :return: str - regular expression
    """
    regex = list()
    if neg:
        regex = [f"\b*no \w*{i}\w*" for i in wrds] +\
              [f"\b*not \w*{i}\w*" for i in wrds]
        return regex
    regex = [f"\b*\w*{i}\w*" for i in wrds]
    return regex


def splitData(df: pd.DataFrame, train_size: float = 0.8):
    """
    This function splits a dataset into train and test
    according to train_size
    :param df: pd.DataFrame - dataframe
    :param train_size: float - size of train dataset, default 80%
    :return: tuple(pd.DataFrame) - train and test sets
    """
    train_idx = trainIdx(train_size=train_size, length=df.shape[0])
    train = df.iloc[train_idx].copy()
    test = df.iloc[~df.index.isin(train_idx)].copy()
    return train, test


def trainIdx(train_size: float, length: int, low: int = 0):
    """
    This function returns an array of random integers for train
    test splitting
    :param train_size: float - train set size as a percentage of total size
    :param length: float - total size
    :param low: int - lower bound for integers, default 0
    :return: np.array(int) - random indices
    """
    rng = range(low, length)
    size = int(train_size * length)
    idxs = random.sample(population=rng, k=size)
    return idxs
