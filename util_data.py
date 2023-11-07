import pandas as pd
import numpy as np
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

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
    return clean


def extract_keywords(doc: str,
                     stop_words: list,
                     model: SentenceTransformer,
                     top_n: int = 5,
                     n_gram_range: tuple = (1,3)):
    """
    This function extracts relevant keywords of variable length n_gram_range in a sentence 
    and returns the top_n most relevant words using a Transformer-based model to generate text
    embeddings. The relevant keywords are selected by cosine-similarity
    :param doc: str - text
    :param stop_words: list - list of stop words for maksing
    :param model: SentenceTransformer - transformer based model for embeddings
    :param top_n: int - top n keywords to return, default 5
    :param n_gram_range: tuple - range length of keywords, default (1 to 3)
    :return: list - list of relevant keywords in the text
    """
    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
    candidates = count.get_feature_names_out()
    # create embeddings
    doc_embedding = model.encode([doc])
    candidate_embeddings = model.encode(candidates)
    # calculate distance between doc and candidates
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
    return keywords


def class_keywords(classifier, kwrds: pd.Series):
    """
    This function runs a classifier to retrieve sentiment
    scores for a list of keywords and returns a dataframe
    with the corresponding keyword score pairing
    :param classifier: some datatype - LLM classifier
    :param kwrds: pd.Series - series of keywords per catalyst
    :return: pd.DataFrame - dataframe of keyword-sentiment score pair
    """
    flat_kw = np.ravel(kwrds.to_list()).tolist() # flatten list
    kw_class = classifier(flat_kw) # sentiment score with LLM classifier
    kw_df = pd.DataFrame(kw_class)
    kw_df['keywords'] = flat_kw # add keywords
    kw_df = kw_df.sort_values(['score'], ascending=False)
    return kw_df


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


def main():
    data_raw = pd.read_excel(BIOPHARM, sheet_name='catalysts', header=None)
    data_clean = cleanCat(raw=data_raw)
    data_clean.to_sql('catalysts', ENGINE, if_exists='replace', index=False)



if __name__ == '__main__':
    ENGINE = sqlite3.connect('./src/data.db')
    BIOPHARM = './src/bio-data.xlsm'
    main()