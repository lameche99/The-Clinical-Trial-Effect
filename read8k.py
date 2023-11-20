import pandas as pd
import os, re, sqlite3
import requests, gc, time
from bs4 import BeautifulSoup as bs

def parse8k(doc: str):
    """
    This function uses regular expressions (regex) to clean raw text SEC files
    :param doc: str -- string of raw text
    :return: str -- string of clean text
    """
    # remove new-lines
    doc = re.sub(r'(\r\n|\n|\r)', ' ',doc)
    # remove certain text
    doc = re.sub(r'<DOCUMENT>\s*<TYPE>(?:GRAPHIC|ZIP|EXCEL|PDF|XML|JSON).*?</DOCUMENT>', ' ', doc)
    doc = re.sub(r'<SEC-HEADER>.*?</SEC-HEADER>', ' ', doc)
    doc = re.sub(r'<IMS-HEADER>.*?</IMS-HEADER>', ' ', doc)
    # replace characters
    doc = re.sub(r'&nbsp;', ' ', doc)
    doc = re.sub(r'&#160;', ' ', doc)
    doc = re.sub(r'&amp;', '&', doc)
    doc = re.sub(r'&#38;', '&', doc)
    # replace encoded characters to whitespace
    doc = re.sub(r'&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', ' ', doc)
    soup = bs(doc, 'html.parser')
    for tag in soup.find_all('xbrl'):
        # don't remove if there is item detail
        fp_result = tag(text=re.compile(r'(?i)item\s*\d', re.IGNORECASE))
        event = len(fp_result)
        # otherwise remove
        if event == 0:
            tag.decompose()
    # remove tables
    for tag in soup.find_all('table'):
        temp = tag.get_text()
        numbers = sum(c.isdigit() for c in temp)
        letters = sum(c.isalpha() for c in temp)
        n_l_ratio = 1.0
        if (numbers + letters) > 0:
            n_l_ratio = numbers / (numbers + letters)
        event = 0
        if (event == 0) and (n_l_ratio > 0.1):
            tag.decompose()
    # remove styling text
    text = soup.get_text()
    text = re.sub(r'<(?:ix|link|xbrli|xbrldi).*?>.*?<\/.*?>', ' ', text)
    # remove extra whitespace
    text = ''.join(line.strip() for line in text.split('\n'))
    # additional cleaning
    text = re.sub(r'--;', ' ', text)
    text = re.sub(r'__', ' ', text)
    # more cleaning
    cleaner = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = re.sub(cleaner, ' ', text)
    temp_match = re.search(r'^.*?item(\s)*\d', text, flags=re.IGNORECASE)
    if temp_match != None:
        text = re.sub(r'^.*?item(\s)*\d', '', text, count=1, flags=re.IGNORECASE)
    # replace more than one whitespace with single whitespace
    text = re.sub(r'\s+', ' ', text)

    return text

def download8k(records: pd.DataFrame, year: int, qtr: str):
    for i in range(len(records)):
        tick = TICKERS[CIKS.index(records.iloc[i].cik)]
        fname = f'{tick}_{year}_{qtr}_8K.txt'
        fpath = os.path.join(FILEDIR, fname)
        furl = os.path.join(SECDIR, records.iloc[i].url)
        response = requests.get(furl, stream=True, headers= {'User-Agent': 'Mozilla/5.0'})
        with open(fpath, 'wb') as out:
            out.write(response.content)
        out.close()
        time.sleep(0.1)

def scrape8k(year: int, qtr: str):
    print(f'--- Scraping 8K for {year}-{qtr} ---')
    print(f'Companies Avaliable: {len(TICKERS)}')
    DIR8K = f'{year}/{qtr}'
    url = f'https://www.sec.gov/Archives/edgar/full-index/{year}/{qtr}/master.idx'
    lines = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True).content.decode("utf-8", "ignore").splitlines()
    records = pd.DataFrame([item.split('|') for item in lines], columns=['cik', 'name', 'filing', 'date', 'url']).iloc[11:]
    matches = records.loc[(records.cik.isin(CIKS)) & (records.filing == '8-K')].copy()
    print(f'--- Found {len(matches)} 8K records for {year}-{qtr} ---')
    cikunique = len(matches.cik.unique())
    print(f'Could not find records for {len(CIKS) - cikunique} companies.')
    return matches

def clean8k():
    corpus = list()
    for filename in os.listdir(FILEDIR):
        curpath = os.path.join(FILEDIR, filename)
        name = filename.split('_')
        with open(curpath, 'r', encoding='UTF-8', errors='ignore') as f:
            text = f.read()
        f.close()
        try:
            fclean = parse8k(doc=text)
            corpus.append([name[0], name[1], name[2], fclean])
        except Exception:
            continue
    return corpus

def main():
    years = [(2009 + i) for i in range(15)]
    quarters = ['QTR1', 'QTR2', 'QTR3', 'QTR4']
    # create list of tuples for year-quarter pairs
    yr_qtr = list()
    for y in years:
        for q in quarters:
            yr_qtr.append((y, q))
    yr_qtr = sorted(yr_qtr, key= lambda x: (x[0], x[1]))
    gc.collect()
    for yr, q in yr_qtr:
        recs = scrape8k(year=yr, qtr=q) # scrape records and file urls
        print(f'--- Downloading 8K for {yr}-{q} ---')
        download8k(records=recs, year=yr, qtr=q) # download 8Ks and store them
        time.sleep(0.1)
    
    print('--- Cleaning 8K Data ---')
    clean = clean8k()
    corpus = pd.DataFrame(clean, columns=['ticker', 'year', 'quarter', 'filing'])
    gc.collect()
    corpus.to_sql('8K', engine)

    print('--- Done. ---')
    return

if __name__ == '__main__':
    gc.enable()
    engine = sqlite3.connect('./out/sec-8ks.db')
    OUTDIR = 'out/8Ks'
    SECDIR = 'https://www.sec.gov/Archives'
    FILEDIR = os.path.join(os.getcwd(), OUTDIR)
    if OUTDIR not in os.listdir(os.getcwd()):
        os.mkdir(path=OUTDIR)
    firms = pd.read_csv('./out/ciks.csv')
    TICKERS = firms.ticker.tolist()
    CIKS = firms.cik.astype(str).tolist()
    main()