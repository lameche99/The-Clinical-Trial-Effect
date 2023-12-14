# RoBERTa-Biopharma-Trading

In this paper we implement a machine learning (ML) framework, combining fine-tuned decoder-only large language models (LLMs) like PubMedBERT and FinBERT, with current well known tree-based machine learning models like XGBoost. Our objective is to examine and predict the abnormal returns in pharmaceutical companies' stock prices due to the information contained in announcements regarding clinical trial developments. We fine-tune pre-trained LLMs on a corpus of almost 4000 unique clinical trial announcements on a text-classification task for positive, negative or neutral sentiment, achieving an F1-score of 0.86, and extract sentiment polarity scores for these announcements, as well as the respective companies 8-Ks using FinBERT. The results are aggregated and, together with technical indicators and financial information about the company, fed into XGBoost and Graph Convolutional Network models for a classification task of Normalized Cumulative Abnormal Returns.


The paper can be found [here](./The-Clinical-Trial-Effect.pdf).
