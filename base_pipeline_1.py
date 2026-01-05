import pyterrier as pt
import pandas as pd
import json
import os
from functions import keywords_extractor, thesaurus_based_expansion
from pyterrier.measures import *

#pt.init()


# GET THE QUERIES AND QRELS

with open("test_queries.json", "r", encoding = "utf-8") as file:
    data = json.load (file)
test_queries = pd.DataFrame.from_dict(data)
test_queries.rename(columns={"query_id": "qid", "question": "query"}, inplace=True)
# get a subset cuz otherwise it was taking 78 minutes with 10'000 queries
test_queries_small = test_queries.sample(1000, random_state=42) 


with open("test_qrels.json", "r", encoding = "utf-8") as file:
    data = json.load (file)
test_qrels = pd.DataFrame.from_dict(data)
test_qrels.rename(columns={"query_id": "qid", "para_id": "docno"}, inplace=True)



# GET THE INDEX
index_path = os.path.abspath("terrier_index")  # absolute path
index = pt.IndexFactory.of(index_path)



# USING bm25 vs tfidf with pseudo relevance feedback

bm25 = pt.terrier.Retriever(index, wmodel="BM25")
rm3_pipe_bm25 = bm25 >> pt.rewrite.RM3(index) >> bm25

tfidf = pt.terrier.Retriever(index, wmodel="TF_IDF")
rm3_pipe_tfidf = tfidf >> pt.rewrite.RM3(index) >> tfidf

resultsbm25 = pt.Experiment(
    [bm25, rm3_pipe_bm25],
    test_queries_small,
    test_qrels,
    ["map", AP@1000,P@5,P@10]
    )
print(resultsbm25)

resultstfidf = pt.Experiment(
    [bm25, rm3_pipe_tfidf],
    test_queries_small,
    test_qrels,
    ["map", AP@1000,P@5,P@10]
    )
print(resultstfidf)
