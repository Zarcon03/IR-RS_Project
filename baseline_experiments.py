import json
import os
import pandas as pd
import pyterrier as pt
from eval_metrics import EVAL_METRICS
from functions import keywords_extractor, thesaurus_based_expansion
from tqdm import tqdm
tqdm.pandas()


class BaselineExperiments():
    def __init__(self, queries_path, qrels_path, base_index_path = "terrier_index", keywords_index_path = "keywords_index"):
        self.queries_path = queries_path
        self.qrels_path = qrels_path
        self.base_index_path = base_index_path
        self.keywords_index_path = keywords_index_path
        self.queries_expanded = False
        with tqdm(total=4, desc="Loading indexes, queries and qrels...", unit="step") as pbar:
            self.load_base_index()
            pbar.update(1)
            self.load_keywords_index()
            pbar.update(1)
            self.load_queries()
            pbar.update(1)
            self.load_qrels()
            pbar.update(1)

    def load_base_index(self):
        index_abs_path = os.path.abspath(self.base_index_path)
        self.base_index = pt.IndexFactory.of(index_abs_path)

    def load_queries(self):
        with open(self.queries_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            self.queries = pd.DataFrame.from_dict(data)
            self.queries.rename(columns={"query_id": "qid", "question": "query"}, inplace=True)
            self.queries_small = self.queries.sample(1000, random_state=42)
        
    def load_qrels(self):
        with open(self.qrels_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            self.qrels = pd.DataFrame.from_dict(data)
            self.qrels.rename(columns={"query_id": "qid", "para_id": "docno"}, inplace=True)
    
    def load_keywords_index(self):
        index_abs_path = os.path.abspath(self.keywords_index_path)
        self.keywords_index = pt.IndexFactory.of(index_abs_path)
    
    def thesaurus_query_expansion(self, queries: pd.DataFrame) -> pd.DataFrame:
        expanded_queries = queries.copy()

        expanded_queries["query_0"] = expanded_queries["query"]

        tqdm.pandas(desc="Expanding queries with thesaurus")
        expanded_queries["query"] = expanded_queries["query"].progress_apply(
            lambda q: q + " " + " ".join(
                thesaurus_based_expansion(q, keywords_extractor(q))
            )
        )
        self.queries_expanded = True
        return expanded_queries

    def run_experiment_1(self):
        bm25 = pt.terrier.Retriever(self.base_index, wmodel="BM25")
        rm3_pipe_bm25 = bm25 >> pt.rewrite.RM3(self.base_index) >> bm25

        tfidf = pt.terrier.Retriever(self.base_index, wmodel="TF_IDF")
        rm3_pipe_tfidf = tfidf >> pt.rewrite.RM3(self.base_index) >> tfidf

        print("Experiment 1:")
        experiment1_results = pt.Experiment(
            [bm25, rm3_pipe_bm25, tfidf, rm3_pipe_tfidf],
            self.queries,
            self.qrels,
            EVAL_METRICS,
            verbose=True
        )
        print(experiment1_results)
    
    def run_experiment_2(self):
        if not self.queries_expanded:
            self.expanded_queries = self.thesaurus_query_expansion(self.queries)
        
        bm25 = pt.terrier.Retriever(self.base_index, wmodel="BM25")
        rm3_pipe_bm25 = bm25 >> pt.rewrite.RM3(self.base_index) >> bm25

        tfidf = pt.terrier.Retriever(self.base_index, wmodel="TF_IDF")
        rm3_pipe_tfidf = tfidf >> pt.rewrite.RM3(self.base_index) >> tfidf

        print("Experiment 2:")
        experiment2_results = pt.Experiment(
            [bm25, rm3_pipe_bm25, tfidf, rm3_pipe_tfidf],
            self.expanded_queries,
            self.qrels,
            EVAL_METRICS,
            verbose=True
            )
        print(experiment2_results)
    
    def run_experiment_3(self):
        if not self.queries_expanded:
            self.expanded_queries = self.thesaurus_query_expansion(self.queries)

        bm_25 = pt.terrier.Retriever(self.keywords_index, wmodel="BM25")
        rm3_pipe_bm25 = bm_25 >> pt.rewrite.RM3(self.keywords_index) >> bm_25

        print("Experiment 3 with original queries:")
        experiment3_results_a = pt.Experiment(
            [bm_25, rm3_pipe_bm25],
            self.queries,
            self.qrels,
            EVAL_METRICS,
            verbose=True
            )
        print(experiment3_results_a)
        
        print("Experiment 3 with expanded queries:")
        experiment3_results_b = pt.Experiment(
            [bm_25, rm3_pipe_bm25],
            self.expanded_queries,
            self.qrels,
            EVAL_METRICS,
            verbose=True
            )
        print(experiment3_results_b)
