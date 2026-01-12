import pandas as pd
import pyterrier as pt
from constants import EVAL_METRICS, RESULTS_FOLDER
from functions import keywords_extractor, thesaurus_based_expansion
from tqdm import tqdm
from collection import BenchmarkCollection
from indexes import BenchmarkIndex
from pathlib import Path
from pyterrier_dr import RetroMAE
from pyterrier_t5 import MonoT5ReRanker

class BenchmarkExperiments():
    def __init__(self, collection: BenchmarkCollection, indexes: BenchmarkIndex, results_folder=RESULTS_FOLDER):
        self.collection = collection
        self.indexes = indexes
        self.results_folder = Path(results_folder).resolve()

        if not (hasattr(self.collection, "queries") and hasattr(self.collection, "qrels")):
            raise RuntimeError("Queries and Qrels must be loaded before running experiments.")
        
        self.create_results_folder()

    def create_results_folder(self):
        self.results_folder.mkdir(parents=True, exist_ok=True)
        
    def thesaurus_query_expansion(self, queries: pd.DataFrame) -> pd.DataFrame:
        expanded_queries = queries.copy()

        expanded_queries["query_0"] = expanded_queries["query"]

        tqdm.pandas(desc="Expanding queries with thesaurus")
        expanded_queries["query"] = expanded_queries["query"].progress_apply(
            lambda q: q + " " + " ".join(
                thesaurus_based_expansion(q, keywords_extractor(q))
            )
        )
        return expanded_queries

    def run_experiment_1(self, test_on_sample=True):
        if not hasattr(self.indexes, "basic_index"):
            raise RuntimeError("To run this experiment basic_index must be loaded, try load_basic_index()")
        
        bm25 = pt.terrier.Retriever(self.indexes.basic_index, wmodel="BM25")
        rm3_pipe_bm25 = bm25 >> pt.rewrite.RM3(self.indexes.basic_index) >> bm25

        tfidf = pt.terrier.Retriever(self.indexes.basic_index, wmodel="TF_IDF")
        rm3_pipe_tfidf = tfidf >> pt.rewrite.RM3(self.indexes.basic_index) >> tfidf
        
        if test_on_sample:
            if not hasattr(self.collection, "queries_sample"):
                raise RuntimeError("No sampled queries available. Call sample_queries() first.")

            queries_to_use = self.collection.queries_sample
            print(f"Running Experiment 1 on sampled queries ({len(queries_to_use)} queries).")
            names = [
                f"experiment1_bm25_sample_{len(queries_to_use)}_queries", 
                f"experiment1_bm25_rm3_sample_{len(queries_to_use)}_queries",
                f"experiment1_tfidf_sample_{len(queries_to_use)}_queries",
                f"experiment1_tfidf_rm3_sample_{len(queries_to_use)}_queries"
            ]
        else:
            queries_to_use = self.collection.queries
            print(f"Running Experiment 1 on full query set ({len(queries_to_use)} queries).")
            names = [
                "experiment1_bm25", 
                "experiment1_bm25_rm3",
                "experiment1_tfidf",
                "experiment1_tfidf_rm3"
            ]
        
        save_dir = self.results_folder / "experiment_1"
        save_dir.mkdir(parents=True, exist_ok=True)

        experiment1_results = pt.Experiment(
            [bm25, rm3_pipe_bm25, tfidf, rm3_pipe_tfidf],
            queries_to_use,
            self.collection.qrels,
            EVAL_METRICS,
            verbose=True,
            names =names,
            save_dir=save_dir,
            save_mode="reuse",
            save_format="trec"
        )
        print(experiment1_results)
    
    def run_experiment_2(self, test_on_sample=True):
        if not hasattr(self.indexes, "basic_index"):
            raise RuntimeError("To run this experiment basic_index must be loaded, try load_basic_index()")
        
        bm25 = pt.terrier.Retriever(self.indexes.basic_index, wmodel="BM25")
        rm3_pipe_bm25 = bm25 >> pt.rewrite.RM3(self.indexes.basic_index) >> bm25

        tfidf = pt.terrier.Retriever(self.indexes.basic_index, wmodel="TF_IDF")
        rm3_pipe_tfidf = tfidf >> pt.rewrite.RM3(self.indexes.basic_index) >> tfidf

        if test_on_sample:
            if not hasattr(self.collection, "queries_sample"):
                raise RuntimeError("No sampled queries available. Call sample_queries() first.")

            queries_to_use = self.collection.queries_sample
            print(f"Running Experiment 2 on sampled queries ({len(queries_to_use)} queries).")
            names = [
                f"experiment2_bm25_sample_{len(queries_to_use)}_queries", 
                f"experiment2_bm25_rm3_sample_{len(queries_to_use)}_queries",
                f"experiment2_tfidf_sample_{len(queries_to_use)}_queries",
                f"experiment2_tfidf_rm3_sample_{len(queries_to_use)}_queries"
            ]
        else:
            queries_to_use = self.collection.queries
            print(f"Running Experiment 2 on full query set ({len(queries_to_use)} queries).")
            names = [
                "experiment2_bm25", 
                "experiment2_bm25_rm3",
                "experiment2_tfidf",
                "experiment2_tfidf_rm3"
            ]
        
        save_dir = self.results_folder / "experiment_2"
        save_dir.mkdir(parents=True, exist_ok=True)
            
        expanded_queries = self.thesaurus_query_expansion(queries_to_use)
        experiment2_results = pt.Experiment(
            [bm25, rm3_pipe_bm25, tfidf, rm3_pipe_tfidf],
            expanded_queries,
            self.collection.qrels,
            EVAL_METRICS,
            verbose=True,
            names=names,
            save_dir=save_dir,
            save_mode="reuse",
            save_format="trec"
            )
        print(experiment2_results)
    
    def run_experiment_3(self, test_on_sample=True):
        if not hasattr(self.indexes, "keywords_expanded_index"):
            raise RuntimeError("To run this experiment keywords_expanded_index must be loaded, try load_keywords_expanded_index()")
        
        bm_25 = pt.terrier.Retriever(self.indexes.keywords_expanded_index, wmodel="BM25")
        rm3_pipe_bm25 = bm_25 >> pt.rewrite.RM3(self.indexes.keywords_expanded_index) >> bm_25

        if test_on_sample:
            if not hasattr(self.collection, "queries_sample"):
                raise RuntimeError("No sampled queries available. Call sample_queries() first.")

            queries_to_use = self.collection.queries_sample
            print(f"Running Experiment 3 on sampled queries ({len(queries_to_use)} queries).")
            names_a = [
                f"experiment3A_bm25_sample_{len(queries_to_use)}_queries",
                f"experiment3A_bm25_rm3_sample_{len(queries_to_use)}_queries"
            ]
            names_b = [
                f"experiment3B_bm25_sample_{len(queries_to_use)}_queries",
                f"experiment3B_bm25_rm3_sample_{len(queries_to_use)}_queries"
            ]
        else:
            queries_to_use = self.collection.queries
            print(f"Running Experiment 3 on full query set ({len(queries_to_use)} queries).")
            names_a = [
                "experiment3A_bm25",
                "experiment3A_bm25_rm3"
            ]
            names_b = [
                "experiment3B_bm25",
                "experiment3B_bm25_rm3"
            ]

        save_dir = self.results_folder / "experiment_3"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        experiment3_results_a = pt.Experiment(
            [bm_25, rm3_pipe_bm25],
            queries_to_use,
            self.collection.qrels,
            EVAL_METRICS,
            verbose=True,
            names=names_a,
            save_dir=save_dir,
            save_mode="reuse",
            save_format="trec"
            )
        print("Original queries\n", experiment3_results_a)
        
        expanded_queries = self.thesaurus_query_expansion(queries_to_use)
        experiment3_results_b = pt.Experiment(
            [bm_25, rm3_pipe_bm25],
            expanded_queries,
            self.collection.qrels,
            EVAL_METRICS,
            verbose=True,
            names=names_b,
            save_dir=save_dir,
            save_mode="reuse",
            save_format="trec"
            )
        print("Expanded queries\n", experiment3_results_b)

    def run_experiment_4(self, test_on_sample=True):
        if not hasattr(self.indexes, "two_fields_index"):
            raise RuntimeError("To run this experiment two_fields_index must be loaded, try load_two_fields_index()")

        # Matching only with keywords
        bm25f_keywords = pt.terrier.Retriever(self.indexes.two_fields_index, wmodel="BM25F", controls = {'w.0' : 0, 'w.1' : 1})
        # Matching only with text
        bm25f_text = pt.terrier.Retriever(self.indexes.two_fields_index, wmodel="BM25F", controls = {'w.0' : 1, 'w.1' : 0})
        # Combination of the two fileds
        bm25f_combination = pt.terrier.Retriever(self.indexes.two_fields_index, wmodel="BM25F", controls = {'w.0' : 0.7, 'w.1' : 0.3})

        if test_on_sample:
            if not hasattr(self.collection, "queries_sample"):
                raise RuntimeError("No sampled queries available. Call sample_queries() first.")
            
            queries_to_use = self.collection.queries_sample
            print(f"Running Experiment 4 on sampled queries ({len(queries_to_use)} queries).")
            names_kw = [f"experiment4_bm25f_keywords_sample_{len(queries_to_use)}_queries"]
            names_txt = [f"experiment4_bm25f_text_sample_{len(queries_to_use)}_queries"]
            names_comb = [f"experiment4_bm25f_combination_sample_{len(queries_to_use)}_queries"]
        else:
            queries_to_use = self.collection.queries
            print(f"Running Experiment 4 on full query set ({len(queries_to_use)} queries).")
            names_kw = ["experiment4_bm25f_keywords"]
            names_txt = ["experiment4_bm25f_text"]
            names_comb = ["experiment4_bm25f_combination"]

        save_dir = self.results_folder / "experiment_4"
        save_dir.mkdir(parents=True, exist_ok=True)

        experiment4_results_kw = pt.Experiment(
            [bm25f_keywords],
            queries_to_use,
            self.collection.qrels,
            EVAL_METRICS,
            verbose=True,
            names=names_kw,
            save_dir=save_dir,
            save_mode="reuse",
            save_format="trec"
        )
        print("BM25F on keywords\n", experiment4_results_kw)

        experiment4_results_txt = pt.Experiment(
            [bm25f_text],
            queries_to_use,
            self.collection.qrels,
            EVAL_METRICS,
            verbose=True,
            names=names_txt,
            save_dir=save_dir,
            save_mode="reuse",
            save_format="trec"
        )
        print("BM25F on text\n", experiment4_results_txt)

        experiment4_results_comb = pt.Experiment(
            [bm25f_combination],
            queries_to_use,
            self.collection.qrels,
            EVAL_METRICS,
            verbose=True,
            names=names_comb,
            save_dir=save_dir,
            save_mode="reuse",
            save_format="trec"
        )
        print("BM25F on combination\n", experiment4_results_comb)

    def run_experiment_5(self, test_on_sample=True):
        if not(hasattr(self.indexes, "basic_index") and hasattr(self.indexes, "dense_index")):
            raise RuntimeError("To run this experiment basic_index and dense_index must be loaded, try load_basic_index() and load_dense_index")
        
        bm25 = pt.terrier.Retriever(self.indexes.basic_index, wmodel="BM25")
        model = RetroMAE.msmarco_distill()

        retrieval_pipe_biencoder = model >> self.indexes.dense_index.retriever()
        scorer_pipe_biencoder = model >> self.indexes.dense_index.scorer()
        retrieval_pipe_bm25_biencoder = (bm25 % 1000) >> scorer_pipe_biencoder

        if test_on_sample:
            if not hasattr(self.collection, "queries_sample"):
                raise RuntimeError("No sampled queries available. Call sample_queries() first.")
            
            queries_to_use = self.collection.queries_sample
            print(f"Running Experiment 5 on sampled queries ({len(queries_to_use)} queries).")
            names = [
                f"experiment5_biencoder_sample_{len(queries_to_use)}_queries",
                f"experiment5_bm25_biencoder_sample_{len(queries_to_use)}_queries"
            ]
        else:
            queries_to_use = self.collection.queries
            print(f"Running Experiment 5 on full query set ({len(queries_to_use)} queries).")
            names = [
                "experiment5_biencoder",
                "experiment5_bm25_biencoder"
            ]
        
        save_dir = self.results_folder / "experiment_5"
        save_dir.mkdir(parents=True, exist_ok=True)

        experiment5_results = pt.Experiment(
            [retrieval_pipe_biencoder, retrieval_pipe_bm25_biencoder],
            queries_to_use,
            self.collection.qrels,
            EVAL_METRICS,
            verbose=True,
            names=names,
            save_dir=save_dir,
            save_mode="reuse",
            save_format="trec"
        )           
        print(experiment5_results)
    
    def run_experiment_6(self, test_on_sample=True):
        if not(hasattr(self.indexes, "basic_index")):
            raise RuntimeError("To run this experiment basic_index must be loaded, try load_basic_index()")
        
        bm25 = pt.terrier.Retriever(self.indexes.basic_index, wmodel="BM25")
        monoT5 = MonoT5ReRanker(batch_size = 16)

        mono_pipe = (bm25 % 100) >> (pt.text.get_text(self.indexes.basic_index, "text")) >> monoT5

        if test_on_sample:
            if not hasattr(self.collection, "queries_sample"):
                raise RuntimeError("No sampled queries available. Call sample_queries() first.")
            
            queries_to_use = self.collection.queries_sample
            print(f"Running Experiment 6 on sampled queries ({len(queries_to_use)} queries).")
            names = [
                f"experiment6_bm25_monoT5_sample_{len(queries_to_use)}_queries"
            ]
        else:
            queries_to_use = self.collection.queries
            print(f"Running Experiment 6 on full query set ({len(queries_to_use)} queries).")
            names = [
                "experiment6_bm25_monoT5"
            ]
        
        save_dir = self.results_folder / "experiment_6"
        save_dir.mkdir(parents=True, exist_ok=True)

        experiment6_results = pt.Experiment(
            [mono_pipe],
            queries_to_use,
            self.collection.qrels,
            EVAL_METRICS,
            verbose=True,
            names=names,
            save_dir=save_dir,
            save_mode="reuse",
            save_format="trec"
        )
        print(experiment6_results)