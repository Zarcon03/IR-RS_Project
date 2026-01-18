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
        
        self._create_results_folder()

    def _create_results_folder(self):
        self.results_folder.mkdir(parents=True, exist_ok=True)
        
    def _thesaurus_query_expansion(self, queries: pd.DataFrame) -> pd.DataFrame:
        expanded_queries = queries.copy()

        expanded_queries["query_0"] = expanded_queries["query"]

        tqdm.pandas(desc="Expanding queries with thesaurus")
        expanded_queries["query"] = expanded_queries["query"].progress_apply(
            lambda q: q + " " + " ".join(
                thesaurus_based_expansion(q, keywords_extractor(q))
            )
        )
        return expanded_queries

    def _check_indexes(self, *index_names: str):
        for index_name in index_names:
            if not hasattr(self.indexes, index_name):
                raise RuntimeError(f"Index '{index_name}' is not loaded in BenchmarkIndex. Please load it before running the experiment.")
    
    def _get_queries(self, experiment_id: int, test_on_sample: bool) -> tuple[pd.DataFrame, str]:
        if test_on_sample:
            if not hasattr(self.collection, "queries_sample"):
                raise RuntimeError("No sampled queries available. Call sample_queries() first.")
            queries_to_use = self.collection.queries_sample
            print(f"Running Experiment {experiment_id} on sampled queries ({len(queries_to_use)} queries).")
            return queries_to_use, f"_sample_{len(queries_to_use)}_queries"
        else:
            queries_to_use = self.collection.queries
            print(f"Running Experiment {experiment_id} on full query set ({len(queries_to_use)} queries).")
            return queries_to_use, None
        
    def _make_names(self, experiment_id: int | str, pipeline_names: list[str], suffix: str | None) -> list[str]:
        if suffix:
            return [f"experiment{experiment_id}_{name}{suffix}" for name in pipeline_names]
        else:
            return [f"experiment{experiment_id}_{name}" for name in pipeline_names]
        
    def _create_results_subfolder(self, experiment_id: int) -> Path:
        save_dir = self.results_folder / f"experiment_{experiment_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir


    def run_experiment_1(self, test_on_sample=True):
        self._check_indexes("basic_index")
        
        bm25 = pt.terrier.Retriever(self.indexes.basic_index, wmodel="BM25")
        rm3_pipe_bm25 = bm25 >> pt.rewrite.RM3(self.indexes.basic_index) >> bm25

        tfidf = pt.terrier.Retriever(self.indexes.basic_index, wmodel="TF_IDF")
        rm3_pipe_tfidf = tfidf >> pt.rewrite.RM3(self.indexes.basic_index) >> tfidf

        queries, suffix = self._get_queries(1, test_on_sample)
        names = self._make_names(1, ["bm25", "bm25_rm3", "tfidf", "tfidf_rm3"], suffix)
        
        save_dir = self._create_results_subfolder(1)

        experiment1_results = pt.Experiment(
            [bm25, rm3_pipe_bm25, tfidf, rm3_pipe_tfidf],
            queries,
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
        self._check_indexes("basic_index")
        
        bm25 = pt.terrier.Retriever(self.indexes.basic_index, wmodel="BM25")
        rm3_pipe_bm25 = bm25 >> pt.rewrite.RM3(self.indexes.basic_index) >> bm25

        tfidf = pt.terrier.Retriever(self.indexes.basic_index, wmodel="TF_IDF")
        rm3_pipe_tfidf = tfidf >> pt.rewrite.RM3(self.indexes.basic_index) >> tfidf

        queries, suffix = self._get_queries(2, test_on_sample)
        names = self._make_names(2, ["bm25", "bm25_rm3", "tfidf", "tfidf_rm3"], suffix)
        
        save_dir = self._create_results_subfolder(2)
            
        expanded_queries = self._thesaurus_query_expansion(queries)
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
        self._check_indexes("keywords_expanded_index")
        
        bm_25 = pt.terrier.Retriever(self.indexes.keywords_expanded_index, wmodel="BM25")
        rm3_pipe_bm25 = bm_25 >> pt.rewrite.RM3(self.indexes.keywords_expanded_index) >> bm_25

        queries, suffix = self._get_queries(3, test_on_sample)
        names_a = self._make_names("3A", ["bm25", "bm25_rm3"], suffix)
        names_b = self._make_names("3B", ["bm25", "bm25_rm3"], suffix)

        save_dir = self._create_results_subfolder(3)
        
        experiment3_results_a = pt.Experiment(
            [bm_25, rm3_pipe_bm25],
            queries,
            self.collection.qrels,
            EVAL_METRICS,
            verbose=True,
            names=names_a,
            save_dir=save_dir,
            save_mode="reuse",
            save_format="trec"
            )
        print("Original queries\n", experiment3_results_a)
        
        expanded_queries = self._thesaurus_query_expansion(queries)
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
        self._check_indexes("two_fields_index")

        # Matching only with keywords
        bm25f_keywords = pt.terrier.Retriever(self.indexes.two_fields_index, wmodel="BM25F", controls = {'w.0' : 0, 'w.1' : 1})
        # Matching only with text
        bm25f_text = pt.terrier.Retriever(self.indexes.two_fields_index, wmodel="BM25F", controls = {'w.0' : 1, 'w.1' : 0})
        # Combination of the two fileds
        bm25f_combination = pt.terrier.Retriever(self.indexes.two_fields_index, wmodel="BM25F", controls = {'w.0' : 0.7, 'w.1' : 0.3})

        queries, suffix = self._get_queries(4, test_on_sample)
        names = self._make_names(4, ["bm25f_keywords", "bm25f_text", "bm25f_combination"], suffix)

        save_dir = self._create_results_subfolder(4)

        experiment4_results = pt.Experiment(
            [bm25f_keywords, bm25f_text, bm25f_combination],
            queries,
            self.collection.qrels,
            EVAL_METRICS,
            verbose=True,
            names=names,
            save_dir=save_dir,
            save_mode="reuse",
            save_format="trec"
        )
        print(experiment4_results)

    def run_experiment_5(self, test_on_sample=True):
        self._check_indexes("basic_index", "dense_index")
        
        bm25 = pt.terrier.Retriever(self.indexes.basic_index, wmodel="BM25")
        model = RetroMAE.msmarco_distill()

        retrieval_pipe_biencoder = model >> self.indexes.dense_index.retriever()
        scorer_pipe_biencoder = model >> self.indexes.dense_index.scorer()
        retrieval_pipe_bm25_biencoder = (bm25 % 1000) >> scorer_pipe_biencoder

        queries, suffix = self._get_queries(5, test_on_sample)
        names = self._make_names(5, ["biencoder", "bm25_biencoder"], suffix)
        
        save_dir = self._create_results_subfolder(5)

        experiment5_results = pt.Experiment(
            [retrieval_pipe_biencoder, retrieval_pipe_bm25_biencoder],
            queries,
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
        self._check_indexes("basic_index")
        
        bm25 = pt.terrier.Retriever(self.indexes.basic_index, wmodel="BM25")
        monoT5 = MonoT5ReRanker(batch_size = 16)

        mono_pipe = (bm25 % 100) >> (pt.text.get_text(self.indexes.basic_index, "text")) >> monoT5

        queries, suffix = self._get_queries(6, test_on_sample)
        names = self._make_names(6, ["bm25_monoT5"], suffix)
        
        save_dir = self._create_results_subfolder(6)

        experiment6_results = pt.Experiment(
            [mono_pipe],
            queries,
            self.collection.qrels,
            EVAL_METRICS,
            verbose=True,
            names=names,
            save_dir=save_dir,
            save_mode="reuse",
            save_format="trec"
        )
        print(experiment6_results)