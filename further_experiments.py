import json
from operator import index
import os
import pandas as pd
import pyterrier as pt
from eval_metrics import EVAL_METRICS
from functions import keywords_extractor, thesaurus_based_expansion
from tqdm import tqdm
tqdm.pandas()
from baseline_experiments import BaselineExperiments


"""
CHILD CLASS OF BaselineExperiments
+ complete_index_path ="keywords_and_text_index" attribute
+ load_complete_index(self) method
+ run_further_experiment_1(self) method
"""

class FurtherExperiments(BaselineExperiments):

    def __init__(         
        self,
        queries_path,
        qrels_path,
        base_index_path="terrier_index",
        keywords_index_path="keywords_index",
        complete_index_path ="keywords_and_text_index"

    ):
    
        # Call the parent constructor
        super().__init__(
            queries_path,
            qrels_path,
            base_index_path,
            keywords_index_path
        )
        self.complete_index_path = complete_index_path
        self.load_complete_index()

    def load_complete_index(self):
        index_abs_path = os.path.abspath(self.complete_index_path)
        self.complete_index = pt.IndexFactory.of(index_abs_path)
    

    def run_further_experiment_1(self):
        # try bo1 on same setting as baseline_3 instead of rm3
        if not self.queries_expanded:
            self.expanded_queries_small = self.thesaurus_query_expansion(self.queries_small)

        bm_25 = pt.terrier.Retriever(self.keywords_index, wmodel="BM25")
        bo1_pipe_bm25 = bm_25 >> pt.rewrite.Bo1QueryExpansion(self.keywords_index) >> bm_25

        experiment3_results_a = pt.Experiment(
            [bm_25, bo1_pipe_bm25],
            self.queries_small,
            self.qrels,
            EVAL_METRICS
            )
        print("Further experiment 1 (not expanded queries):\n", experiment3_results_a)

        experiment3_results_b = pt.Experiment(
            [bm_25, bo1_pipe_bm25],
            self.expanded_queries_small,
            self.qrels,
            EVAL_METRICS
            )
        print("Further experiment 1 results (expanded queries):\n", experiment3_results_b)    


    def run_further_experiment_2(self):
        # try bo1 on same setting as baseline_3 instead of rm3
        # but use complete_index for matching keywords, and adding text from text_field
        if not self.queries_expanded:
            self.expanded_queries_small = self.thesaurus_query_expansion(self.queries_small)
        # try bo1 on same setting as baseline_3 instead of rm3
        # but use complete_index for matching keywords, and adding text from text_field
        
        
        # first matching only with keywords
        br = pt.terrier.Retriever(
            self.complete_index,
            wmodel="BM25",
            controls = {'w.0' : 1, 'w.1' : 0}
        )

        """
        # query expansion with text only
        qe = pt.rewrite.Bo1QueryExpansion(
            self.complete_index,
            controls = {'w.0' : 1, 'w.1' : 0}
        )"""

        qe = pt.terrier.Retriever(
            self.complete_index,
            wmodel="DPH", 
            controls =
                {"qemodel" : "Bo1", 
                "qe" : "on", 
                'w.0' : 1, 'w.1' : 0}
        )

        br_final = pt.terrier.Retriever(
            self.complete_index,
            wmodel="BM25",
            controls = {'w.0' : 1, 'w.1' : 0}
        )

        #NOTE THIS SHI' SOMEHOW PERFORMS IDENTICAL TO BM25....
        bo1_field_pipe = br >> qe >> br_final
        
        bm_25 = pt.terrier.Retriever(
            self.complete_index, 
            wmodel="BM25",
            controls = {'w.0' : 0.6, 'w.1' : 0.4}
        )

        further_experiment_2_results = pt.Experiment(
            [bm_25, bo1_field_pipe],
            self.expanded_queries_small,
            self.qrels,
            EVAL_METRICS
        )
        
        print("Further experiment 2 results (expanded queries):\n", further_experiment_2_results)   




fe = FurtherExperiments("test_queries.json", "test_qrels.json")
#fe.run_further_experiment_1()
fe.run_further_experiment_2()



"""
    properties={
        "index.document.class": "org.terrier.structures.FSAFieldDocumentIndex"
    }
"""