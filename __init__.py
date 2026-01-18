from collection import BenchmarkCollection
from indexes import BenchmarkIndex
from experiments import BenchmarkExperiments
from llm import llm

if __name__ == "__main__":
    collection = BenchmarkCollection()
    collection.load_documents()
    collection.load_queries()
    collection.load_qrels()
    collection.sample_queries(n=1000)

    indexes = BenchmarkIndex(collection)
    #indexes.create_basic_index()
    #indexes.create_keywords_expanded_index()
    #indexes.create_two_fields_index()
    #indexes.create_dense_index()
    indexes.load_basic_index()
    indexes.load_keywords_expanded_index()
    indexes.load_two_fields_index()
    indexes.load_dense_index()


    experiments = BenchmarkExperiments(collection=collection, indexes=indexes)
    #experiments.run_experiment_1(test_on_sample=True)
    #experiments.run_experiment_2(test_on_sample=True)
    #experiments.run_experiment_3(test_on_sample=True)
    #experiments.run_experiment_4(test_on_sample=True)
    #experiments.run_experiment_5(test_on_sample=True)
    #experiments.run_experiment_6(test_on_sample=True)

    #rag = llm(collection=collection, indexes=indexes)
    #answer = rag.answer_query("When did the king of spain died?")
    #print(answer)
