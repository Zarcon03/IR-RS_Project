import streamlit as st
from collection import BenchmarkCollection
from indexes import BenchmarkIndex
from llm import llm

@st.cache_resource # Cache the collection and indexes to avoid reloading on every interaction
def load_collection_and_indexes():
    collection = BenchmarkCollection()
    indexes = BenchmarkIndex(collection)
    indexes.load_basic_index()
    return collection, indexes


@st.cache_resource # Cache the RAG model
def load_rag(_collection, _indexes):
    return llm(collection=_collection, indexes=_indexes)

collection, indexes = load_collection_and_indexes() # Load collection and indexes this has to be done on every interaction
rag = load_rag(collection, indexes) # Load RAG model this also has to be done on every interaction

st.title("Retrieval-Augmented Generation Answering System")

# query = st.text_input("Enter your query:")
# if st.button("Get Answer"):
#     with st.spinner("Generating answer..."):
#         answer = rag.answer_query(query)
#     st.write("### Answer:")
#     st.write(answer)

with st.form("query_form"):
    query = st.text_input(
        "Your question",
        placeholder="Who is the author of the book, Horrors of Slavery?" # Answer: WILLIAM RAY
    )
    submitted = st.form_submit_button("Get Answer")


if submitted and query:
    with st.spinner("Generating answer..."):
        answer = rag.answer_query(query)
    st.write("### Answer:")
    st.write(answer)
