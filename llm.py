import pyterrier as pt
from openai import OpenAI
from ollama import Client as OllamaClient
from ollama import chat as ollama_chat
from pyterrier_t5 import MonoT5ReRanker
from transformers import T5Tokenizer # useful chat on tokenization, finetuning and Mono T5: https://chatgpt.com/share/696468e8-c33c-8009-88ba-5843b8a46d21
from collection import BenchmarkCollection
from indexes import BenchmarkIndex
from functions import TokenizerWrapper

class llm():
    def __init__(self, collection:BenchmarkCollection, indexes:BenchmarkIndex):
        self.collection = collection
        self.indexes = indexes

        self.create_tokenizer()

    def create_tokenizer(self):
        base_tok = T5Tokenizer.from_pretrained("t5-base")
        self.tokenizer = TokenizerWrapper(base_tok)

    def retriever(self, query:str, document_context_number=3):
        if not hasattr(self.indexes, "basic_index"):
            raise RuntimeError("The retriever uses the basic_index, make sure it is loaded: try load_basic_index()")
        
        bm25 = pt.terrier.Retriever(self.indexes.basic_index, wmodel="BM25")
        monoT5 = MonoT5ReRanker(batch_size = 16)

        mono_pipeline = (
        (bm25 % 100)                              
        >> pt.text.get_text(self.indexes.basic_index, "text")
        >> pt.text.sliding(                   
            length=256,
            stride=128,
            text_attr = "text",
            prepend_attr=None,
            tokenizer = self.tokenizer)
        >> monoT5                              
        >> pt.text.max_passage()            
        )

        results = mono_pipeline.search(query)

        context = ""
        for i in range(document_context_number):
            context += (
                f"CONTEXT NUMBER: {i+1}\n"
                f"DOCUMENT ID: {i+1}\n"
                f"{results.iloc[i]['docno']}\n\n"
                f"{results.iloc[i]['text']}\n\n"
            )

        return context
    
    def answer_openai(self, prompt: str, context: str, model: str, server: str = "local", api_key: str | None = None):
        if server == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required for online usage")
            client = OpenAI(api_key=api_key)

        elif server == "local":
            client = OpenAI(
                # Note: ollama also supports Openai endpoint
                # you can run local models through ollama
                # base_url = "http://localhost:11434/v1"
                base_url="http://localhost:1234/v1", # perfect if using LM-studio server
                api_key="not-needed"
            )

        else:
            raise ValueError("server must be 'local' or 'openai'")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. "
                        "Answer reading the context ONLY if it is relevant."
                        "IF THE TOPIC OF DISCUSSION IS NOT IN THE CONTEXT REPLY 'I DO NOT KNOW' OTHERWISE MENTION THE RELEVANT THINGS IN THE CONTEXT"
                    ),
                },
                {"role": "tool", "content": context},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_completion_tokens=256,
        )

        return response.choices[0].message.content
    
    def answer_ollama(self, prompt: str, context: str, model: str, server: str = "local", api_key: str | None = None):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer reading the context ONLY if it is relevant."
                    "IF THE TOPIC OF DISCUSSION IS NOT IN THE CONTEXT REPLY 'I DO NOT KNOW' OTHERWISE MENTION THE RELEVANT THINGS IN THE CONTEXT."
                ),
            },
            {"role": "user", "content": f"{context}\n\n{prompt}"},
        ]

        if server == "cloud":
            if not api_key:
                raise ValueError("Ollama API key is required for cloud usage")

            client = OllamaClient(
                host="https://ollama.com",
                headers={"Authorization": "Bearer " + api_key},
            )

            output = ""
            for part in client.chat(model, messages=messages, stream=True):
                output += part["message"]["content"]
            return output

        elif server == "local":
            response = ollama_chat(model=model, messages=messages)
            return response["message"]["content"]

        else:
            raise ValueError("server must be 'local' or 'cloud'")
        
    def answer_query(self, question:str):
        """
        !!! possible implementation for more flexibility
        !!! potentially compatible with online openai, requires some adjustments due to tool calling
        question = input("Question: ").strip()
        endpoint = input("Endpoint (lmstudio / ollama): ").strip().lower()
        server = input("Server (offline / online): ").strip().lower()
        model = input("Model (model name e.g. 'google/gemma-3-12b' or 'gemma3:4b'): ").strip().lower() or "google/gemma-3-12b"
        api_key = input("API key (press Enter if none): ").strip() or None
        """

        endpoint = 'ollama'
        server = 'online'
        model = 'gpt-oss:120b-cloud'
        api_key = '84fc73f900d1493f9156107a6d485d5e.I8sgRuzuKB-9VAAC-kng6e_3'

        print("generating answer with:")
        print(endpoint)
        print(server)
        print(model)

        context = self.retriever(question)

        if endpoint == "lmstudio":
            if server == "offline":
                server_mode = "local" 
            else:
                ValueError("Endpoint lmstudio only supports offline server")

            answer = self.answer_openai(
                prompt=question,
                context=context,
                model=model,
                server=server_mode,
                api_key=api_key,
            )


        elif endpoint == "ollama":
            model = "gemma3:4b" #hard coded for semplicity, to eliminate 
            server_mode = "local" if server == "offline" else "cloud"

            answer = self.answer_ollama(
                context=context,
                prompt=question,
                model=model,
                server=server_mode,
                api_key=api_key,
            )

        else:
            raise ValueError("Endpoint must be 'lmstudio' or 'ollama'")
        
        final_answer = answer + "\n\n" + context
        return final_answer