import openai
import tiktoken
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from utils.Preprocessing.constant import EMBEDDING_MODEL, COMPLETIONS_MODEL, MODEL_NAME

class Embeddings:
    def __init__(self):
        """
        Define all the constants
        """
        self.model_name = MODEL_NAME
        self.completions_model = COMPLETIONS_MODEL
        self.embedding_model = EMBEDDING_MODEL

    def get_embedding(self, text: str):
        """
        Create embeddings
        """
        result = openai.Embedding.create(
            model= self.embedding_model,
            input=text
        )
        return result["data"][0]["embedding"]

    def compute_doc_embeddings(self, df: pd.DataFrame):
        """
        Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
        Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
        """
        return {
            idx: self.get_embedding(r.content) for idx, r in df.iterrows()
        }
    
    def load_embeddings(self, fname: str):
        """
        Read the document embeddings and their keys from a CSV.
        fname is the path to a CSV with exactly these named columns: 
            "title", "heading", "0", "1", ... up to the length of the embedding vectors.
        """
        df = pd.read_csv(fname, header=0)
        max_dim = max([int(c) for c in df.columns if c != "title" and c != "heading"])
        return {
            (r.title, r.heading): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
        }

    def get_hf_embeddings(self, text: str, model):
        sentence_embeddings = model.encode(text)
        sentence_embeddings = sentence_embeddings.reshape(1, -1)
        sentence_embeddings = normalize(sentence_embeddings)
        return sentence_embeddings[0]

    def compute_doc_embeddings_hf(self, df: pd.DataFrame, model):
        return {
            idx: self.get_hf_embeddings(r.content, model) for idx, r in df.iterrows()
        }
    
    def vector_similarity(self, x: list, y: list):
        """
        Returns the similarity between two vectors.
        Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
        """
        return np.dot(np.array(x), np.array(y))


    def order_document_sections_by_query_similarity(self, query: str, contexts: dict, sentence_tf_model):
        """
        Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
        to find the most relevant sections. 
        Return the list of document sections, sorted by relevance in descending order.
        """
        query_embedding = self.get_hf_embeddings(query, sentence_tf_model)

        document_similarities = sorted([
            (self.vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
        ], reverse=True)

        return document_similarities