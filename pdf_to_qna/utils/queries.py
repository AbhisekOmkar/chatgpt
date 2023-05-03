import tiktoken
import pandas as pd
import openai
import os
from utils.Preprocessing.constant import ENCODING, SEPARATOR, MAX_SECTION_LEN
from utils.Preprocessing.embeddings import Embeddings

openai.api_key = os.getenv("OPENAI_API_KEY")

class Query(Embeddings):

    def __init__(self):
        super().__init__()
        self.encoding = tiktoken.get_encoding(ENCODING)
        self.separator_len = len(self.encoding.encode(SEPARATOR))
        self.max_section_length = MAX_SECTION_LEN
        self.completion_api_params = {
            "temperature": 0.0,
            "max_tokens": 1000,
            "model": self.completions_model,
        }

    def construct_prompt(self, question: str, context_embeddings: dict, df: pd.DataFrame, sentence_tf_model):
        """
        Fetch relevant 
        """
        most_relevant_document_sections = Embeddings.order_document_sections_by_query_similarity(self, question, context_embeddings, sentence_tf_model)
        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []
        for _, section_index in most_relevant_document_sections:
            # Add contexts until we run out of space.        
            document_section = df.loc[section_index]
            document_tokens = len(self.encoding.encode(document_section.content))
            chosen_sections_len += document_tokens + self.separator_len
            if chosen_sections_len > self.max_section_length:
                break
            chosen_sections.append(SEPARATOR + document_section.content.replace("\n", " "))
            chosen_sections_indexes.append(str(section_index))
        header = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say "Sorry, I don't know about this."\n\nContext:\n"""
        return header + "".join(chosen_sections) + "\n\n Q: " + question + "\n A:"

    def answer_query_with_context(self, query: str, df: pd.DataFrame, document_embeddings: dict, sentence_tf_model, show_prompt: bool = False):
        prompt = self.construct_prompt(query, document_embeddings,df, sentence_tf_model)
        response = openai.Completion.create(
            prompt=prompt,
            **self.completion_api_params
        )
        return response["choices"][0]["text"].strip(" \n")