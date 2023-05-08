from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma

class Embeddings:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

    def create_a_memory_store(self):
        """
        Create a memory store for conversation
        """
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return memory

    def text_embeddings(self, texts):
        """
        Text embeddings
        """
        docsearch = FAISS.from_texts(texts, self.embeddings)
        return docsearch
    
    def similary_search_for_query(self, query, docsearch):
        """
        Similarity search
        """
        docs = docsearch.similarity_search(query)
        return docs
    
    def create_vector_store(self, pages):
        """
        Create a vector store
        """
        vectorstore = Chroma.from_documents(pages, self.embeddings)
        return vectorstore
    