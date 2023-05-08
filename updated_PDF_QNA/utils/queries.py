from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI

class Query:
    def __init__(self):
        pass

    def initialize_chain(self):
        lang_chain = load_qa_chain(OpenAI(), chain_type="stuff")
        return lang_chain

    def answer_to_query(self, chain, docs, query):
        return chain.run(input_documents=docs, question=query)
    
    def conversational_retrieval_query(self, vectorstore, memory):
        qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), memory=memory)
        return qa