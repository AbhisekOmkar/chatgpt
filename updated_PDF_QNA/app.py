from utils.Preprocessing.read_pdf_to_chunks import load_pdf_to_pages
from utils.Preprocessing.embeddings import Embeddings
from utils.queries import Query
import sys
import os
import openai

print('wait Bot Initializing...')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENAI_API_KEY"] = 'YOUR-SECRET-KEY'
embeddings = Embeddings()
query = Query()

if __name__=="__main__":
    #  take path input from terminal
    path_to_pdf = str(sys.argv[1])
    # Load pdf and return the splitted and chunked data
    documents = load_pdf_to_pages(path_to_pdf)
    vectorstore = embeddings.create_vector_store(documents)
    memory = embeddings.create_a_memory_store()
    qa = query.conversational_retrieval_query(vectorstore, memory)
    print("Please go ahead bot is ready, type exit anytime to end the conversation")
    while 1:
        # take question as input
        questions = input('Please enter you next question or type exit to end the conversation\nQuestion: ')
        if questions == "exit":
            break
        result = qa({"question": questions})
        print(f'Answer: {result["answer"]}')