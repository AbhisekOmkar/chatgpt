from utils.Preprocessing.pdf_to_dataframe import read_pdf_to_dataframe
from utils.Preprocessing.embeddings import Embeddings
from utils.queries import Query
from sentence_transformers import SentenceTransformer 
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
embedding = Embeddings()
query = Query()
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
sentence_tf_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

if __name__=="__main__":
    print('wait Bot Initializing...')
    #  take path input from terminal
    path_to_pdf = str(sys.argv[1])
    df_cook = read_pdf_to_dataframe(path_to_pdf)
    cook_embeddings_hf = embedding.compute_doc_embeddings_hf(df_cook, sentence_tf_model)
    print("Please go ahead bot is ready, type exit anytime to end the conversation")

    while 1:
        # take question as input
        questions = input('Please enter you next question or type exit to end the conversation\nQuestion: ')
        if questions == "exit":
            break
        res = query.answer_query_with_context(questions, df_cook, cook_embeddings_hf, sentence_tf_model, show_prompt = True)
        print(f'Answer: {res}')