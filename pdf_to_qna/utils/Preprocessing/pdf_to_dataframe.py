import pandas as pd
import PyPDF2
import os
import errno

def read_pdf_to_dataframe(pdf_file):
    if not os.path.exists(pdf_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), pdf_file)
    else:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        df_cook = pd.DataFrame(columns=['page_number', 'content'])
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            df_temp = pd.DataFrame([{'page_number': page_num + 1, 'content': text}])
            df_cook = pd.concat([df_cook, df_temp], ignore_index=True)
        return df_cook