from langchain.document_loaders import PyPDFLoader
import PdfReader
import os
import errno

def load_pdf_to_pages(pdf_file):
  """
  read pdf file
  """
  if not os.path.exists(pdf_file):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), pdf_file)
  else:
    loader = PyPDFLoader(pdf_file)
    pages = loader.load_and_split()
    return pages

def read_pdf(pdf_file):
  """
  read pdf file
  """
  if not os.path.exists(pdf_file):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), pdf_file)
  else:
    pdf_reader = PdfReader(pdf_file)
    return pdf_reader
   
def convert_pdf_text_to_raw_data(reader):
  """
  read data from the file and put them into a variable called raw_text
  """
  if reader:
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text

def convert_raw_data_to_chunks(raw_text):
  """
  split the text that we read into smaller chunks so that during information 
  retreival we don't hit the token size limits.
  """
  text_splitter = CharacterTextSplitter(        
      separator = "\n",
      chunk_size = 1000,
      chunk_overlap  = 200,
      length_function = len,
  )
  texts = text_splitter.split_text(raw_text)
  return texts

def convert_pdf_text_to_chunks(pdf_file):
  """
  This function calls convert_pdf_text_to_raw_data() and convert_raw_data_to_chunks()
  """
  if pdf_file:
    reader = read_pdf(pdf_file)
    raw_data = convert_pdf_text_to_raw_data(reader)
    chunk_data = convert_raw_data_to_chunks(raw_data)
    return chunk_data
