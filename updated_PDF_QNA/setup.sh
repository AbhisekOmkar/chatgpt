#!/bin/sh
python3 -m venv updated_pdf_qna
source updated_pdf_qna/bin/activate
pip install -r updated_pdf_qna_requirements.txt
deactivate