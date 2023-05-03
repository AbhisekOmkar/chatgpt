#!/bin/sh
python3 -m venv pdf_to_qna
source pdf_to_qna/bin/activate
pip install -r pdf_to_qna_requirements.txt
deactivate