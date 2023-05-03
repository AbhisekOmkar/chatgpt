#!/bin/sh
python3 -m venv excel_bot_env
source excel_bot_env/bin/activate
pip install -r requirements.txt
deactivate