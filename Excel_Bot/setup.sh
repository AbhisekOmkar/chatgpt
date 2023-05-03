#!/bin/sh
python3 -m venv excel_bot_env
source excel_bot_env/bin/activate
pip install -r excel_bot_requirements.txt
deactivate