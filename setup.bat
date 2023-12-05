@echo off

rem Set the desired Python version
set PYTHON_VERSION=3.9.16

rem Create a virtual environment
python -m venv .venv2

rem Activate the virtual environment
call .venv\Scripts\activate

rem Update pip to the latest version
python -m pip install --upgrade pip

rem Install requirements
pip install -r requirements.txt

echo Virtual environment setup complete.
echo To activate the virtual environment, run: .\.venv\Scripts\activate
