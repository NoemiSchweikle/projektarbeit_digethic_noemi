# projektarbeit_digethic_noemi

## Intro

This repository contains the source code and data for the final project of the EN ISO / IEC 17024- certification of Noemi Schweikle.  

## Setup

### Linux and Mac Users

- run the setup script `./setup.sh` or `sh setup.sh`

### Windows Users

- run the setup script `.\setup.ps1`
- if running the script does not work due to access rights, try following command in your terminal: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

## Development

- Mac/Linux: activate python environment: `source .venv/bin/activate`
- Windows: activate python environment: `.\.venv\Scripts\Activate.ps1`
- run python script: `python <filename.py> `, e.g. `python train.py`
- install new dependency: `pip install sklearn`
- save current installed dependencies back to requirements.txt: `pip freeze > requirements.txt`
- to start Jupyter lab run `jupyter lab --ip=127.0.0.1 --port=8888`