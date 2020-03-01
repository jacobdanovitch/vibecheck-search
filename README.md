# VibeCheck Search

Server for VibeCheck project. 

## Installation

```shell
pip install -r requirements.txt
pip install -r server/dev_requirements.txt
pip install -e server

source .env
start_server
```

## API

```
:8000/predict [POST]
  - {'query': ..., 'track_id': ..., 'n': ...}
```
