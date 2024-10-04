# NewPOTATO: Human-in-the-Loop Open Information Extraction

## Overview
NewPotato is an MVP project aimed at implementing Open Information Extraction (OIE) principles with a Human-in-the-Loop (HITL) approach. The project consists of a backend API built with FastAPI, a frontend built with Streamlit, and a core package named `newpotato` that contains the core logic of the project.

## Experimental OIE

To try the experimental OIE, first start the parser with
`python api/ud_parser.py -l en -t`

Then load, parse, and save an LSOIE sample (this will only take long the first time, the UD parser
caches its results):
`python newpotato/datasets/lsoie.py -i sample_data/lsoie_5k.tsv -s sample_data/lsoie_5k.hitl`

Finally, learn patterns, evaluate them, and write the predicted triplets to a file:
`python newpotato/evaluate/eval_extractor.py -l sample_data/lsoie_5k.hitl -e sample_data/lsoie_5k_events.tsv -s 10`

Control verbosity of the above command with `-v` and `-d`.

## Features

- Text parsing with graphbrain
- HITL process for refining triplet extraction
- Knowledge graph visualization
- RESTful API for interaction
- Streamlit frontend for user interaction

## Installation

### Pre-requisites
- Python 3.11
- Docker

### Development
#### Backend
The backend functionalities are implemented as a REST API built with FastAPI. The code is located in the `api/` directory.

#### Frontend
The frontend is built using Streamlit and is located in the `frontend/` directory.

#### Core Package
The core functionalities are in the `newpotato` package.

#### Running Locally
To run the project locally for development:

1. Install the dependencies:
    ```bash
    pip install -e .
    ```
2. Start the FastAPI server:
    ```bash
    uvicorn api.main:app --reload
    ```
3. Start the Streamlit app:
    ```bash
    streamlit run frontend/app.py
    ```

#### Running with Docker
The devlopment environment can be also used from .devcontainer in VSCode. It can be found under the .devcontainer folder.

### Production
The production environment is built using Docker Compose. To run the project in production:
```bash
docker-compose -f deploy/docker-compose.yml up
```
