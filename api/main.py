import logging
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from newpotato.hitl import HITLManager, TextParser

app = FastAPI()
hitl_manager = HITLManager()
parser = TextParser()

logging.basicConfig(level=logging.INFO)


class TextToParse(BaseModel):
    text: str


class Annotation(BaseModel):
    text: str
    pred: int
    args: List[int] = Field(..., min_items=2)


@app.post("/parse")
def parse_text(text_to_parse: TextToParse):
    try:
        parsed_graphs = parser.parse(text_to_parse.text)

        for graph in parsed_graphs:
            hitl_manager.store_parsed_graphs(graph["text"], [graph])
    except Exception as e:
        logging.error(f"Error parsing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok"}


@app.post("/annotate")
def annotate_text(annotation: Annotation):
    try:
        hitl_manager.store_triplet(annotation.text, annotation.pred, annotation.args)
    except Exception as e:
        logging.error(f"Error annotating text: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok"}


@app.get("/tokens/{text}")
def get_tokens(text: str):
    if not hitl_manager.is_parsed(text):
        raise HTTPException(status_code=400, detail="Text not parsed")

    tokens = hitl_manager.get_tokens(text)
    indexed_tokens = [
        {"index": i, "token": str(token)} for i, token in enumerate(tokens)
    ]

    return {"tokens": indexed_tokens}


@app.get("/triplets")
def get_triplets():
    return {"triplets": hitl_manager.get_triplets()}


@app.get("/sentences")
def get_sentences():
    try:
        parsed_graphs = hitl_manager.parsed_graphs

        sentences = [sen for sen in parsed_graphs.keys() if sen != "latest"]

        return {"sentences": sentences}
    except Exception as e:
        logging.error(f"Error getting sentences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rules")
def get_rules():
    hitl_manager.annotate_graphs_with_triplets()
    hitl_manager.extract_rules()
    return {"rules": hitl_manager.get_rules()}


@app.get("/annotated_graphs")
def get_annotated_graphs():
    return {"annotated_graphs": hitl_manager.get_annotated_graphs()}


@app.post("/classify_sentence")
def classify_sentence(text_to_classify: TextToParse):
    try:
        graphs = parser.parse(text_to_classify.text)
        main_graph = graphs[0]["main_edge"]
        matches = hitl_manager.classify(main_graph)
    except Exception as e:
        logging.error(f"Error classifying sentence: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not matches:
        return {"status": "No matches found"}
    else:
        return {"status": "ok", "matches": matches}
