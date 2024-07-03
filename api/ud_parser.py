import logging
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tuw_nlp.grammar.text_to_ud import TextToUD

logging.basicConfig(
    format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
    level=logging.INFO,
    force=True,
)


LANG = "en"
NLP_CACHE = "nlp_cache"

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    models['parser'] = TextToUD(LANG, NLP_CACHE)
    yield
    models['parser'].nlp.save_cache_if_changed()


app = FastAPI(lifespan=lifespan)
graph_type = "UD"


class TextToParse(BaseModel):
    """Model for text to be parsed.

    Args:
        text (str): Text to be parsed.
    """

    text: str


class ParserParams(BaseModel):
    """Model for text to be parsed.

    Args:
        params: dictionary of parameters
    """

    params: dict


@app.get("/get_params")
def get_params() -> Dict[str, Any]:
    params = models['parser'].get_params()
    return {"status": "ok", "params": params}


@app.post("/check_params")
def check_params(params: ParserParams) -> Dict[str, Any]:
    parser_params = ParserParams(params=models['parser'].get_params())
    if params != parser_params:
        logging.error(f"parser params mismatch: {params=}, {parser_params=}")
        raise HTTPException(
            status_code=400,
            detail=f"parser params mismatch: {params=}, {parser_params=}",
        )
    return {"status": "ok"}


@app.post("/parse")
def parse(text_to_parse: TextToParse) -> Dict[str, Any]:
    """Classifies the text based on stored rules.

    Args:
        text (str): Text to be parsed.

    Returns:
        Dict[str, Any]: Dictionary containing parsing results
    """

    logging.info(f"Parsing text: {text_to_parse.text}")
    try:
        json_graphs = []
        for graph in models['parser'](text_to_parse.text):
            json_graph = graph.to_json()
            json_graphs.append(json_graph)

        return {"status": "ok", "graphs": json_graphs, "graph_type": graph_type}

    except Exception as e:
        logging.error(f"Parsing error: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
