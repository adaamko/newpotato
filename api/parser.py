import logging
import traceback
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from newpotato.parser import TextParser

logging.basicConfig(
    format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
    level=logging.INFO,
)

app = FastAPI()
parser = TextParser()


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
    params = parser.get_params()
    return {"status": "ok", "params": params}


@app.post("/check_params")
def check_params(params: ParserParams) -> Dict[str, Any]:
    parser_params = parser.get_params()
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

    logging.info("parsing text")
    try:
        graphs = parser.parse(text_to_parse.text)
        logging.info("parsing successful")
        json_graphs = [graph.to_json() for graph in graphs]
        return {"status": "ok", "graphs": json_graphs}

    except Exception as e:
        logging.error(f"Error in text classification: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
