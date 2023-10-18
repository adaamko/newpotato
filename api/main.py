import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from newpotato.hitl import HITLManager, TextParser

app = FastAPI()
hitl_manager = HITLManager()
parser = TextParser()

logging.basicConfig(level=logging.INFO)


class TextToParse(BaseModel):
    text: str


@app.post("/toks", status_code=200)
def get_toks(text_to_parse: TextToParse):
    if not hitl_manager.is_parsed(text_to_parse.text):
        error = "get_toks called for unparsed sentence"
        logging.error(error)
        raise HTTPException(status_code=400, detail=error)

    toks = hitl_manager.get_tokens(text_to_parse.text)

    return {"status": "ok", "toks": toks}


@app.post("/parse", status_code=200)
def parse_text(text_to_parse: TextToParse):
    try:
        parsed_graphs = parser.parse(text_to_parse.text)
        hitl_manager.store_parsed_graphs(
            text_to_parse.text, parsed_graphs=parsed_graphs
        )
    except Exception as e:
        logging.error(f"Error parsing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok"}
