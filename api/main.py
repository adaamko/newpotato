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


@app.post("/parse", status_code=200)
def parse_text(text_to_parse: TextToParse):
    try:
        parsed_graphs = parser.parse(text_to_parse.text)
        hitl_manager.store_parsed_graphs(parsed_graphs=parsed_graphs)
    except Exception as e:
        logging.error(f"Error parsing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "ok"}
