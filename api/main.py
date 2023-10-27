import logging
from dataclasses import astuple
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from newpotato.hitl import HITLManager

# Initialize FastAPI, HITLManager
app = FastAPI()
hitl_manager = HITLManager()
# Set up logging
logging.basicConfig(level=logging.INFO)


# Define request and response models
class TextToParse(BaseModel):
    """Model for text to be parsed.

    Args:
        text (str): Text to be parsed.
    """

    text: str


class Annotation(BaseModel):
    """Model for text annotations.

    Args:
        text (str): Text to be annotated.
        pred Tuple[int, ...]: Predicate of the triplet.
        args List[Tuple[int, ...]]: Arguments of the triplet.
    """

    text: str
    pred: Tuple[int, ...]
    args: List[Tuple[int, ...]] = Field(..., min_items=2)


# Parsing Endpoints
@app.post("/parse")
def parse_text(text_to_parse: TextToParse):
    """Parses the text and stores parsed graphs.

    Args:
        text_to_parse (TextToParse): Text to be parsed.
    """

    logging.info("Initiating text parsing.")
    try:
        hitl_manager.add_text_to_graphs(text_to_parse.text)
        logging.info("Text parsing and storage successful.")
        return {"status": "ok"}
    except Exception as e:
        logging.error(f"Error parsing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Annotation Endpoints
@app.post("/annotate")
def annotate_text(annotation: Annotation) -> Dict[str, Any]:
    """Stores text annotation as triplet.

    Args:
        annotation (Annotation): Text annotation.
    """

    logging.info("Initiating annotation storage.")
    try:
        hitl_manager.store_triplet(annotation.text, annotation.pred, annotation.args)
        logging.info("Annotation storage successful.")
        return {"status": "ok"}
    except Exception as e:
        logging.error(f"Error storing annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Token Retrieval Endpoints
@app.get("/tokens/{text}")
def get_tokens(text: str) -> Dict[str, Any]:
    """Retrieves tokens for parsed text.

    Args:
        text (str): Text to retrieve tokens for.

    Returns:
        Dict[str, Any]: Dictionary containing tokens.
    """
    logging.info(f"Retrieving tokens for text: {text}.")
    if not hitl_manager.is_parsed(text):
        logging.warning("Text not parsed.")
        raise HTTPException(status_code=400, detail="Text not parsed")
    tokens = hitl_manager.get_tokens(text)
    indexed_tokens = [
        {"index": i, "token": str(token)} for i, token in enumerate(tokens)
    ]
    logging.info("Token retrieval successful.")
    return {"tokens": indexed_tokens}


# Triplet Retrieval Endpoints
@app.get("/triplets")
def get_triplets() -> Dict[str, Any]:
    """Retrieves stored triplets.

    Returns:
        Dict[str, Any]: Dictionary containing triplets.
    """
    logging.info("Initiating triplet retrieval.")
    try:
        sen_to_triplets = hitl_manager.get_triplets()
        sen_to_triplets = {
            sen: [astuple(triplet) for triplet in triplets]
            for sen, triplets in sen_to_triplets.items()
        }

        logging.info("Triplet retrieval successful.")
        return {"triplets": sen_to_triplets}
    except Exception as e:
        logging.error(f"Error retrieving triplets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Sentence Retrieval Endpoints
@app.get("/sentences")
def get_sentences() -> Dict[str, List[str]]:
    """Retrieves all parsed sentences.

    Returns:
        Dict[str, Any]: Dictionary containing sentences.
    """
    logging.info("Initiating sentence retrieval.")
    try:
        parsed_graphs = hitl_manager.parsed_graphs
        sentences = [sen for sen in parsed_graphs.keys() if sen != "latest"]
        logging.info("Sentence retrieval successful.")
        return {"sentences": sentences}
    except Exception as e:
        logging.error(f"Error retrieving sentences: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Rule Endpoints
@app.get("/rules")
def get_rules() -> Dict[str, Any]:
    """Extracts and retrieves rules based on stored triplets.

    Returns:
        Dict[str, Any]: Dictionary containing rules.
    """
    logging.info("Initiating rule extraction and retrieval.")
    try:
        rules = hitl_manager.get_rules()
        logging.info("Rule extraction and retrieval successful.")
        return {"rules": rules}
    except Exception as e:
        logging.error(f"Error in rule extraction or retrieval: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/annotated_graphs")
def get_annotated_graphs() -> Dict[str, Any]:
    """Retrieves annotated graphs.

    Returns:
        Dict[str, Any]: Dictionary containing annotated graphs.
    """
    logging.info("Initiating annotated graph retrieval.")
    try:
        annotated_graphs = hitl_manager.get_annotated_graphs()
        logging.info("Annotated graph retrieval successful.")
        return {"annotated_graphs": annotated_graphs}
    except Exception as e:
        logging.error(f"Error retrieving annotated graphs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Text Classification Endpoints
@app.post("/classify_text")
def classify_text(text_to_classify: TextToParse) -> Dict[str, Any]:
    """Classifies the text based on stored rules.

    Args:
        text_to_classify (TextToParse): Text to be classified.

    Returns:
        Dict[str, Any]: Dictionary containing classification results.
    """

    logging.info("Initiating text classification.")
    try:
        matches_by_text = hitl_manager.extract_triplets_from_text(text_to_classify.text)
        logging.info("Text classification successful.")
        if not matches_by_text:
            return {"status": "No matches found"}
        else:
            return {"status": "ok", "matches": matches_by_text}

    except Exception as e:
        logging.error(f"Error in text classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))
