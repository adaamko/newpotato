import logging
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from newpotato.datatypes import Triplet
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


class ExtractorData(BaseModel):
    """Model for data to load extractor.

    Args:
        extractor_data (dict): data for loading extractor
    """

    extractor_data: dict


class Data(BaseModel):
    """Model for graphs and triplets to be loaded.

    Args:
        graphs (dict): parsed graphs
        triplets (dict): triplet annotation
    """

    graphs: dict
    triplets: dict


# Data loading endpoints
@app.post("/load_rules")
def load_rules(extractor_data: ExtractorData):
    """Load extractor from saved state

    Args:
        extractor_data (ExtractorData): data for loading the extractor
    """

    logging.info("Loading extractor")
    try:
        hitl_manager.load_extractor(extractor_data.extractor_data)
        logging.info("Extractor loaded successfully")
        return {"status": "ok"}
    except Exception as e:
        logging.error(f"Error loading rules: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load_data")
def load_data(data: Data):
    """Load graphs and triplets from saved state

    Args:
        data (Data): data for loading graphs and triplets
    """

    logging.info("Loading graphs and triplets")
    try:
        hitl_manager.load_data(data.graphs, data.triplets)
        logging.info("Data loaded successfully")
        return {"status": "ok"}
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Parsing Endpoints
@app.post("/parse")
def parse_text(text_to_parse: TextToParse):
    """Parses the text and stores parsed graphs.

    Args:
        text_to_parse (TextToParse): Text to be parsed.
    """

    logging.info("Initiating text parsing.")
    try:
        hitl_manager.get_graphs(text_to_parse.text)
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
        graph = hitl_manager.get_graphs(annotation.text)[0]
        triplet = Triplet(annotation.pred, annotation.args, graph)
        if not triplet.mapped:
            logging.warning("Triplet not mapped.")
            return {
                "status": "error",
                "detail": "Triplet not mapped",
                "graph": graph.to_json(),
            }

        hitl_manager.store_triplet(annotation.text, triplet, True)

        logging.info("Annotation storage successful.")
        return {"status": "ok"}
    except Exception as e:
        logging.error(f"Error storing annotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Token Retrieval Endpoints
@app.get("/tokens/")
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


# Get annotated triplets
@app.get("/triplets")
def get_triplets(text: str) -> Dict[str, Any]:
    """Retrieves annotated triplets.

    Returns:
        Dict[str, Any]: Dictionary containing annotated triplets.
    """
    logging.info("Initiating annotated triplets retrieval.")
    try:
        sen_to_triplets = hitl_manager.get_true_triplets()

        if text not in sen_to_triplets:
            logging.warning(f"No annotated triplets found for text: {text}")
            return {"triplets": []}
        triplets = [
            (triplet.pred, triplet.args, str(triplet), text)
            for triplet in sen_to_triplets[text]
        ]

        logging.debug(f"Retrieved triplets: {triplets}")
        logging.info("Annotated triplets retrieval successful.")
        return {"triplets": triplets}
    except Exception as e:
        logging.error(f"Error retrieving annotated triplets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Delete Triplet endpoint
@app.delete("/triplets")
def delete_triplet(annotation: Annotation) -> Dict[str, Any]:
    """Deletes a triplet.

    Args:
        triplet (Annotation): Triplet to be deleted.
    """

    logging.info("Initiating triplet deletion.")
    try:
        logging.info(
            f"Annotation: {annotation.text}, {annotation.pred}, {annotation.args}"
        )
        graph = hitl_manager.get_graphs(annotation.text)[0]
        triplet = Triplet(annotation.pred, annotation.args, graph)
        logging.info(f"Deleting triplet: {triplet}")
        hitl_manager.delete_triplet(annotation.text, triplet)
        logging.info("Triplet deletion successful.")
        return {"status": "ok"}
    except Exception as e:
        logging.error(f"Error deleting triplet: {e}")
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
        matches_by_text = hitl_manager.extract_triplets_from_text(
            text_to_classify.text, convert_to_text=True
        )
        logging.info("Text classification successful.")
        if not matches_by_text:
            return {"status": "No matches found"}
        else:
            return {"status": "ok", "matches": matches_by_text}

    except Exception as e:
        logging.error(f"Error in text classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))
