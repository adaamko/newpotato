import requests
import streamlit as st

API_URL = "http://localhost:8000"


def api_request(
    method: str, endpoint: str, payload: dict = None, query_params: dict = None
):
    """Generic API request function.

    Args:
        method (str): HTTP method.
        endpoint (str): API endpoint.
        payload (dict, optional): Parameters. Defaults to None.

    Returns:
        dict: Response.
    """
    url = f"{API_URL}/{endpoint}"

    response = requests.request(method, url, json=payload, params=query_params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(response.json())
        raise Exception(response.json())


def add_annotation(text: str, pred: tuple[int, ...], args: list[tuple[int, ...]]):
    """Annotate sentence with triplets.

    Args:
        text (str): Sentence.
        pred (tuple[int, ...]): Predicate.
        args (list[tuple[int, ...]]): Arguments.
    """
    return api_request(
        "POST", "annotate", payload={"text": text, "pred": pred, "args": args}
    )


def delete_annotation(text: str, pred: tuple[int, ...], args: list[tuple[int, ...]]):
    """Delete annotation from sentence.

    Args:
        text (str): Sentence.
        pred (tuple[int, ...]): Predicate.
        args (list[tuple[int, ...]]): Arguments.
    """
    return api_request(
        "DELETE", "triplets", payload={"text": text, "pred": pred, "args": args}
    )


def fetch_sentences():
    """Fetch sentences from the API.

    Returns:
        list: List of sentences.
    """
    return api_request("GET", "sentences")["sentences"]


def fetch_tokens(sentence: str):
    """Fetch tokens from the API.

    Args:
        sentence (str): Sentence to fetch tokens for.

    Returns:
        list: List of tokens.
    """
    return api_request("GET", "tokens/", query_params={"text": sentence})["tokens"]


def fetch_triplets(sentence: str):
    """Fetch triplets from the API.

    Args:
        sentence (str): Sentence to fetch triplets for.

    Returns:
        list: List of triplets.
    """
    return api_request("GET", "triplets/", query_params={"text": sentence})["triplets"]


def fetch_all_triplets():
    """Fetch all triplets from the API.

    Returns:
        list: List of triplets.
    """
    return api_request("GET", "triplets/all")["triplets"]


def fetch_rules(learn: bool = True):
    """Fetch rules from the API.

    Returns:
        dict: Dict of rules.
    """
    return api_request("GET", "rules/", query_params={"learn": learn})["rules"]


def fetch_annotated_graphs():
    """Fetch annotated graphs from the API.

    Returns:
        dict: Dict of annotated graphs.
    """
    return api_request("GET", "annotated_graphs")["annotated_graphs"]


def fetch_inference_for_text(text: str):
    """Fetch inference from the API.

    Args:
        text (str): Text to infer.

    Returns:
        dict: Inference.
    """
    return api_request("POST", "infer", payload={"text": text})["matches"]


def fetch_inference_for_sentences(sentences: list):
    """Fetch inference from the API.

    Args:
        sentences (list): Sentences to infer.

    Returns:
        dict: Inference.
    """
    return api_request("POST", "infer/sentences", payload={"sentences": sentences})[
        "matches"
    ]


def init_session_states():
    """Initialize session states."""
    if "train_classifier" not in st.session_state:
        st.session_state.train_classifier = True
    if "sentences" not in st.session_state:
        st.session_state["sentences"] = []
        st.session_state["sentences_data"] = {}
    if "knowledge_graph" not in st.session_state:
        st.session_state["knowledge_graph"] = None
    if "rules" not in st.session_state:
        st.session_state["rules"] = None
