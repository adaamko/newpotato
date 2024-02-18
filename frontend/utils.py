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


def fetch_rules():
    """Fetch rules from the API.

    Returns:
        dict: Dict of rules.
    """
    return api_request("GET", "rules")["rules"]


def fetch_annotated_graphs():
    """Fetch annotated graphs from the API.

    Returns:
        dict: Dict of annotated graphs.
    """
    return api_request("GET", "annotated_graphs")["annotated_graphs"]
