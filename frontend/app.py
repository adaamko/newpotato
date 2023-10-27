import json

import requests
import streamlit as st
from streamlit_text_annotation import text_annotation

API_URL = "http://localhost:8000"


def api_request(method: str, endpoint: str, payload: dict = None):
    """Generic API request function.

    Args:
        method (str): HTTP method.
        endpoint (str): API endpoint.
        payload (dict, optional): Parameters. Defaults to None.

    Returns:
        dict: Response.
    """
    url = f"{API_URL}/{endpoint}"
    response = requests.request(method, url, json=payload)
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
    return api_request("GET", f"tokens/{sentence}")["tokens"]


def fetch_triplets():
    """Fetch triplets from the API.

    Returns:
        dict: Dict of triplets.
    """
    return api_request("GET", "triplets")["triplets"]


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


def upload_text_file():
    """Upload a text file to the API.

    Returns:
        str: the text in the file.
    """
    file = st.file_uploader("Upload a text file", type=["txt"])
    if file is not None:
        text = file.read().decode("utf-8")
        return text
    else:
        return None


def annotate_sentence(selected_sentence: str):
    """Annotate the selected sentence and submit it to the API."""
    tokens = fetch_tokens(selected_sentence)
    sentence_annotation = text_annotation(
        {
            "allowEditing": True,
            "tokens": [
                {
                    "text": f"{tok['index']}_{tok['token']}",
                    "labels": [],
                    "style": {
                        "color": "purple",
                        "font-size": "14px",
                        "font-weight": "bold",
                    },
                }
                for tok in tokens
            ],
            "labels": [
                {
                    "text": "PRED",
                    "style": {
                        "color": "white",
                        "background-color": "red",
                        "font-size": "12px",
                        "border": "1px solid red",
                        "border-radius": "4px",
                        "text-transform": "uppercase",
                    },
                },
                {
                    "text": "ARG1",
                    "style": {
                        "color": "white",
                        "background-color": "green",
                        "font-size": "12px",
                        "border-radius": "4px",
                    },
                },
                {
                    "text": "ARG2",
                    "style": {
                        "color": "white",
                        "background-color": "blue",
                        "font-size": "12px",
                        "border-radius": "4px",
                    },
                },
            ],
        }
    )

    if sentence_annotation:
        sentence_annotation = json.loads(sentence_annotation)
        PREDS = [
            tok["text"]
            for tok in sentence_annotation["tokens"]
            if "PRED" in tok["labels"]
        ]
        ARG1 = [
            tok["text"]
            for tok in sentence_annotation["tokens"]
            if "ARG1" in tok["labels"]
        ]
        ARG2 = [
            tok["text"]
            for tok in sentence_annotation["tokens"]
            if "ARG2" in tok["labels"]
        ]

        st.session_state["sentences_data"][selected_sentence]["annotations"].append(
            {
                "pred": tuple(tok.split("_")[0] for tok in PREDS),
                "args": [
                    tuple(tok.split("_")[0] for tok in arg) for arg in (ARG1, ARG2)
                ],
            }
        )


def main():
    """Main function."""
    st.title("NewPotato Streamlit App")

    # Initialize or get Streamlit state
    if "sentences" not in st.session_state:
        st.session_state["sentences"] = []
        st.session_state["sentences_data"] = {}

    home, add_sentence, annotate, view_rules, inference = st.tabs(
        ["Home", "Add Sentence", "Annotate", "View Rules", "Inference"]
    )

    with home:
        st.write("Welcome to the NewPotato HITL system.")

    with add_sentence:
        upload_text = upload_text_file()

        if upload_text:
            sentences = st.text_area(
                "Text input", value=upload_text, height=300, key="text"
            )
        else:
            sentences = st.text_area("Text input", height=300, key="text")

        if st.button("Submit Sentence"):
            payload = {"text": sentences}
            response = api_request("POST", "parse", payload)
            if response:
                st.session_state["sentences"] = fetch_sentences()
                st.session_state["sentences_data"] = {
                    sen: {"text": sen, "annotations": []}
                    for sen in st.session_state["sentences"]
                }
                st.success("Text parsed successfully.")

    with annotate:
        selected_sentence = st.selectbox(
            "Select a sentence to annotate:", st.session_state["sentences"]
        )

        if selected_sentence:
            annotate_sentence(selected_sentence)

            if st.button("Submit Annotation"):
                payload = {
                    "text": selected_sentence,
                    "pred": st.session_state["sentences_data"][selected_sentence][
                        "annotations"
                    ][-1]["pred"],
                    "args": st.session_state["sentences_data"][selected_sentence][
                        "annotations"
                    ][-1]["args"],
                }
                response = api_request("POST", "annotate", payload)
                if response:
                    st.success("Annotation added successfully.")

    with view_rules:
        # Annotated Graphs
        if st.button("Get Annotated Graphs"):
            annotated_graphs = fetch_annotated_graphs()
            st.write("Annotated Graphs:")
            for graph in annotated_graphs:
                st.write(graph)

        if st.button("Get Rules"):
            rules = fetch_rules()
            st.write("Extracted Rules:")
            # print as strings
            for rule in rules:
                st.write(rule)

    with inference:
        sentence_to_classify = st.text_input("Enter a sentence to classify:")

        if st.button("Classify"):
            if sentence_to_classify:
                payload = {"text": sentence_to_classify}
                response = api_request("POST", "classify_text", payload)
                if response["status"] == "No matches found":
                    st.write("No matches found.")
                else:
                    st.write("Matches:")
                    st.write(response["matches"])
            else:
                st.write("Please enter a sentence to classify.")


if __name__ == "__main__":
    main()
