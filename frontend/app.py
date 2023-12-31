import json
from collections import defaultdict

import requests
import streamlit as st
from st_cytoscape import cytoscape
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


# Enhanced Cytoscape Graph Visualization Function
def visualize_kg(knowledge_graph):
    elements = []

    unique_nodes = set()
    for triplet, _ in knowledge_graph.items():
        unique_nodes.add(triplet[1])
        unique_nodes.add(triplet[2])

    # Add nodes
    for node in unique_nodes:
        elements.append(
            {
                "data": {"id": node, "label": node},
                "selectable": False,
            }
        )

    # Add edges
    for triplet, _ in knowledge_graph.items():
        elements.append(
            {
                "data": {
                    "id": f"{triplet[1]}-{triplet[0]}-{triplet[2]}",
                    "label": triplet[0],
                    "source": triplet[1],
                    "target": triplet[2],
                },
                "selectable": True,
            }
        )

    stylesheet = [
        {
            "selector": "node",
            "style": {
                "background-color": "#007bff",
                "label": "data(id)",
                "color": "#fff",
                "text-valign": "center",
                "text-halign": "center",
                "font-size": "12px",
                "width": "40px",
                "height": "40px",
                "border-color": "#fff",
                "border-width": "2px",
            },
        },
        {
            "selector": "edge",
            "style": {
                "width": 3,
                "line-color": "#00fa9a",
                "target-arrow-color": "#00fa9a",
                "target-arrow-shape": "triangle",
                "curve-style": "bezier",
                "label": "data(label)",
                "font-size": "10px",
                "color": "#000",
                "text-background-opacity": 1,
                "text-background-color": "#fff",
                "text-background-padding": "3px",
                "text-background-shape": "roundrectangle",
            },
        },
    ]

    selected = cytoscape(
        elements,
        stylesheet,
        selection_type="single",
        key="graph",
        layout={"name": "klay"},
    )

    if selected:
        edges = selected["edges"]
        if edges:
            edge_id = edges[0]  # Assuming single selection
            arg0, rel, arg1 = edge_id.split("-")

            # Display related information for the selected edge
            st.write(f"Selected Edge: {rel} between {arg0} and {arg1}")
            for sentence, rules in knowledge_graph[(rel, arg0, arg1)]:
                st.write(f"Sentence: {sentence}")
                for rule in rules:
                    st.write(f"Rule: {rule}")


def main():
    """Main function."""
    st.set_page_config(page_title="NewPotato Streamlit App", layout="wide")
    st.title("NewPotato Streamlit App")

    # Initialize or get Streamlit state
    if "sentences" not in st.session_state:
        st.session_state["sentences"] = []
        st.session_state["sentences_data"] = {}
    if "knowledge_graph" not in st.session_state:
        st.session_state["knowledge_graph"] = None

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
        sentence_to_classify = st.text_area("Text input", height=300)

        if st.button("Classify"):
            if sentence_to_classify.strip():
                payload = {"text": sentence_to_classify}
                response = api_request("POST", "classify_text", payload)
                if response["status"] == "No matches found":
                    st.write("No matches found.")
                else:
                    text_to_matches = response["matches"]
                    triplets_to_sentence_and_rule = defaultdict(list)

                    for text, matches in text_to_matches.items():
                        triplets = matches["matches"]
                        rules = matches["rules_triggered"]

                        for triplet in triplets:
                            triplet_tuple = (
                                triplet["REL"],
                                triplet["ARG0"],
                                triplet["ARG1"],
                            )
                            triplets_to_sentence_and_rule[triplet_tuple].append(
                                (text, rules)
                            )

                    st.session_state["knowledge_graph"] = triplets_to_sentence_and_rule

            else:
                st.write("Please enter a sentence to classify.")

        if st.session_state["knowledge_graph"]:
            st.write("Knowledge Graph")
            # Use cytoscape to display the knowledge graph
            # In the knowledge graph, the nodes are the ARG0, ARG1, the edges are the REL
            # The edges will be selectable, and when selected, the sentences and rules that triggered the edge will be displayed

            visualize_kg(st.session_state["knowledge_graph"])


if __name__ == "__main__":
    main()
