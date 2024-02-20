import json
from collections import defaultdict

import pandas as pd
import streamlit as st
from graphbrain import hedge
from graphbrain.notebook import *  # noqa
from graphbrain.notebook import _edge2html_vblocks
from st_cytoscape import cytoscape
from streamlit_text_annotation import text_annotation
from utils import (
    api_request,
    fetch_annotated_graphs,
    fetch_rules,
    fetch_sentences,
    fetch_tokens,
    fetch_triplets,
    init_session_states,
)


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


def upload_json_file():
    """Upload a JSON file with HITL state data.

    Returns:
        dict: the data in the file.
    """
    file = st.file_uploader("Upload a JSON file")
    if file is not None:
        data = json.load(file)
        return data
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
                "background-color": "#8BC34A",  # Light Green
                "label": "data(id)",
                "color": "#FFFFFF",  # White for text
                "text-valign": "center",
                "text-halign": "center",
                "text-wrap": "wrap",
                "text-max-width": "120px",
                "font-size": "12px",
                "width": "125px",
                "height": "125px",
                "border-color": "#FFFFFF",
                "border-width": "2px",
            },
        },
        {
            "selector": "edge",
            "style": {
                "width": 4,
                "line-color": "#FFC107",  # Amber
                "target-arrow-color": "#FFC107",
                "target-arrow-shape": "triangle",
                "curve-style": "bezier",
                "label": "data(label)",
                "font-size": "12px",
                "color": "#000000",  # Black for text
                "text-background-opacity": 1,
                "text-background-color": "#FFFFFF",  # White background for text
                "text-background-padding": "3px",
                "text-background-shape": "roundrectangle",
            },
        },
    ]

    col1, col2 = st.columns(2)

    with col1:
        selected = cytoscape(
            elements,
            stylesheet,
            selection_type="single",
            key="graph",
            layout={"name": "klay"},
            height="600px",
        )

    with col2:
        if selected:
            edges = selected["edges"]
            if edges:
                edge_id = edges[0]  # Assuming single selection
                arg0, rel, arg1 = edge_id.split("-")

                # Display related information for the selected edge

                st.markdown(
                    "<style> .big-font { font-size:20px !important; } .highlight { background-color: lightyellow; } .rule-style { background-color: lightgrey; padding: 10px; border-radius: 5px; margin: 5px 0; } </style>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<div class='big-font'>Selected Edge: <br> <b>ARG0:</b> {arg0} <br> <b>REL:</b> {rel} <br> <b>ARG1:</b> {arg1}</div>",
                    unsafe_allow_html=True,
                )

                for sentence, rules in knowledge_graph[(rel, arg0, arg1)]:
                    st.markdown(
                        f"<div class='big-font'><div class='highlight'>Sentence: {sentence} </div></div>",
                        unsafe_allow_html=True,
                    )

                    for rule in rules:
                        st.markdown(
                            "<div class='big-font'><div class='rule-style'><b>The matched pattern:</b></div></div>",
                            unsafe_allow_html=True,
                        )

                        html = _edge2html_vblocks(hedge(rule))
                        st.write(html, unsafe_allow_html=True)
                        # st.write(f"Rule: {rule}")


def main():
    """Main function."""
    st.set_page_config(page_title="NewPotato Streamlit App", layout="wide")
    st.title("NewPotato Demo")

    init_session_states()

    add_sentence, annotate, view_rules, inference, load = st.tabs(
        ["Add Sentence", "Annotate", "View Rules", "Inference", "Load"]
    )

    with add_sentence:
        upload_text = upload_text_file()

        if upload_text:
            sentences = st.text_area(
                "Text input", value=upload_text, height=300, key="text"
            )
        else:
            sentences = st.text_area("Text input", height=300, key="text")

        if st.button("Submit Sentence"):
            # Remove 2: from sentences, e.g. "The mandatory 2:Manufacturer provides a human-readable" should be "The mandatory Manufacturer provides a human-readable"
            sentences = sentences.replace("2:", "")

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
                st.session_state.train_classifier = True
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
                if response["status"] == "ok":
                    st.success("Annotation added successfully.")
                elif response["status"] == "error":
                    st.error(
                        "Could not map the annotation to the sentence, please look at the graph and try again: "
                    )
                    with st.expander("Graph"):
                        html = _edge2html_vblocks(hedge(response["graph"]["main_edge"]))
                        st.write(html, unsafe_allow_html=True)

            current_annotations = fetch_triplets(selected_sentence)

            if current_annotations:
                st.write("Current Annotations:")
                df = pd.DataFrame(
                    {
                        "pred": "_".join([str(i) for i in annotation[0]]),
                        "args": ",".join(
                            "_".join([str(i) for i in arg]) for arg in annotation[1]
                        ),
                        "triplet": annotation[2],
                        "delete": False,
                    }
                    for annotation in current_annotations
                )
                edited_df = st.data_editor(df, hide_index=True, key="data_editor")

                if st.button("Delete Selected"):
                    st.session_state.train_classifier = True
                    selected_triplets = edited_df[edited_df["delete"] == True]

                    if not selected_triplets.empty:
                        # iterate on rows, leave out the column names
                        for triplet in selected_triplets.values:
                            pred = triplet[0].split("_")
                            pred = [int(i) for i in pred]
                            pred = tuple(pred)

                            args = [
                                tuple(int(i) for i in arg.split("_"))
                                for arg in triplet[1].split(",")
                            ]
                            payload = {
                                "text": selected_sentence,
                                "pred": pred,
                                "args": args,
                            }
                            response = api_request("DELETE", "triplets", payload)

                            if response["status"] == "ok":
                                st.success("Annotation deleted successfully.")
                            else:
                                st.error("Could not delete the annotation")
                    st.rerun()

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
                sentence_to_classify = sentence_to_classify.replace("2:", "")
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

    with load:
        data = upload_json_file()
        if data:
            if "parsed_graphs" in data and "triplets" in data:
                load_data = st.checkbox("Load data")
            if "extractor_data" in data:
                load_rules = st.checkbox("Load rules")

            if st.button("Load (and overwrite!)"):
                if load_data:
                    payload = {
                        "graphs": data["parsed_graphs"],
                        "triplets": data["triplets"],
                    }
                    response = api_request("POST", "load_data", payload)
                    if response["status"] == "ok":
                        st.success("Data loaded successfully.")
                    else:
                        st.error("Failed to load data")
                if load_rules:
                    payload = {"extractor_data": data["extractor_data"]}
                    response = api_request("POST", "load_rules", payload)
                    if response["status"] == "ok":
                        st.success("Rules loaded successfully.")
                    else:
                        st.error("Failed to load rules")


if __name__ == "__main__":
    main()
