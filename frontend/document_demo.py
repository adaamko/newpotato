import json
from datetime import datetime

import pandas as pd
import streamlit as st
from streamlit_text_annotation import text_annotation
from utils import (
    add_annotation,
    delete_annotation,
    fetch_documents,
    fetch_sentences,
    fetch_tokens,
    fetch_triplets,
    init_session_states,
    parse_text,
)


def annotate_sentence(selected_sentence: str):
    """Annotate the selected sentence and submit it to the API."""
    tokens = fetch_tokens(selected_sentence)
    sentence_annotation = text_annotation(
        {
            "allowEditing": True,
            "tokens": [
                {
                    "text": f"{tok['token']}",
                    "id": tok["index"],
                    "labels": [],
                    "style": {
                        "color": "black",
                        "font-size": "13px",
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
        },
        key=selected_sentence,
    )

    if sentence_annotation:
        annotated_sentence = json.loads(sentence_annotation)
        PREDS = [
            f"{tok['id']}_{tok['text']}"
            for tok in annotated_sentence["tokens"]
            if "PRED" in tok["labels"]
        ]
        ARG1 = [
            f"{tok['id']}_{tok['text']}"
            for tok in annotated_sentence["tokens"]
            if "ARG1" in tok["labels"]
        ]
        ARG2 = [
            f"{tok['id']}_{tok['text']}"
            for tok in annotated_sentence["tokens"]
            if "ARG2" in tok["labels"]
        ]

        st.session_state["sentences_data"][selected_sentence]["annotations"] = [
            {
                "pred": tuple(tok.split("_")[0] for tok in PREDS),
                "args": [
                    tuple(tok.split("_")[0] for tok in arg) for arg in (ARG1, ARG2)
                ],
                "added": False,
            }
        ]


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


def main():
    """Main function."""
    st.set_page_config(page_title="NewPotato Streamlit App", layout="wide")
    st.title("NewPotato Demo")

    init_session_states()

    add_sentence, annotate, inference = st.tabs(
        ["Add Sentence", "Annotate", "Inference"]
    )

    with add_sentence:
        upload_text = upload_text_file()

        if upload_text:
            text = st.text_area("Text input", value=upload_text, height=300, key="text")
        else:
            text = st.text_area("Text input", height=300, key="text")

        if st.button("Submit Sentence"):
            documents = [doc for doc in text.split("\n") if doc.strip()]

            parsed_documents = fetch_documents()
            if parsed_documents:
                doc_id = len(parsed_documents)
            else:
                doc_id = 0

            for doc in documents:
                parse_text(doc, doc_id)
                doc_id += 1

            st.session_state.sentences = fetch_sentences()
            st.session_state["sentences_data"] = {
                sen: {"text": sen, "annotations": []}
                for sen in st.session_state["sentences"]
            }
            st.success("Text parsed successfully.")

    with annotate:
        documents = fetch_documents()

        if documents:
            df = pd.DataFrame(
                {
                    "Document ID": [i for i, _ in enumerate(documents)],
                    "Document": [" ".join(doc) for doc in documents],
                    # "Selected": [False for _ in documents],
                }
            )

            document_browser = st.data_editor(
                df,
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                disabled=["Document ID", "Document"],
            )

            # Select id to annotate
            selected = st.selectbox("Select Document ID", df["Document ID"])

            sentences = documents[selected]

            for i, sen in enumerate(sentences):
                annotate_sentence(sen)

            if st.button("Submit Annotations"):
                st.session_state.train_classifier = True

                # Search for annotations that have been added ("added": False)
                for sentence, data in st.session_state["sentences_data"].items():
                    for annotation in data["annotations"]:
                        if not annotation["added"]:
                            response = add_annotation(
                                sentence, annotation["pred"], annotation["args"]
                            )

                            if response["status"] == "ok":
                                st.success("Annotation added successfully.")
                                # Mark the annotation as added
                                annotation["added"] = True
                            elif response["status"] == "error":
                                st.error(
                                    "Could not map the annotation to the sentence, please look at the graph and try again: "
                                )

            current_annotations = []

            for sen in sentences:
                current_annotations.extend(
                    [(triplet, sen) for triplet in fetch_triplets(sen)]
                )

            if current_annotations:
                st.write("Current Annotations:")
                df = pd.DataFrame(
                    {
                        "pred": "_".join([str(i) for i in annotation[0]]),
                        "args": ",".join(
                            (
                                "_".join([str(i) for i in arg])
                                if arg is not None
                                else "None"
                            )
                            for arg in annotation[1]
                        ),
                        "triplet": annotation[2],
                        "sentence": sen,
                        "delete": False,
                    }
                    for annotation, sen in current_annotations
                )
                edited_df = st.data_editor(
                    df, hide_index=True, key="annotation_data_editor"
                )
                if st.button("Delete Selected"):
                    st.session_state.train_classifier = True
                    selected_triplets = edited_df[edited_df["delete"] == True]

                    if not selected_triplets.empty:
                        # iterate on rows, leave out the column names
                        for triplet in selected_triplets.values:
                            pred = triplet[0].split("_")
                            pred = [int(i) for i in pred]
                            pred = tuple(pred)

                            if not triplet[1].split(",")[0].strip():
                                args = [None, None]
                            else:
                                args = [
                                    tuple(int(i) for i in arg.split("_"))
                                    for arg in triplet[1].split(",")
                                ]
                            selected_sentence = triplet[3]
                            response = delete_annotation(selected_sentence, pred, args)

                            if response["status"] == "ok":
                                st.success("Annotation deleted successfully.")
                                # Remove from "sentences_data"
                                st.session_state["sentences_data"][selected_sentence][
                                    "annotations"
                                ] = [
                                    ann
                                    for ann in st.session_state["sentences_data"][
                                        selected_sentence
                                    ]["annotations"]
                                    if ann["pred"] != pred or ann["args"] != args
                                ]
                            else:
                                st.error("Could not delete the annotation")
                    st.rerun()


if __name__ == "__main__":
    main()
