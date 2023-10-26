import logging
from typing import List, Tuple

import editdistance
from graphbrain.hyperedge import Hyperedge


def _text2subedge(edge: Hyperedge, text: str):
    """
    find subedge in edge corresponding to the phrase in text. Based on graphbrain.learner.text2subedge.

    Args:
        edge (Hyperedge): the Graphbrain Hyperedge in which to look for the subedge
        text (str): the phrase to be covered by the subedge

    Returns:
        Hyperedge: the best matching hyperedge
        int: edit distance of best matching edge's words from phrase
        int: length of best matching edge
    """
    best_edge = edge
    input_text = text.lower()
    # edge_txt = hg.get_str_attribute(edge, 'text').strip().lower()
    edge_txt = edge.label()
    best_distance = editdistance.eval(edge_txt, input_text)
    best_length = len(edge_txt)

    if edge.not_atom:
        for subedge in edge:
            sedge, distance, length = _text2subedge(subedge, input_text)
            if distance < best_distance or (
                distance == best_distance and length < best_length
            ):
                best_edge = sedge
                best_distance = distance
                best_length = length

    return best_edge, best_distance, best_length


def text2subedge(edge: Hyperedge, text: str):
    """
    find subedge in edge corresponding to the phrase in text. Based on graphbrain.learner.text2subedge.

    Args:
        edge (Hyperedge): the Graphbrain Hyperedge in which to look for the subedge
        text (str): the phrase to be covered by the subedge

    Returns:
        Hyperedge: the best matching hyperedge
    """
    subedge, _, _ = _text2subedge(edge, text)
    logging.debug(f'text2subedge mapping "{text}" to {subedge}')
    return subedge


def phrase2text(phrase, words):
    return " ".join(words[i] for i in phrase)


def get_variables(
    edge: Hyperedge,
    words: List[str],
    triplet: Tuple[Tuple[int, ...], List[Tuple[int, ...]]],
):
    pred, args = triplet
    variables = {"REL": text2subedge(edge, phrase2text(pred, words))}
    variables.update(
        {
            f"ARG{i}": text2subedge(edge, phrase2text(arg, words))
            for i, arg in enumerate(args)
        }
    )
    return variables
