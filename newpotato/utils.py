import logging
from typing import List, Dict, Any

import editdistance
from graphbrain.hyperedge import Hyperedge

from newpotato.datatypes import Triplet


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


def edge2toks(edge: Hyperedge, graph: Dict[str, Any]):
    """
    find IDs of tokens covered by an edge of a graph

    Args:
        edge (Hyperedge): the Graphbrain Hyperedge to be mapped to token IDs
        graph (Dict[str, Any]): the Graphbrain Hypergraph of the full utterance

    Returns:
        Tuple[int, ...]: tuple of token IDs covered by the subedge
    """

    print('atom2words:', graph["atom2word"])
    print('key types:', [type(key) for key in graph["atom2word"]])
    print('atom types:', [type(atom) for atom in edge.all_atoms()])
    return tuple(graph["atom2word"][atom][1] for atom in edge.all_atoms())


def phrase2text(phrase, words):
    return " ".join(words[i] for i in phrase)


def get_variables(
    edge: Hyperedge,
    words: List[str],
    triplet: Triplet,
):
    pred, args = triplet.pred, triplet.args
    variables = {"REL": text2subedge(edge, phrase2text(pred, words))}
    variables.update(
        {
            f"ARG{i}": text2subedge(edge, phrase2text(arg, words))
            for i, arg in enumerate(args)
        }
    )
    return variables


def matches2triplets(matches: List[Dict], graph: Dict[str, Any]):
    triplets = []
    for triple_dict in matches:
        pred = None
        args = []
        for key, edge in triple_dict.items():
            if key == "REL":
                pred = edge2toks(edge, graph)
            else:
                args.append(edge2toks(edge, graph))

        triplets.append(Triplet(pred, args))

    return triplets
