import logging
from typing import List, Dict, Any, Tuple

import editdistance
from graphbrain.hyperedge import Atom, Hyperedge

from newpotato.datatypes import Triplet

NON_WORD_ATOMS = {"list/J/.", "+/B.am/.", "+/B.mm/."}


def _text2subedge(edge: Hyperedge, text: str) -> Tuple[Hyperedge, int, int]:
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


def text2subedge(edge: Hyperedge, text: str) -> Hyperedge:
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

    logging.debug(f'edge: {edge}, graph["atom2word"]: {graph["atom2word"]}')
    # converting UniqueAtoms to Atoms so that edge atoms can match
    atoms2toks = {Atom(atom): tok for atom, tok in graph["atom2word"].items()}
    
    toks = []
    for atom in edge.all_atoms():
        if atom not in atoms2toks:
            assert str(atom) in NON_WORD_ATOMS, f'no token corresponding to atom "{atom}"'
        else:
            toks.append(atoms2toks[atom][1])

    return tuple(toks)


def get_variables(edge: Hyperedge, words: List[str], triplet: Triplet) -> Dict[str, Hyperedge]:
    """
    get the variables from a hypergraph that correspond to parts of a triplet

    Args:
        edge (Hyperedge): the graph containing the variables
        words (List[str]): the words of the sentence
        triplet (Triplet): the triplet for which the variables are to be extracted

    Returns:
        Dict[str, Hyperedge] the dictionary of variables
    """

    def phrase2text(phrase, words):
        return " ".join(words[i] for i in phrase)

    pred, args = triplet.pred, triplet.args
    variables = {"REL": text2subedge(edge, phrase2text(pred, words))}
    variables.update(
        {
            f"ARG{i}": text2subedge(edge, phrase2text(arg, words))
            for i, arg in enumerate(args)
        }
    )
    return variables


def matches2triplets(matches: List[Dict], graph: Dict[str, Any]) -> List[Triplet]:
    """
    convert graphbrain matches on a sentence to triplets of the tokens of the sentence

    Args:
        matches (List[Dict]): a list of hypergraphs corresponding to the matches
        graphs (Dict[str, Any]]): The hypergraph of the sentence
    
    Returns:
        List[Triplet] the list of triplets corresponding to the matches
    """
    triplets = []
    for triple_dict in matches:
        pred = []
        args = []
        for key, edge in triple_dict.items():
            if key == "REL":
                pred = edge2toks(edge, graph)
            else:
                args.append(edge2toks(edge, graph))

        triplets.append(Triplet(pred, args))

    return triplets
