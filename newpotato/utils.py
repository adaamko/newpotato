import itertools
import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import editdistance
from graphbrain.hyperedge import Hyperedge

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

    logging.debug(f"edge2toks\n{edge=}\n{graph=}")

    toks = set()
    strs_to_atoms = defaultdict(list)
    for atom, word in graph["atom2word"].items():
        strs_to_atoms[atom.to_str()].append(atom)

    to_disambiguate = []
    for atom in edge.all_atoms():
        atom_str = atom.to_str()
        if atom_str not in strs_to_atoms:
            assert (
                str(atom) in NON_WORD_ATOMS
            ), f'no token corresponding to {atom=} in {strs_to_atoms=}'
        else:
            cands = strs_to_atoms[atom_str]
            if len(cands) == 1:
                toks.add(graph["atom2word"][cands[0]][1])
            else:
                to_disambiguate.append([graph["atom2word"][cand][1] for cand in cands])

    if len(to_disambiguate) > 0:
        hyp_sets = set()
        for cand in itertools.product(*to_disambiguate):
            hyp_set = toks | set(cand)
            # https://stackoverflow.com/a/33575259/2753770
            if sorted(hyp_set) == list(range(min(hyp_set), max(hyp_set) + 1)):
                hyp_sets.add(tuple(sorted(hyp_set)))
        if len(hyp_sets) > 1:
            logging.warning(f"cannot disambiguate: {edge=}, {hyp_sets=}, {graph=}")
        for tok in list(hyp_sets)[0]:
            toks.add(tok)

    return tuple(sorted(toks))


def get_variables(
    edge: Hyperedge, words: List[str], triplet: Triplet
) -> Dict[str, Hyperedge]:
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
    logging.debug('mapped')
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
                args.append((int(key[-1]), edge2toks(edge, graph)))
        
        sorted_args = [arg[1] for arg in sorted(args)]
        triplets.append(Triplet(pred, sorted_args))

    return triplets
