import itertools
import logging
from collections import defaultdict
from typing import List, Dict, Any

from graphbrain.hyperedge import Hyperedge

from newpotato.datatypes import Triplet

NON_WORD_ATOMS = {"list/J/.", "+/B.am/.", "+/B.mm/."}


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
            ), f"no token corresponding to {atom=} in {strs_to_atoms=}"
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
        triplets.append(Triplet(pred, sorted_args, graph))

    return triplets
