import itertools
import logging
from collections import defaultdict
from typing import Any, Dict, List

from graphbrain.hyperedge import Hyperedge

from newpotato.constants import NON_WORD_ATOMS
from newpotato.datatypes import Triplet


def print_tokens(sentence, hitl, console):
    tokens = hitl.get_tokens(sentence)
    console.print("[bold cyan]Tokens:[/bold cyan]")
    console.print(" ".join(f"{i}_{tok}" for i, tok in enumerate(tokens)))


def _get_single_triplet_from_user(console):
    annotation = input("> ")
    if annotation == "":
        return None

    if annotation == "O":
        return "O"

    try:
        phrases = [
            tuple(int(n) for n in ids.split("_")) for ids in annotation.split(",")
        ]
        return phrases[0], phrases[1:]
    except ValueError:
        console.print("[bold red]Could not parse this:[/bold red]", annotation)
        return False


def get_single_triplet_from_user(sentence, hitl, console, expect_mappable=True):
    console.print(
        """
        [bold cyan] Enter comma-separated list of predicate and args, with token IDs in each separated by underscores, e.g.: 0_The 1_boy 2_has 3_gone 4_to 5_school -> 2_3,0_1,4_5
        Alternatively, type O to get triplet from oracle (if loaded).
        Press ENTER to finish.
        
        [/bold cyan]"""
    )
    graph = hitl.parsed_graphs[sentence]
    while True:
        triplet = _get_single_triplet_from_user(console)
        if triplet is None:
            # user is done
            return None
        if triplet is False:
            # syntax error
            continue
        if triplet == "O":
            # oracle
            return "O"

        pred, args = triplet
        mapped_triplet = Triplet(pred, args, graph)
        if expect_mappable and mapped_triplet.mapped is False:
            console.print(
                f"[bold red] Could not map annotation {triplet} to subedges, please provide alternative (or press ENTER to skip)[/bold red]"
            )
            continue

        return mapped_triplet


def get_triplets_from_user(sentence, hitl, console):
    print_tokens(sentence, hitl, console)

    while True:
        triplet = get_single_triplet_from_user(sentence, hitl, console)
        if triplet is None:
            break
        elif triplet == "O":
            if hitl.oracle is None:
                console.print("[bold red]The Oracle is not loaded.[/bold red]")
            if sentence not in hitl.oracle:
                console.print(
                    "[bold red]The Oracle does not annotate this sentence.[/bold red]"
                )
            for triplet, is_true in hitl.oracle[sentence]:
                console.print(f"[bold cyan]Oracle says: {triplet}[/bold cyan]")
                hitl.store_triplet(sentence, triplet, is_true)
        else:
            hitl.store_triplet(sentence, triplet, True)


def get_triplet_from_annotation(
    pred, args, sen, sen_graph, hitl, console, ask_user=True
):
    triplet = Triplet(pred, args, sen_graph)
    if not triplet.mapped:
        console.print(
            f"[bold red]Could not map annotation {str(triplet)} to subedges)[/bold red]"
        )
        if not ask_user:
            console.print("[bold red]Returning unmapped triplet[/bold red]")
            return triplet

        console.print(
            "[bold red]Please provide alternative (or press ENTER to skip)[/bold red]"
        )
        print_tokens(sen, hitl, console)
        triplet = get_single_triplet_from_user(sen, hitl, console)
        if triplet is None:
            console.print("[bold red]No triplet returned[/bold red]")
    return triplet


def edge2toks(edge: Hyperedge, graph: Dict[str, Any]):
    """
    find IDs of tokens covered by an edge of a graph
    If some atom names match more than one token, candidate token sequences are disambiguated
    based on length and the shortest sequence (i.e. the one with the fewest gaps) is returned

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
        logging.debug(f"edge2toks disambiguation needed: {toks=}, {to_disambiguate=}")
        hyp_sets = []
        for cand in itertools.product(*to_disambiguate):
            hyp_toks = sorted(toks | set(cand))
            hyp_length = hyp_toks[-1] - hyp_toks[0]
            hyp_sets.append((hyp_length, hyp_toks))

        shortest_hyp = sorted(hyp_sets)[0][1]
        logging.debug(f"{shortest_hyp=}")
        return set(shortest_hyp)

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
