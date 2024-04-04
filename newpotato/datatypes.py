import logging
from collections import defaultdict
from typing import Dict, Set, Tuple

from graphbrain.hyperedge import Hyperedge, hedge, unique
from spacy.tokens.doc import Doc

from newpotato.constants import NON_ATOM_WORDS, NON_WORD_ATOMS


class UnmappableTripletError(Exception):
    pass


class GraphParse(dict):
    """A class to handle Graphbrain graphs.

    A graphbrain parser result looks like this:
    {
        'main_edge': (loves/Pd.so.|f--3s-/en adam/Cp.s/en andi/Cp.s/en),
        'extra_edges': set(),
        'failed': False,
        'text': 'Adam loves Andi.',
        'atom2word': {loves/Pd.so.|f--3s-/en: ('loves', 1), adam/Cp.s/en: ('Adam', 0), andi/Cp.s/en: ('Andi', 2)},
        'atom2token': {adam/Cp.s/en: Adam, andi/Cp.s/en: Andi, loves/Pd.so.|f--3s-/en: loves},
        'spacy_sentence': Adam loves Andi.,
        'resolved_corefs': (loves/Pd.so.|f--3s-/en adam/Cp.s/en andi/Cp.s/en)
        'word2atom': {0: adam/Cp.s/en, 1: loves/Pd.so.|f--3s-/en, 2: andi/Cp.s/en}
    }

    """

    @staticmethod
    def from_json(data, spacy_vocab):
        graph = GraphParse()
        graph["spacy_sentence"] = Doc(spacy_vocab).from_json(data["spacy_sentence"])[:]
        graph["text"] = data["text"]
        graph["failed"] = data["failed"]
        graph["extra_edges"] = set(data["extra_edges"])
        graph["main_edge"] = hedge(data["main_edge"])
        graph["resolved_corefs"] = hedge(data["resolved_corefs"])
        graph["word2atom"] = {
            int(i_str): unique(hedge(atom_str))
            for i_str, atom_str in data["word2atom"].items()
        }
        graph["atom2word"], graph["atom2token"] = {}, {}
        for tok in graph["spacy_sentence"]:
            if tok.i not in graph["word2atom"]:
                continue
            atom = graph["word2atom"][tok.i]
            graph["atom2token"][atom] = tok
            graph["atom2word"][atom] = (tok.text, tok.i)

        return graph

    def to_json(self):
        return {
            "spacy_sentence": self["spacy_sentence"].as_doc().to_json(),
            "extra_edges": sorted(list(self["extra_edges"])),
            "main_edge": self["main_edge"].to_str(),
            "resolved_corefs": self["resolved_corefs"].to_str(),
            "text": self["text"],
            "word2atom": {
                word: atom.to_str() for word, atom in self["word2atom"].items()
            },
            "failed": self["failed"],
        }


def _toks2subedge(
    edge: Hyperedge,
    toks_to_cover: Tuple[int],
    all_toks: Tuple[int],
    words_to_i: Dict[str, Set[int]],
) -> Tuple[Hyperedge, Set[str]]:
    """
    recursive helper function of toks2subedge

    Args:
        edge (Hyperedge): the Graphbrain Hyperedge in which to look for the subedge
        words (tuple): the tokens to be covered by the subedge
        all_toks (tuple): all tokens in the sentence
        words_to_i (dict): words mapped to token indices

    Returns:
        Hyperedge: the best matching subedge
        set: tokens covered by the matching hyperedge
        set: additional tokens in the matching hyperedge
    """
    words_to_cover = [all_toks[i] for i in toks_to_cover]
    logging.debug(f"_toks2subedge got: {edge=}, {words_to_cover=}")
    if edge.is_atom():
        lowered_word = edge.label().lower()
        if lowered_word not in words_to_i:
            assert (
                lowered_word in NON_WORD_ATOMS
            ), f"no token corresponding to edge label {lowered_word} and it is not listed as a non-word atom"

        toks = words_to_i[lowered_word]
        relevant_toks = toks & toks_to_cover
        if len(relevant_toks) > 0:
            return edge, relevant_toks, set()
        else:
            if lowered_word in NON_ATOM_WORDS:
                return edge, set(), set()
            return edge, set(), toks

    relevant_toks, irrelevant_toks = set(), set()
    for subedge in edge:
        s_edge, subedge_relevant_toks, subedge_irrelevant_toks = _toks2subedge(
            subedge, toks_to_cover, all_toks, words_to_i
        )
        if subedge_relevant_toks == toks_to_cover:
            # a subedge covering everything, search can stop
            logging.debug(
                f"_toks2subedge: subedge covers everything, returning {s_edge=}, {subedge_relevant_toks=}, {subedge_irrelevant_toks=}"
            )
            return s_edge, subedge_relevant_toks, subedge_irrelevant_toks

        relevant_toks |= subedge_relevant_toks
        irrelevant_toks |= subedge_irrelevant_toks

    # more than one relevant subedge OR no words covered
    logging.debug(
        f"_toks2subedge: returning {edge=}, {relevant_toks=}, {irrelevant_toks=}"
    )
    return edge, relevant_toks, irrelevant_toks


def toks2subedge(
    edge: Hyperedge,
    toks: Tuple[int],
    all_toks: Tuple[int],
    words_to_i: Dict[str, Set[int]],
) -> Hyperedge:
    """
    find subedge in edge corresponding to the phrase in text.
    Based on graphbrain.learner.text2subedge, but keeps track of the set of words covered
    by partial results.
    If an exact match is not possible, returns the smallest subedge that covers all words.
    Raises ValueError if input edge does not contain all words.

    Args:
        edge (Hyperedge): the Graphbrain Hyperedge in which to look for the subedge
        words (set): the words to be covered by the subedge

    Returns:
        Hyperedge: the best matching hyperedge
        bool: whether the matching edge is exact (contains all the words and no other words)
    """
    try:
        toks_to_cover = {
            tok for tok in toks if all_toks[tok].lower() not in NON_ATOM_WORDS
        }
    except IndexError:
        logging.warning(f"some token IDs out of range: {toks=}, {all_toks=}")
        raise UnmappableTripletError()
    subedge, relevant_toks, irrelevant_toks = _toks2subedge(
        edge, toks_to_cover, all_toks, words_to_i
    )
    logging.debug(f"toks2subedge: {subedge=}, {relevant_toks=}, {irrelevant_toks=}")

    if toks_to_cover == relevant_toks:
        if len(irrelevant_toks) == 0:
            return subedge, relevant_toks, True
        return subedge, relevant_toks, False
    else:
        words = [all_toks[t] for t in toks_to_cover]
        logging.warning(
            f"hyperedge {edge} does not contain all words in {words}\n{toks_to_cover=}, {relevant_toks=}"
        )
        raise UnmappableTripletError()


class Triplet:
    """A class to handle triplets.

    A triplet consists  of a predicate and a list of arguments.
    """

    def __init__(
        self, pred, args, sen_graph=None, mapped=False, variables=None, strict=True
    ):
        logging.debug(f"triple init got: pred: {pred}, args: {args}")
        self.pred = None if pred is None else tuple(int(i) for i in pred)
        self.args = tuple(tuple(int(i) for i in arg) for arg in args)
        self.mapped = mapped
        self.variables = variables
        if sen_graph is not None:
            self.map_to_subgraphs(sen_graph, strict=strict)

    @staticmethod
    def from_json(data):
        return Triplet(
            data["pred"],
            data["args"],
            mapped=data["mapped"],
            variables=data["variables"],
        )

    @staticmethod
    def from_json_and_graph(data, sen_graph):
        return Triplet(
            data["pred"],
            data["args"],
            sen_graph=sen_graph,
            mapped=data["mapped"],
            variables=data["variables"],
        )

    def to_json(self):
        return {
            "pred": self.pred,
            "args": self.args,
            "mapped": self.mapped,
            "variables": self.variables,
        }

    def __eq__(self, other):
        return self.pred == other.pred and self.args == other.args

    def __hash__(self):
        return hash((self.pred, self.args))

    def to_str(self, graph):
        toks = [tok for tok in graph["spacy_sentence"]]
        pred_phrase = (
            "" if self.pred is None else "_".join(toks[a].text for a in self.pred)
        )
        args_str = ", ".join(
            "_".join(toks[a].text for a in phrase) for phrase in self.args
        )
        return f"{pred_phrase}({args_str})"

    def __str__(self):
        if self.mapped:
            return self.to_str(self.sen_graph)
        else:
            return f"{self.pred=}, {self.args=}"

    def __repr__(self):
        return str(self)

    def _map_to_subgraphs(self, sen_graph, strict=True):
        """
        helper function for map_to_subgraphs
        """
        all_toks = tuple(tok.text for tok in sen_graph["spacy_sentence"])
        words_to_i = defaultdict(set)
        for i, word in enumerate(all_toks):
            words_to_i[word.lower()].add(i)

        edge = sen_graph["main_edge"]
        variables = {}

        if self.pred is not None:
            rel_edge, relevant_toks, exact_match = toks2subedge(
                edge, self.pred, all_toks, words_to_i
            )
            if not exact_match and strict:
                logging.warning(
                    f"cannot map pred {self.pred} to subedge of {edge} (closest: {rel_edge}"
                )
                raise UnmappableTripletError
            variables["REL"] = rel_edge
            mapped_pred = tuple(sorted(relevant_toks))
        else:
            mapped_pred = None

        mapped_args = []
        for i in range(len(self.args)):
            arg_edge, relevant_toks, exact_match = toks2subedge(
                edge, self.args[i], all_toks, words_to_i
            )
            if not exact_match and strict:
                logging.warning(
                    f"cannot map arg {self.args[i]} to subedge of {edge} (closest: {arg_edge}"
                )
                raise UnmappableTripletError()
            variables[f"ARG{i}"] = arg_edge
            mapped_args.append(tuple(sorted(relevant_toks)))

        return mapped_pred, mapped_args, variables

    def map_to_subgraphs(self, sen_graph, strict=True):
        """
        map predicate and arguments of a triplet (each a tuple of token indices) to
        corresponding subgraphs (Hyperedges). The mapping may change the indices, since words
        not showing up in the hypergraph (e.g. punctuation) are not to be considered part of the triplet
        """
        try:
            mapped_pred, mapped_args, variables = self._map_to_subgraphs(
                sen_graph, strict=strict
            )
        except UnmappableTripletError:
            logging.warning(
                f"could not map triplet ({self.pred=}, {self.args=} to {sen_graph=}"
            )
            return False
        else:
            self.pred = mapped_pred
            self.args = tuple(mapped_args)
            self.variables = variables
            self.mapped = True
            self.sen_graph = sen_graph
            return True
