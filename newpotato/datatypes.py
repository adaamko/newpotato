import logging
from typing import Set, Tuple

from graphbrain.hyperedge import hedge, Hyperedge, unique
from spacy.tokens.doc import Doc


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


def _text2subedge(edge: Hyperedge, words: Set[str]) -> Tuple[Hyperedge, Set[str]]:
    """
    recursive helper function of text2subedge

    Args:
        edge (Hyperedge): the Graphbrain Hyperedge in which to look for the subedge
        words (set): the words to be covered by the subedge

    Returns:
        Hyperedge: the best matching subedge
        set: words covered by the matching hyperedge
    """
    if edge.is_atom():
        word = edge.label()
        if word in words:
            # an edge matching one word
            return edge, set([word])
        # an irrelevant edge
        return edge, set()

    words_covered = set()
    relevant_subedges = []
    for subedge in edge:
        s_edge, subedge_words_covered = _text2subedge(subedge)
        if subedge_words_covered == words:
            # a subedge covering everything, search can stop
            return s_edge, subedge_words_covered
        elif len(subedge_words_covered) > 0:
            words_covered |= subedge_words_covered
            relevant_subedges.append(s_edge)

    if len(relevant_subedges) == 1:
        # only one relevant subedge
        return relevant_subedges[0], words_covered

    # more than one relevant subedge OR no words covered
    return edge, words_covered


def text2subedge(edge: Hyperedge, words: Set[str]) -> Hyperedge:
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
    subedge, words_covered = _text2subedge(edge, words)
    logging.debug(f'text2subedge: best subedge for "{words}": {subedge}')
    if words == words_covered:
        return subedge, True
    elif words.issubset(words_covered):
        return subedge, False
    else:
        raise ValueError(f"hyperedge {edge} does not contain all words in {words}")


class Triplet:
    """A class to handle triplets.

    A triplet consists  of a predicate and a list of arguments.
    """

    def __init__(self, pred, args, sen_graph=None, strict=True):
        logging.debug(f"triple init got: pred: {pred}, args: {args}")
        self.pred = tuple(int(i) for i in pred)
        self.args = tuple(tuple(int(i) for i in arg) for arg in args)
        self.mapped = False
        self.variables = None
        if sen_graph is not None:
            self.map_to_subgraphs(sen_graph, strict=strict)

    @staticmethod
    def from_json(data):
        return Triplet(data["pred"], data["args"])

    def to_json(self):
        return {"pred": self.pred, "args": self.args}

    def __eq__(self, other):
        return self.pred == other.pred and self.args == other.args

    def __hash__(self):
        return hash((self.pred, self.args))

    def to_str(self, graph):
        toks = self.get_tokens(graph)
        pred_phrase = "_".join(toks[a].text for a in self.pred)
        args_str = ", ".join(
            "_".join(toks[a].text for a in phrase) for phrase in self.args
        )
        return f"{pred_phrase}({args_str})"

    def __str__(self):
        if self.mapped:
            return self.to_str(self.sen_graph)
        else:
            return f"{self.pred=}, {self.args=}"

    def map_to_subgraphs(self, sen_graph, strict=True):
        words = [tok.text for tok in sen_graph["spacy_sentence"]]

        def phrase2words(phrase):
            return set(words[i] for i in phrase)

        edge = sen_graph["main_edge"]
        rel_edge, exact_match = text2subedge(edge, phrase2words(self.pred))
        if not exact_match and strict:
            logging.warning(
                f"cannot map pred {self.pred} to subedge of {edge} (closest: {rel_edge}"
            )
            return False
        variables = {"REL": rel_edge}

        for i, arg in enumerate(self.args):
            arg_edge, exact_match = text2subedge(edge, phrase2words(arg))
            if not exact_match and strict:
                logging.warning(
                    f"cannot map arg {arg} to subedge of {edge} (closest: {rel_edge}"
                )
                return False
            variables[f"ARG{i}"] = arg_edge

        self.variables = variables
        self.mapped = True
        self.sen_graph = sen_graph
        return True
