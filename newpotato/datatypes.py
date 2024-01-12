import logging

from graphbrain.hyperedge import hedge, unique
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
            "word2atom": self["word2atom"],
            "failed": self["failed"],
        }


class Triplet:
    """A class to handle triplets.

    A triplet consists  of a predicate and a list of arguments.
    """

    def __init__(self, pred, args):
        logging.debug(f"triple init got: pred: {pred}, args: {args}")
        self.pred = tuple(int(i) for i in pred)
        self.args = tuple(tuple(int(i) for i in arg) for arg in args)

    @staticmethod
    def from_json(data):
        return Triplet(data["pred"], data["args"])

    def to_json(self):
        return {"pred": self.pred, "args": self.args}

    def __eq__(self, other):
        return self.pred == other.pred and self.args == other.args

    def __hash__(self):
        return hash((self.pred, self.args))
