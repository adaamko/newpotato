import logging

from graphbrain.hyperedge import hedge
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
    def get_atom2token(atom2word, spacy_sentence):
        id2token = {tok.i: tok for tok in spacy_sentence}
        atom2token = {
            hedge(atom): id2token[word[1]] for atom, word in atom2word.items()
        }
        return atom2token

    @staticmethod
    def from_json(data, spacy_vocab):
        graph = GraphParse()
        spacy_sentence = Doc(spacy_vocab).from_json(data["spacy_sentence"])[:]
        print("loaded spacy sentence:", spacy_sentence)

        for key, value in data.items():
            new_value = value
            if key == "atom2token":
                new_value = GraphParse.get_atom2token(data["atom2word"], spacy_sentence)
            elif key == "atom2word":
                new_value = {hedge(k2): tuple(v2) for k2, v2 in value.items()}
            elif key == "word2atom":
                new_value = {int(k2): v2 for k2, v2 in value.items()}
            elif key == "spacy_sentence":
                new_value = spacy_sentence
            elif key == "extra_edges":
                new_value = set(value)
            elif key in ("main_edge", "resolved_corefs"):
                new_value = hedge(value)

            graph[key] = new_value

        return graph

    def to_json(self):
        d = {}
        for key, value in self.items():
            new_value = value
            if key == "atom2token":
                # we cannot serialize spacy Tokens so we reconstruct them from atom2word
                new_value = None
            elif key == "atom2word":
                new_value = {k2.to_str(): v2 for k2, v2 in value.items()}
            elif key == "spacy_sentence":
                new_value = value.as_doc().to_json()
            elif key == "extra_edges":
                # value is a set
                new_value = sorted(list(value))
            elif key in ("main_edge", "resolved_corefs"):
                # values are Hyperedges:
                new_value = value.to_str()

            d[key] = new_value

        return d


class Triplet:
    """A class to handle triplets.

    A triplet consists  of a predicate and a list of arguments.
    """

    def __init__(self, pred, args):
        logging.debug(f'triple init got: pred: {pred}, args: {args}')
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
