from dataclasses import dataclass
from typing import List, Tuple

from graphbrain.hyperedge import hedge, Hyperedge


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

    def to_json(self):
        d = {}
        for key, value in self.items():
            new_value = value
            if key in ('atom2token', 'atom2word'):
                new_value = {k2.to_str(): v2 for k2, v2 in value.items()}
            elif key == 'spacy_sentence':
                new_value = value.as_doc().to_json()
            elif isinstance(value, set):
                new_value = list(value)
            elif isinstance(value, Hyperedge):
                new_value = value.to_str()
            
            d[key] = new_value

        print('returning this:', d)
        return d


@dataclass
class Triplet:
    """A class to handle triplets.

    A triplet consists  of a predicate and a list of arguments.
    """

    pred: Tuple[int, ...]
    args: List[Tuple[int, ...]]

    def to_json(self):
        return {"pred": self.pred, "args": self.args}
