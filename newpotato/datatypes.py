import logging
from functools import total_ordering
from typing import List, Tuple

from tuw_nlp.graph.graph import Graph
from tuw_nlp.graph.ud_graph import UDGraph


@total_ordering
class Triplet:
    """A class to handle triplets.

    A triplet consists  of a predicate and a list of arguments.
    """

    def __init__(self, pred, args, toks=None):
        logging.debug(f"triple init got: pred: {pred}, args: {args}")
        self.pred = None if pred is None else tuple(int(i) for i in pred)
        self.args = tuple(
            tuple(int(i) for i in arg) if arg is not None else None for arg in args
        )
        self.toks = toks
        self.mapped = False

    @staticmethod
    def from_json(data):
        if data["type"] == "triplet":
            return Triplet(
                data["pred"],
                data["args"],
            )
        elif data["type"] == "graph_mapped":
            return GraphMappedTriplet.from_json(data)
        else:
            raise ValueError(data["type"])

    def to_json(self):
        return {"pred": self.pred, "args": self.args, "type": "triplet"}

    def __eq__(self, other):
        return (
            isinstance(other, Triplet)
            and self.pred == other.pred
            and self.args == other.args
        )

    def __lt__(self, other):
        return self._as_tuple() < other._as_tuple()

    def __hash__(self):
        return hash(self._as_tuple())

    def _as_tuple(self):
        return (self.pred, self.args)

    def to_str(self, toks):
        pred_phrase = "" if self.pred is None else "_".join(toks[a] for a in self.pred)
        args_str = ", ".join(
            "_".join(toks[a] for a in phrase) if phrase is not None else "None"
            for phrase in self.args
        )
        return f"{pred_phrase}({args_str})"

    def __str__(self):
        if self.toks:
            return self.to_str(self.toks)
        else:
            return f"{self.pred=}, {self.args=}"

    def __repr__(self):
        return str(self)


class GraphMappedTriplet(Triplet):
    def __init__(self, triplet: Triplet, pred_graph: Graph, arg_graphs: Tuple[Graph]):
        super(GraphMappedTriplet, self).__init__(
            triplet.pred, triplet.args, toks=triplet.toks
        )
        self.pred_graph = pred_graph
        self.arg_graphs = arg_graphs
        self.mapped = True

    @staticmethod
    def from_json(data):
        triplet = Triplet(
            data["pred"],
            data["args"],
        )
        pred_graph = (
            UDGraph.from_json(data["pred_graph"])
            if data["pred_graph"] is not None
            else None
        )
        arg_graphs = [
            UDGraph.from_json(graph) if graph is not None else None
            for graph in data["arg_graphs"]
        ]

        return GraphMappedTriplet(triplet, pred_graph, arg_graphs)

    @property
    def arg_roots(self):
        return [a_graph.root for a_graph in self.arg_graphs]
    
    def to_json(self):
        return {
            "pred": self.pred,
            "args": self.args,
            "pred_graph": self.pred_graph.to_json()
            if self.pred_graph is not None
            else None,
            "arg_graphs": [
                graph.to_json() if graph is not None else None
                for graph in self.arg_graphs
            ],
            "type": "graph_mapped",
        }


def triplets_to_str(triplets: List[Triplet]) -> List[str]:
    """
    Returns human-readable versions of triplets for a sentence

    Args:
        triplets (List[Triplet]): the triplets to convert

    Returns:
        List[str]: the human-readable form of the triplet
    """
    return [str(triplet) for triplet in triplets]
