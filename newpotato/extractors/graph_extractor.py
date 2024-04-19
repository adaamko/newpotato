from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from newpotato.datatypes import Triplet
from newpotato.extractors.extractor import Extractor
from newpotato.extractors.graph_parser_client import GraphParserClient

from tuw_nlp.graph.graph import Graph


class GraphMappedTriplet(Triplet):
    def __init__(self, triplet: Triplet, pred_graph: Graph, arg_graphs: Tuple[Graph]):
        super(GraphMappedTriplet, self).__init__(triplet.pred, triplet.args)
        self.pred_graph = pred_graph
        self.arg_graphs = arg_graphs
        self.mapped = True


class GraphBasedExtractor(Extractor):
    @staticmethod
    def from_json(data: Dict[str, Any]):
        raise NotImplementedError

    def to_json(self) -> Dict[str, Any]:
        raise NotImplementedError

    def __init__(
        self,
        parser_url: Optional[str] = "http://localhost:7277",
        default_relation: Optional[str] = None,
    ):
        super(GraphBasedExtractor, self).__init__()
        self.text_parser = GraphParserClient(parser_url)
        self.default_relation = default_relation

    def _parse_text(self, text: str):
        """
        Parse the given text.

        Args:
            text (str): The text to parse.

        Returns:
            TODO
        """
        graphs = self.text_parser.parse(text)
        for graph in graphs:
            yield graph.text, graph

    def get_tokens(self, sen) -> List[str]:
        """
        Get the tokens of the given text.
        """
        return self.parsed_graphs[sen].tokens

    def get_lemmas(self, sen) -> List[str]:
        """
        Get the lemmas of the given text.
        """
        return [w.lemma for w in self.parsed_graphs[sen].stanza_sen.words]

    def get_rules(self, text_to_triplets):
        pred_graphs = Counter()
        arg_graphs = defaultdict(Counter)
        for text, triplets in text_to_triplets.items():
            # toks = self.get_tokens(text)
            lemmas = self.get_lemmas(text)
            for triplet in triplets:
                if triplet.pred is not None:
                    pred_graphs[triplet.pred_graph] += 1
                    pred_lemmas = tuple(lemmas[i] for i in triplet.pred)
                else:
                    pred_lemmas = (self.default_relation,)
                for arg_graph in triplet.arg_graphs:
                    arg_graphs[pred_lemmas][arg_graph] += 1
        self.pred_graphs = pred_graphs
        self.arg_graphs = arg_graphs

    def print_rules(self):
        print(f"{self.pred_graphs=}")
        print(f"{self.arg_graphs=}")

    def extract_triplets_from_text(self, text, **kwargs):
        raise NotImplementedError

    def map_triplet(self, triplet, sentence, **kwargs):
        graph = self.parsed_graphs[sentence]
        pred_subgraph = (
            graph.subgraph_from_tok_ids(
                triplet.pred, handle_unconnected="shortest_path"
            )
            if triplet.pred is not None
            else None
        )
        arg_subgraphs = [
            graph.subgraph_from_tok_ids(arg, handle_unconnected="shortest_path")
            for arg in triplet.args
        ]
        return GraphMappedTriplet(triplet, pred_subgraph, arg_subgraphs)

    def infer_triplets(self, sen: str, **kwargs) -> List[Triplet]:
        raise NotImplementedError
