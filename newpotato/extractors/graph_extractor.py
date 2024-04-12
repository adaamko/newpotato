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


class GraphBasedExtractor(Extractor):
    @staticmethod
    def from_json(data: Dict[str, Any]):
        raise NotImplementedError

    def to_json(self) -> Dict[str, Any]:
        raise NotImplementedError

    def __init__(
        self,
        parser_url: Optional[str] = "http://localhost:7277",
    ):
        super(GraphBasedExtractor, self).__init__()
        self.text_parser = GraphParserClient(parser_url)

    def parse_text(self, text: str):
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

    def get_rules(self, text_to_triplets):
        for text, triplets in text_to_triplets.items():
            pass

    def extract_triplets_from_text(self, text, **kwargs):
        raise NotImplementedError

    def map_triplet(self, triplet, sentence, **kwargs):
        graph = self.parsed_graphs[sentence]
        pred_subgraph = graph.subgraph_from_tok_ids(triplet.pred)
        arg_subgraphs = [graph.subgraph_from_tok_ids(arg) for arg in triplet.args]
        return GraphMappedTriplet(triplet, pred_subgraph, arg_subgraphs)

    def infer_triplets(self, sen: str, **kwargs) -> List[Triplet]:
        raise NotImplementedError
