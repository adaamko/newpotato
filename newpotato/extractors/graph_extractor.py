from typing import Any, Dict, List, Optional

from newpotato.datatypes import Triplet
from newpotato.extractors.extractor import Extractor
from newpotato.extractors.graph_parser_client import GraphParserClient


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

    def get_rules(self, *args, **kwargs):
        raise NotImplementedError

    def extract_triplets_from_text(self, text, **kwargs):
        raise NotImplementedError

    def map_triplet(self, triplet, sentence, **kwargs):
        raise NotImplementedError

    def infer_triplets(self, sen: str, **kwargs) -> List[Triplet]:
        raise NotImplementedError
