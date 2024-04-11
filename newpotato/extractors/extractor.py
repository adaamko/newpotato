from typing import Any, Dict, List


from newpotato.datatypes import Triplet


def get_extractor_cls(e_type):
    if e_type == "graphbrain":
        from newpotato.extractors.graphbrain_extractor import GraphbrainExtractor
        return GraphbrainExtractor
    else:
        raise ValueError(f"unknown extractor type: {e_type}")


class Extractor:
    """Abstract class for all extractors"""

    @staticmethod
    def from_json(data: Dict[str, Any]):
        cls = get_extractor_cls(data["extractor_type"])
        return cls.from_json(data)

    def __init__(self):
        self.parsed_graphs = {}

    def to_json(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_tokens(self, sen) -> List[str]:
        raise NotImplementedError

    def get_sentences(self, text: str) -> List[str]:
        return [sen for sen, _ in self.get_graphs(text)]

    def get_rules(self, *args, **kwargs):
        raise NotImplementedError
    
    def is_parsed(self, text):
        """
        Check if the given text is parsed.
        """
        return text in self.parsed_graphs
   
    def extract_triplets_from_text(self, text, **kwargs):
        raise NotImplementedError

    def parse_text(self, text, **kwargs):
        raise NotImplementedError

    def get_graph(self, sen: str):
        """
        Get graph for sentence that is already parsed

        Args:
            sen (str): the sentence to get the graphs for
        """

        if not self.is_parsed(sen):
            raise ValueError("get_graph can only be called for parsed sentences")

        return self.parsed_graphs[sen]

    def get_graphs(self, text: str) -> List:
        """
        Get graphs for text, parsing it if necessary

        Args:
            text (str): the text to get the graphs for
            graphs (Dict[str, Any]): the graphs corresponding to the sentences of the text
        """
        if text in self.parsed_graphs:
            return {text: self.parsed_graphs[text]}

        graphs = {}
        for sen, graph in self.parse_text(text):
            graphs[sen] = graph
            self.parsed_graphs[sen] = graph
            self.latest = sen
            self.parsed_graphs["latest"] = graph

        return graphs

    def map_triplet(self, triplet, sentence, **kwargs):
        raise NotImplementedError

    def infer_triplets(self, sen: str, **kwargs) -> List[Triplet]:
        raise NotImplementedError
