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

    def to_json(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_tokens(self) -> List[str]:
        raise NotImplementedError

    def get_sentences(self) -> List[str]:
        raise NotImplementedError

    def get_rules(self, *args, **kwargs):
        raise NotImplementedError
   
    def map_triplet(self, triplet, sentence, **kwargs):
        raise NotImplementedError

    def infer_triplets(self, sen: str, **kwargs) -> List[Triplet]:
        raise NotImplementedError
