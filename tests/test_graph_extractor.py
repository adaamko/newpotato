from newpotato.extractors.graph_extractor import GraphBasedExtractor
from newpotato.datatypes import Triplet


def test_graph_extractor():
    text = "John loves Mary"
    ex = GraphBasedExtractor()
    ex.get_graphs(text)
    toks = ex.get_tokens(text)
    triplet = Triplet((2,), ((1,), (3,)), toks=toks)
    mapped_triplet = ex.map_triplet(triplet, text)
    print(f"{mapped_triplet=}")
    text_to_triplets = {text: [mapped_triplet]}
    ex.get_rules(text_to_triplets)
    ex.print_rules()


if __name__ == "__main__":
    test_graph_extractor()
