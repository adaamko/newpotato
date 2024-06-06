from networkx.utils import graphs_equal
from tuw_nlp.graph.ud_graph import UDGraph

from newpotato.datatypes import Triplet, GraphMappedTriplet
from newpotato.extractors.graph_extractor import GraphBasedExtractor


def test_datatypes():
    text = "It was a bright cold day in April, and the clocks were striking thirteen."
    ex = GraphBasedExtractor()
    sentence, graph = list(ex.get_graphs(text).items())[0]
    graph_dict = graph.to_json()
    new_graph = UDGraph.from_json(graph_dict)

    assert graphs_equal(new_graph.G, graph.G)

    # were_striking(clocks, thirteen)
    pred = (12, 13)
    args = ((11,), (14,))
    triplet = Triplet(pred, args)

    mapped_triplet = ex.map_triplet(triplet, sentence)
    assert isinstance(mapped_triplet, GraphMappedTriplet)

    assert isinstance(mapped_triplet.pred_graph, UDGraph)
    assert len(mapped_triplet.pred_graph.tokens) == len(graph.tokens)
    assert len(list(filter(None, mapped_triplet.pred_graph.tokens))) == 2
    assert len(mapped_triplet.pred_graph.G.nodes) == 2
    for arg_graph in mapped_triplet.arg_graphs:
        assert isinstance(arg_graph, UDGraph)
        assert len(arg_graph.tokens) == len(graph.tokens)
        assert len(list(filter(None, arg_graph.G.nodes))) == 1
