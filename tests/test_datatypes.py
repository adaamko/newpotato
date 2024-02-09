from newpotato.parser import TextParserClient
from newpotato.datatypes import GraphParse, Triplet


def test_graph():
    input_text = (
        "It was a bright cold day in April, and the clocks were striking thirteen."
    )
    parser = TextParserClient()
    graph = parser.parse(input_text)[0]
    graph_dict = graph.to_json()
    vocab = parser.get_vocab()
    reloaded_graph = GraphParse.from_json(graph_dict, vocab)
    assert graph["text"] == reloaded_graph["text"]

    def atom2str(d):
        return {a.to_str(): value for a, value in d.items()}

    assert atom2str(graph["atom2word"]) == atom2str(reloaded_graph["atom2word"])

    rereloaded_graph = GraphParse.from_json(reloaded_graph.to_json(), vocab)
    assert atom2str(reloaded_graph["atom2word"]) == atom2str(
        rereloaded_graph["atom2word"]
    )


def test_triplet():
    pred = (1, 2, 3)
    args = ((4, 5), (6, 7))
    triplet = Triplet(pred, args)
    assert triplet.pred == pred
    assert triplet.args == args
