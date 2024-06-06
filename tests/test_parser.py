from tuw_nlp.graph.ud_graph import UDGraph

from newpotato.extractors.graph_parser_client import GraphParserClient


def test_parser():
    parser_url = "http://localhost:7277"
    input_text = (
        "It was a bright cold day in April, and the clocks were striking thirteen."
    )
    parser = GraphParserClient(parser_url)
    graph = parser.parse(input_text)[0]
    # assert parse_text(input_text) == {"status": "ok"}
    assert isinstance(graph, UDGraph)
    assert graph.text == input_text


if __name__ == "__main__":
    print(test_parser())
