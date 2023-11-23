from rich.console import Console

from newpotato.hitl import HITLManager, TextParser
from newpotato.utils import matches2triplets


def test_subedge2toks():
    parser = TextParser()
    hitl = HITLManager()
    console = Console()
    text1 = "John loves Mary"
    graph1 = parser.parse(text1)[0]
    sen1 = graph1["text"]
    hitl.store_parsed_graphs(sen1, graph1)

    pred, args = (1,), [(0,), (2,)]
    hitl.store_triplet(sen1, pred, args)

    annotated_graphs = hitl.get_annotated_graphs()
    rules = hitl.get_rules()

    console.print("[bold green]Annotated Graphs:[/bold green]")
    console.print(annotated_graphs)

    console.print("[bold green]Extracted Rules:[/bold green]")
    console.print(rules)

    text2 = "Mary loves John"
    graph2 = parser.parse(text2)[0]
    sen2 = graph2["text"]
    hitl.store_parsed_graphs(sen2, graph2)

    matches, _ = hitl.extractor.classify(graph2["main_edge"])
    triplets = matches2triplets(matches, graph2)
    print("triplets:", triplets)
    assert triplets[0].pred == (1,)
    assert triplets[0].args == [(0,), (2,)]


if __name__ == "__main__":
    test_subedge2toks()
