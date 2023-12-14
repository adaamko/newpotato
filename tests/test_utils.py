from rich.console import Console

from newpotato.hitl import HITLManager
from newpotato.utils import matches2triplets


def test_subedge2toks():
    hitl = HITLManager()
    console = Console()
    sen1 = "John loves Mary"
    hitl.get_graphs(sen1)[0]
    pred, args = (1,), [(0,), (2,)]
    hitl.store_triplet(sen1, pred, args)

    annotated_graphs = hitl.get_annotated_graphs()
    rules = hitl.get_rules(learn=True)

    console.print("[bold green]Annotated Graphs:[/bold green]")
    console.print(annotated_graphs)

    console.print("[bold green]Extracted Rules:[/bold green]")
    console.print(rules)

    sen2 = "Mary loves John"
    graph2 = hitl.get_graphs(sen2)[0]
    matches = hitl.match_rules(sen2)
    triplets = matches2triplets(matches, graph2)
    print("triplets:", triplets)
    assert triplets[0].pred == (1,)
    assert triplets[0].args == [(0,), (2,)]


if __name__ == "__main__":
    test_subedge2toks()
