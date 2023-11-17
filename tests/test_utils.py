from rich.console import Console

from newpotato.hitl import HITLManager, TextParser
from newpotato.utils import matches2triplets


def test_subedge2toks():
    parser = TextParser()
    hitl = HITLManager()
    console = Console()
    sen1 = "John loves Mary"
    graphs1 = parser.parse(sen1)
    hitl.store_parsed_graphs(sen1, graphs1)

    pred, args = (1,), [(0,), (2,)]
    hitl.store_triplet(sen1, pred, args)
    
    annotated_graphs = hitl.get_annotated_graphs()
    rules = hitl.get_rules()

    console.print("[bold green]Annotated Graphs:[/bold green]")
    console.print(annotated_graphs)

    console.print("[bold green]Extracted Rules:[/bold green]")
    console.print(rules)

    sen2 = "Mary loves John"
    graphs2 = parser.parse(sen2)
    hitl.store_parsed_graphs(sen2, graphs2)

    matches, _ = hitl.extractor.classify(graphs2[0]['main_edge'])
    triplets = matches2triplets(matches, graphs2[0])
    print('triplets:', triplets)


if __name__ == "__main__":
    test_subedge2toks()
