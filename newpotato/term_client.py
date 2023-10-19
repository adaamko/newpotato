import logging

from rich import print
from rich.console import Console
from rich.table import Table

from newpotato.hitl import HITLManager, TextParser

console = Console()


class NPTerminalClient:
    def __init__(self):
        self.parser = TextParser()
        self.hitl = HITLManager()

    def clear_console(self):
        console.clear()

    def classify(self):
        if not self.hitl.get_rules():
            console.print("[bold red]No rules extracted yet[/bold red]")
            return
        else:
            console.print(
                "[bold green]Classifying a sentence, please provide one:[/bold green]"
            )
            sen = input("> ")
            graphs = self.parser.parse(sen)

            main_graph = graphs[0]["main_edge"]

            matches = self.hitl.classify(main_graph)

            if not matches:
                console.print("[bold red]No matches found[/bold red]")
            else:
                console.print("[bold green]Matches:[/bold green]")
                for match in matches:
                    console.print(match)

    def print_status(self):
        triplets = self.hitl.get_triplets()

        self.print_triplets(triplets)

    def print_rules(self):
        self.hitl.annotate_graphs_with_triplets()
        annotated_graphs = self.hitl.get_annotated_graphs()
        self.hitl.extract_rules()
        rules = self.hitl.get_rules()

        console.print("[bold green]Annotated Graphs:[/bold green]")
        console.print(annotated_graphs)

        console.print("[bold green]Extracted Rules:[/bold green]")
        console.print(rules)

    def print_triplets(self, triplets_by_sen):
        console.print("[bold green]Current Triplets:[/bold green]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Sentence")
        table.add_column("Triplets")

        for sen, triplets in triplets_by_sen.items():
            toks = self.hitl.get_tokens(sen)
            triplet_strs = [self.triplet_str(triplet, toks) for triplet in triplets]
            table.add_row(sen, "\n".join(triplet_strs))

        console.print(table)

    def triplet_str(self, triplet, toks):
        pred, args = triplet
        args_str = ", ".join(toks[a].text for a in args)
        return f"{toks[pred]}({args_str})"

    def get_sentence(self):
        console.print("[bold cyan]Enter new sentence:[/bold cyan]")
        sen = input("> ")
        graphs = self.parser.parse(sen)
        self.hitl.store_parsed_graphs(sen, graphs)

    def get_annotation(self):
        tokens = self.hitl.get_tokens("latest")
        console.print("[bold cyan]Tokens:[/bold cyan]")
        console.print(" ".join(f"{i}_{tok}" for i, tok in enumerate(tokens)))

        while True:
            console.print(
                """
                [bold cyan]Enter comma-separated list of token IDs like this: PRED,ARG1,ARG2
                (choose the most important word for each)
                Press enter to finish.
                
                [/bold cyan]"""
            )
            annotation = input("> ")
            if annotation == "":
                break
            try:
                numbers = [int(n.strip()) for n in annotation.split(",")]
            except ValueError:
                console.print("[bold red]Could not parse this:[/bold red]", annotation)
                continue

            pred, args = numbers[0], numbers[1:]
            self.hitl.store_triplet("latest", pred, args)

    def run(self):
        while True:
            self.print_status()
            console.print(
                "[bold cyan]Choose an action:\n\t(S)entence\n\t(A)nnotate\n\t(R)ules\n\t(I)nference\n\t(C)lear\n\t(E)xit\n\t(H)elp[/bold cyan]"
            )
            choice = input("> ").upper()
            if choice == "S":
                self.get_sentence()
            elif choice == "A":
                self.get_annotation()
            elif choice == "R":
                self.print_rules()
            elif choice == "I":
                self.classify()
            elif choice == "C":
                self.clear_console()
            elif choice == "E":
                console.print("[bold red]Exiting...[/bold red]")
                break
            elif choice == "H":
                console.print(
                    "[bold cyan]Help:[/bold cyan]\n"
                    + "\t(S)entence: Enter a new sentence to parse\n"
                    + "\t(A)nnotate: Annotate the latest sentence\n"
                    + "\t(R)ules: Extract rules from the annotated graphs\n"
                    + "\t(C)lear: Clear the console\n"
                    + "\t(E)xit: Exit the program\n"
                    + "\t(H)elp: Show this help message\n"
                )
            else:
                console.print("[bold red]Invalid choice[/bold red]")


def main():
    logging.basicConfig(
        format="%(asctime)s : "
        + "%(module)s (%(lineno)s) - %(levelname)s - %(message)s"
    )
    logging.getLogger().setLevel(logging.INFO)
    client = NPTerminalClient()
    client.run()


if __name__ == "__main__":
    main()
