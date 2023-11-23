import logging

from rich import print
from rich.console import Console
from rich.table import Table

from newpotato.hitl import HITLManager, TextParser
from newpotato.utils import matches2triplets

console = Console()


class NPTerminalClient:
    def __init__(self):
        self.parser = TextParser()
        self.hitl = HITLManager()

    def clear_console(self):
        console.clear()

    def match_rules(self, sen, graphs):
        main_graph = graphs[0]["main_edge"]
        matches, _ = self.hitl.extractor.classify(main_graph)
        return matches

    def suggest_triplets(self):
        for sen, graphs in self.hitl.parsed_graphs.items():
            if sen in self.hitl.text_to_triplets:
                continue
            toks = self.hitl.get_tokens(sen)
            matches = self.match_rules(sen, graphs)
            triplets = matches2triplets(matches, graphs[0])
            for triplet in triplets:
                triplet_str = self.triplet_str(triplet, toks)
                console.print("[bold yellow]How about this?[/bold yellow]")
                console.print(f"[bold yellow]{sen}[/bold yellow]")
                console.print(f"[bold yellow]{triplet_str}[/bold yellow]")
                choice_str = None
                while choice_str not in ("c", "i"):
                    choice_str = input("(c)orrect or (i)ncorrect?")
                positive = True if choice_str == "c" else False
                pred, args = triplet
                self.hitl.store_triplet(sen, pred, args, positive=positive)

    def classify(self):
        if not self.hitl.get_rules():
            console.print("[bold red]No rules extracted yet[/bold red]")
            return
        else:
            console.print(
                "[bold green]Classifying a sentence, please provide one:[/bold green]"
            )
            sen = input("> ")

            matches = self.match_rules(sen)

            if not matches:
                console.print("[bold red]No matches found[/bold red]")
            else:
                console.print("[bold green]Matches:[/bold green]")
                for match in matches:
                    console.print(match)

    def print_status(self):
        triplets = self.hitl.get_true_triplets()

        self.print_triplets(triplets)

    def print_rules(self):
        annotated_graphs = self.hitl.get_annotated_graphs()
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
        pred, args = triplet.pred, triplet.args
        pred_phrase = "_".join(toks[a].text for a in pred)
        args_str = ", ".join("_".join(toks[a].text for a in phrase) for phrase in args)
        return f"{pred_phrase}({args_str})"

    def get_sentence(self):
        console.print("[bold cyan]Enter new sentence:[/bold cyan]")
        sen = input("> ")
        graphs = self.parser.parse(sen)
        self.hitl.store_parsed_graphs(sen, graphs[0])

    def get_annotation(self):
        tokens = self.hitl.get_tokens("latest")
        console.print("[bold cyan]Tokens:[/bold cyan]")
        console.print(" ".join(f"{i}_{tok}" for i, tok in enumerate(tokens)))

        while True:
            console.print(
                """
                [bold cyan] Enter comma-separated list of predicate and args, with token IDs in each separated by underscores, e.g.: 0_The 1_boy 2_has 3_gone 4_to 5_school -> 2_3,0_1,4_5
                Press enter to finish.
                
                [/bold cyan]"""
            )
            annotation = input("> ")
            if annotation == "":
                break
            try:
                phrases = [
                    tuple(int(n) for n in ids.split("_"))
                    for ids in annotation.split(",")
                ]
            except ValueError:
                console.print("[bold red]Could not parse this:[/bold red]", annotation)
                continue

            pred, args = phrases[0], phrases[1:]
            self.hitl.store_triplet("latest", pred, args)

    def run(self):
        while True:
            self.print_status()
            console.print(
                "[bold cyan]Choose an action:\n\t(S)entence\n\t(A)nnotate\n\t(T)riplets\n\t(R)ules\n\t(I)nference\n\t(C)lear\n\t(E)xit\n\t(H)elp[/bold cyan]"
            )
            choice = input("> ").upper()
            if choice == "S":
                self.get_sentence()
            elif choice == "A":
                self.get_annotation()
            elif choice == "R":
                self.print_rules()
            elif choice == "T":
                self.suggest_triplets()
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
                    + "\t(T)riplets: Suggest inferred triplets for sentences\n"
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
