import json
import logging

from rich import print
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from newpotato.hitl import HITLManager
from newpotato.utils import matches2triplets

console = Console()


class NPTerminalClient:
    def __init__(self):
        self.hitl = HITLManager()

    def load_from_file(self):
        while True:
            console.print("[bold cyan]Enter path to HITL state file:[/bold cyan]")
            fn = input("> ")
            try:
                hitl = HITLManager.load(fn)
            except FileNotFoundError:
                console.print(f"[bold red]No such file or directory: {fn}[/bold red]")
            else:
                self.hitl = hitl
                console.print(
                    f"[bold cyan]Successfully loaded HITL state from {fn}[/bold cyan]"
                )
                return

    def write_to_file(self):
        while True:
            console.print("[bold cyan]Enter path to HITL state file:[/bold cyan]")
            fn = input("> ")
            try:
                self.hitl.save(fn)
            except FileNotFoundError:
                console.print(f"[bold red] No such file or directory: {fn}[/bold red]")
            else:
                console.print(
                    f"[bold cyan]Successfully saved HITL state to {fn}[/bold cyan]"
                )
                return

    def clear_console(self):
        console.clear()

    def suggest_triplets(self):
        for sen in self.hitl.get_unannotated_sentences():
            if sen == "latest" or sen in self.hitl.text_to_triplets:
                continue
            toks = self.hitl.get_tokens(sen)
            matches = self.hitl.match_rules(sen)
            graph = self.hitl.parsed_graphs[sen]
            triplets = matches2triplets(matches, graph)
            for triplet in triplets:
                triplet_str = self.triplet_str(triplet, toks)
                console.print("[bold yellow]How about this?[/bold yellow]")
                console.print(f"[bold yellow]{sen}[/bold yellow]")
                console.print(f"[bold yellow]{triplet_str}[/bold yellow]")
                choice_str = None
                while choice_str not in ("c", "i"):
                    choice_str = input("(c)orrect or (i)ncorrect?")
                positive = True if choice_str == "c" else False
                pred, args = triplet.pred, triplet.args
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

            matches = self.hitl.match_rules(sen)

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

    def print_graphs(self):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Sentence")
        table.add_column("Graph")
        for sen, graph in self.hitl.parsed_graphs.items():
            table.add_row(sen, str(graph["main_edge"]))
        console.print(table)

    def triplet_str(self, triplet, toks):
        pred, args = triplet.pred, triplet.args
        pred_phrase = "_".join(toks[a].text for a in pred)
        args_str = ", ".join("_".join(toks[a].text for a in phrase) for phrase in args)
        return f"{pred_phrase}({args_str})"

    def get_sentence(self):
        console.print("[bold cyan]Enter new sentence:[/bold cyan]")
        sen = input("> ")
        self.hitl.get_graphs(sen)

    def upload_file(self):
        console.print("[bold cyan]Enter path of txt or jsonl file:[/bold cyan]")
        fn = input("> ")
        if fn.endswith("txt"):
            self.upload_txt(fn)
        elif fn.endswith("jsonl"):
            self.upload_jsonl(fn)
        else:
            console.print(
                "[bold red]Unknown file format, must be txt or jsonl[/bold red]"
            )

    def upload_txt(self, fn):
        console.print("[bold cyan]Parsing text...[/bold cyan]")
        with open(fn) as f:
            for line in tqdm(f):
                self.hitl.get_graphs(line.strip())
        console.print("[bold cyan]Done![/bold cyan]")

    def upload_jsonl(self, fn):
        console.print("[bold cyan]Reading jsonl...[/bold cyan]")
        with open(fn) as f:
            for i, line in tqdm(enumerate(f)):
                data = json.loads(line)
                self.hitl.store_triplets_from_annotation(data)
        console.print("[bold cyan]Done![/bold cyan]")

    def annotate(self):
        for sen in self.hitl.get_unannotated_sentences(random_order=True, max_sens=3):
            for pred, args in self.get_annotation_for_sen(sen):
                self.hitl.store_triplet(sen, pred, args)

    def get_annotation_for_sen(self, sentence):
        tokens = self.hitl.get_tokens(sentence)
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
            yield pred, args

    def run(self):
        while True:
            self.print_status()
            console.print(
                "[bold cyan]Choose an action:\n\t(S)entence\n\t(U)pload\n\t(G)raphs\n\t(A)nnotate\n\t(T)riplets\n\t(R)ules\n\t(I)nference\n\t(L)oad\n\t(W)rite\n\t(C)lear\n\t(E)xit\n\t(H)elp[/bold cyan]"
            )
            choice = input("> ").upper()
            if choice in ("T", "I") and self.hitl.extractor.classifier is None:
                console.print(
                    "[bold red]That choice requires a classifier, run (R)ules first![/bold red]"
                )
            elif choice == "S":
                self.get_sentence()
            elif choice == "U":
                self.upload_file()
            elif choice == "G":
                self.print_graphs()
            elif choice == "A":
                self.annotate()
            elif choice == "R":
                self.print_rules()
            elif choice == "T":
                self.suggest_triplets()
            elif choice == "I":
                self.classify()
            elif choice == "L":
                self.load_from_file()
            elif choice == "W":
                self.write_to_file()
            elif choice == "C":
                self.clear_console()
            elif choice == "E":
                console.print("[bold red]Exiting...[/bold red]")
                break
            elif choice == "H":
                console.print(
                    "[bold cyan]Help:[/bold cyan]\n"
                    + "\t(S)entence: Enter a new sentence to parse\n"
                    + "\t(U)pload: Upload a file with input text\n"
                    + "\t(G)raphs: Print graphs of parsed sentences\n"
                    + "\t(A)nnotate: Annotate the latest sentence\n"
                    + "\t(T)riplets: Suggest inferred triplets for sentences\n"
                    + "\t(R)ules: Extract rules from the annotated graphs\n"
                    + "\t(L)oad: Load HITL state from file\n"
                    + "\t(W)rite: Write HITL state to file\n"
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
    # logging.getLogger().setLevel(logging.DEBUG)
    client = NPTerminalClient()
    client.run()


if __name__ == "__main__":
    main()
