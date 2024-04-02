import argparse
import logging

# from rich import print
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from newpotato.evaluate import HITLEvaluator
from newpotato.hitl import HITLManager
from newpotato.utils import print_tokens

console = Console()


class NPTerminalClient:
    def __init__(self, args):
        if args.load_state is None:
            console.print("no state file provided, initializing new HITL")
            self.hitl = HITLManager()
        else:
            console.print(f"loading HITL state from {args.load_state}")
            self.hitl = HITLManager.load(args.load_state)

        if args.upload_file:
            self._upload_file(args.upload_file)

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
            for triplet in self.hitl.infer_triplets(sen):
                triplet_str = str(triplet)
                console.print("[bold yellow]How about this?[/bold yellow]")
                console.print(f"[bold yellow]{sen}[/bold yellow]")
                console.print(f"[bold yellow]{triplet_str}[/bold yellow]")
                choice_str = None
                while choice_str not in ("c", "i"):
                    choice_str = input("(c)orrect or (i)ncorrect?")
                positive = True if choice_str == "c" else False
                self.hitl.store_triplet(sen, triplet, positive=positive)

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
        status = self.hitl.get_status()

        status_lines = [f'{status["n_rules"]} rules', f'{status["n_sens"]} sentences']

        if status["n_sens"] > 0:
            status_lines.append(
                f'{status["n_annotated"]} of these annotated ({status["n_annotated"]/status["n_sens"]:.2%})'
            )

        console.print("\n".join(status_lines))

        triplets = self.hitl.get_true_triplets()
        self.print_triplets(triplets, max_n=10)

    def print_rules(self):
        annotated_graphs = self.hitl.get_annotated_graphs()
        rules = self.hitl.get_rules()

        console.print("[bold green]Annotated Graphs:[/bold green]")
        console.print(annotated_graphs)

        console.print("[bold green]Extracted Rules:[/bold green]")
        console.print(rules)

    def print_triplets(self, triplets_by_sen, max_n=None):
        console.print("[bold green]Current Triplets:[/bold green]")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Sentence")
        table.add_column("Triplets")

        for i, (sen, triplets) in enumerate(triplets_by_sen.items()):
            if max_n is not None and i > max_n:
                table.add_row("...", "...")
                break
            triplet_strs = [str(triplet) for triplet in triplets]
            table.add_row(sen, "\n".join(triplet_strs))

        console.print(table)

    def evaluate(self):
        evaluator = HITLEvaluator(self.hitl)
        results = evaluator.get_results()
        for key, value in results.items():
            console.print(f"{key}: {value}")

    def print_graphs(self):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Sentence")
        table.add_column("Graph")
        for sen, graph in self.hitl.parsed_graphs.items():
            table.add_row(sen, str(graph["main_edge"]))
        console.print(table)

    def get_sentence(self):
        console.print("[bold cyan]Enter new sentence:[/bold cyan]")
        sen = input("> ")
        self.hitl.get_graphs(sen)

    def _upload_file(self, fn):
        if fn.endswith("txt"):
            self.upload_txt(fn)
        else:
            console.print("[bold red]Unknown file format, must be txt[/bold red]")

    def upload_file(self):
        console.print("[bold cyan]Enter path of txt or jsonl file:[/bold cyan]")
        fn = input("> ")
        self._upload_file(fn)

    def upload_txt(self, fn):
        console.print("[bold cyan]Parsing text...[/bold cyan]")
        with open(fn) as f:
            for line in tqdm(f):
                self.hitl.get_graphs(line.strip())
        console.print("[bold cyan]Done![/bold cyan]")

    def annotate(self):
        while True:
            console.print(
                "Type the start of the sentence you would like to annotate or enter R to get random unannotated sentences. Or press ENTER to return to finish annotating and return to main menu"
            )
            raw_query = input("> ")
            query = raw_query.strip().lower()
            if not query:
                break
            if query == "r":
                for sen in self.hitl.get_unannotated_sentences(
                    random_order=True, max_sens=3
                ):
                    self.get_triplets_from_user(sen)
            else:
                cands = [
                    sen
                    for sen in self.hitl.parsed_graphs
                    if sen.lower().startswith(query)
                ]
                if len(cands) > 20:
                    console.print("more than 20 matches, please refine")
                    continue
                for i, sen in enumerate(cands):
                    console.print(f"{i}\t{sen}")

                console.print("Enter ID of the sentence you want to annotate")
                choice = input("> ")
                try:
                    sen = cands[int(choice)]
                except (ValueError, IndexError):
                    console.print("[bold red]invalid choice[/bold red]")
                else:
                    self.get_triplets_from_user(sen)

    def get_triplets_from_user(self, sentence):
        print_tokens(sentence, self.hitl, console)

        while True:
            triplet = self.get_single_triplet_from_user(sentence, self.hitl, console)
            if triplet is None:
                break
            else:
                self.hitl.store_triplet(sentence, triplet, True)

    def run(self):
        while True:
            self.print_status()
            console.print(
                "[bold cyan]Choose an action:\n\t(S)entence\n\t(U)pload\n\t(G)raphs\n\t(A)nnotate\n\t(T)riplets\n\t(R)ules\n\t(I)nference\n\t(E)valuate\n\t(L)oad\n\t(W)rite\n\t(C)lear\n\t(Q)uit\n\t(H)elp[/bold cyan]"
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
            elif choice == "E":
                self.evaluate()
            elif choice == "L":
                self.load_from_file()
            elif choice == "W":
                self.write_to_file()
            elif choice == "C":
                self.clear_console()
            elif choice == "Q":
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
                    + "\t(E)valuate: Evaluate rules on annotated sentences\n"
                    + "\t(W)rite: Write HITL state to file\n"
                    + "\t(C)lear: Clear the console\n"
                    + "\t(Q)uit: Exit the program\n"
                    + "\t(H)elp: Show this help message\n"
                )

            else:
                console.print("[bold red]Invalid choice[/bold red]")


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-l", "--load_state", default=None, type=str)
    parser.add_argument("-u", "--upload_file", default=None, type=str)
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(
        format="%(asctime)s : "
        + "%(module)s (%(lineno)s) - %(levelname)s - %(message)s"
    )
    logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    client = NPTerminalClient(args)
    client.run()


if __name__ == "__main__":
    main()
