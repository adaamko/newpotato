import logging

from newpotato.hitl import HITLManager, TextParser


class NPTerminalClient:
    def __init__(self):
        self.parser = TextParser()
        self.hitl = HITLManager()

    def print_status(self):
        patterns = self.hitl.get_patterns()
        triplets = self.hitl.get_triplets()
        print(f"current patterns: {patterns}")
        self.print_triplets(triplets)

    def print_triplets(self, triplets_by_sen):
        print("current triplets:")
        for sen, triplets in triplets_by_sen.items():
            toks = self.hitl.get_tokens(sen)
            print("sen:", sen)
            print("triplets:")
            print("\n".join(self.triplet_str(triplet, toks) for triplet in triplets))
            print()

    def triplet_str(self, triplet, toks):
        pred, args = triplet
        args_str = ", ".join(toks[a].text for a in args)
        return f"{toks[pred]}({args_str})"

    def get_sentence(self):
        print("enter new sentence")
        sen = input("> ")
        graphs = self.parser.parse(sen)
        self.hitl.store_parsed_graphs(sen, graphs)

    def get_annotation(self):
        print(
            " ".join(
                f"{i}_{tok}" for i, tok in enumerate(self.hitl.get_tokens("latest"))
            )
        )
        print()
        while True:
            print("Enter comma-separated list of token IDs like this: PRED,ARG1,ARG2")
            print("(choose the most important word for each)")
            print("Press ENTER if there are no more triplets")
            annotation = input("> ")
            if annotation == "":
                break
            try:
                numbers = [int(n.strip()) for n in annotation.split(",")]
            except ValueError:
                print(f'could not parse this: {annotation}')
                continue

            pred, args = numbers[0], numbers[1:]
            self.hitl.store_triplet("latest", pred, args)

    def run(self):
        while True:
            self.print_status()
            print(
                "choose an action:\n\tadd new (S)entence\n\t(A)nnotate last sentence\n"
            )
            choice = input("> ")
            if choice == "S":
                self.get_sentence()
            elif choice == "A":
                self.get_annotation()
            else:
                continue


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
