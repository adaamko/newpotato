import argparse
import csv
import logging

from rich.console import Console

from newpotato.hitl import AnnotatedWordsNotFoundError, HITLManager
from newpotato.utils import get_triplet_from_annotation

console = Console()


def load_food_disease_dataset(input_file, hitl):
    with open(input_file, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            (
                row_id,
                food_entity,
                disease_entity,
                sentence,
                disease_doid,
                is_cause,
                is_treat,
            ) = row
            if not row_id:
                continue

            is_cause, is_treat = bool(is_cause), bool(is_treat)

            graphs = hitl.get_graphs(sentence)
            if len(graphs) > 1:
                print("sentence split into two:", sentence)
                print([graph["text"] for graph in graphs])
                raise Exception()
            sen_graph = graphs[0]
            sen = sen_graph["text"]

            if not (is_cause or is_treat):
                # no triplets to add
                continue

            pred = None
            try:
                args = [
                    hitl.get_toks_from_txt(food_entity, sen),
                    hitl.get_toks_from_txt(disease_entity, sen),
                ]
            except AnnotatedWordsNotFoundError:
                console.print(
                    f"[bold red]Could not find all words of annotation: {food_entity=}, {disease_entity=}[/bold red]"
                )
                args = None

            triplet = get_triplet_from_annotation(
                pred, args, sen, sen_graph, hitl, console
            )

            hitl.store_triplet(sen, triplet, True)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-i", "--input_file", default=None, type=str)
    parser.add_argument("-s", "--state_file", default=None, type=str)
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

    console.print("initializing HITL session")
    hitl = HITLManager()
    console.print(f"loading FoodDisease data from {args.input_file}")
    load_food_disease_dataset(args.input_file, hitl)

    console.print(f"saving HITL session to {args.state_file}")
    hitl.save(args.state_file)


if __name__ == "__main__":
    main()
