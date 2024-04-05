import argparse
import json
import logging
from typing import Any, Dict

from rich.console import Console
from tqdm import tqdm

from newpotato.hitl import AnnotatedWordsNotFoundError, HITLManager
from newpotato.utils import get_triplet_from_annotation

console = Console()


def get_triplets_from_annotation(data: Dict[str, Any], hitl: HITLManager):
    """
    Get and store annotated triplets from ORE annotated data

    Args:
        data (dict): A dictionary with the keys "sen" and "triplets".
            Triplet annotations must be provided as a list of dictionaries with the keys
            "rel" and "args", each of which must be a substring of the sentence.
        hitl (HITLManager): the HITL session to which the data shall be added
    """
    sen = data["sen"]
    graphs = hitl.get_graphs(sen)
    if len(graphs) > 1:
        print("sentence split into two:", data["sen"])
        print([graph["text"] for graph in graphs])
        raise Exception()
    sen_graph = graphs[0]
    sen = sen_graph["text"]
    for text_triplet in data["triplets"]:
        try:
            pred = hitl.get_toks_from_txt(
                text_triplet["rel"], sen, ignore_brackets=True
            )
            args = [
                hitl.get_toks_from_txt(arg_txt, sen, ignore_brackets=True)
                for arg_txt in text_triplet["args"]
            ]
        except AnnotatedWordsNotFoundError:
            console.print(
                f"[bold red]Could not find all words of annotation: pred={text_triplet['rel']}, args={text_triplet['args']}[/bold red]"
            )
            pred = None
            args = None

        triplet = get_triplet_from_annotation(pred, args, sen, sen_graph, hitl, console)

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
    console.print(f"loading ORE data from {args.input_file}")
    with open(args.input_file) as f:
        for i, line in tqdm(enumerate(f)):
            data = json.loads(line)
            get_triplets_from_annotation(data, hitl)

    console.print(f"saving HITL session to {args.state_file}")
    hitl.save(args.state_file)


if __name__ == "__main__":
    main()
