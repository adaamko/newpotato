import argparse
import csv
import logging

from rich.console import Console

from newpotato.hitl import HITLManager
from newpotato.utils import (
    AnnotatedWordsNotFoundError,
    get_triplet_from_annotation,
    get_toks_from_txt,
)

console = Console()


def untokenize(entity):
    out = entity.replace(" ,", ",")
    out = out.replace("( ", "(")
    out = out.replace(" )", ")")
    return out


def load_fd(input_file):
    with open(input_file, newline="") as csvfile:
        fixed_lines = [line.replace(", ", "| ") for line in csvfile]

    reader = csv.reader(fixed_lines)
    for orig_row in reader:
        row = [field.replace("| ", ", ") for field in orig_row]
        logging.debug(f"{row=}")
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

        yield row_id, sentence, food_entity, disease_entity, is_cause, is_treat


def load_and_map_fd(input_file, extractor, which_rel):
    assert which_rel in ("CAUSE", "TREAT")
    for row_id, sentence, food_entity, disease_entity, is_cause, is_treat in load_fd(
        input_file
    ):
        text_to_graph = list(extractor.parse_text(sentence))
        if len(text_to_graph) > 1:
            logging.error(f"sentence split into two: {sentence}")
            logging.error(f"{text_to_graph=}")
            logging.error("skipping")
            continue

        sen, graph = text_to_graph[0]
        stanza_sen = graph.stanza_sen

        if which_rel == "CAUSE" and not is_cause:
            # no triplets to add
            continue
        elif not is_treat:
            # no triplets to add
            continue

        pred = None

        try:
            args = [
                get_toks_from_txt(untokenize(food_entity), stanza_sen),
                get_toks_from_txt(untokenize(disease_entity), stanza_sen),
            ]
        except AnnotatedWordsNotFoundError:
            logging.warning(
                f"Could not find all words of annotation: {food_entity=}, {disease_entity=}"
            )
            logging.warning("skipping")
            continue

        logging.debug(f"FoodDisease args from text: {args=}")

        triplet = get_triplet_from_annotation(
            pred, args, sen, graph, extractor, console, ask_user=False
        )
        if not triplet.mapped:
            logging.warning(f"unmapped triplet: {row_id=}, {sen=}, {args=}")

        yield sentence, [triplet]


def load_fd_to_hitl(input_file, hitl, which_rel):
    for sen, triplets in load_and_map_fd(input_file, hitl.extractor, which_rel):
        for triplet in triplets:
            hitl.store_triplet(sen, triplet, True)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-i", "--input_file", default=None, type=str)
    parser.add_argument("-s", "--state_file", default=None, type=str)
    parser.add_argument("-r", "--which_rel", default=None, type=str)
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(
        format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print("initializing HITL session")
    hitl = HITLManager()
    console.print(f"loading FoodDisease data from {args.input_file}")
    load_fd_to_hitl(args.input_file, hitl, args.which_rel)

    console.print(f"saving HITL session to {args.state_file}")
    hitl.save(args.state_file)


if __name__ == "__main__":
    main()
