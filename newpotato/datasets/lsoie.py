import argparse
import logging
from collections import defaultdict

from rich.console import Console
from tqdm import tqdm
from tuw_nlp.text.utils import gen_tsv_sens

from newpotato.datatypes import Triplet
from newpotato.hitl import HITLManager

console = Console()


def load_and_map_lsoie(input_file, extractor):
    with open(input_file) as stream:
        for sen_idx, sen in tqdm(enumerate(gen_tsv_sens(stream))):
            sen_str = " ".join(t[1] for t in sen)
            text_to_graph = list(extractor.parse_text(sen_str))
            if len(text_to_graph) > 1:
                logging.error(f"sentence split into two: {sen_str}")
                logging.error(f"{text_to_graph=}")
                logging.error("skipping")
                continue

            sentence, graph = text_to_graph[0]

            arg_dict = defaultdict(list)
            pred = []
            for i, tok in enumerate(sen):
                label = tok[7].split("-")[0]
                if label == "O":
                    continue
                elif label == "P":
                    pred.append(i)
                    continue
                arg_dict[label].append(i)

            pred = tuple(pred)
            args = [
                tuple(indices)
                for label, indices in sorted(
                    arg_dict.items(), key=lambda i: int(i[0][1:])
                )
            ]
            logging.debug(f"{pred=}, {args=}")

            triplet = Triplet(pred, args, toks=graph.tokens)
            mapped_triplet = extractor.map_triplet(triplet, sentence)
            yield sentence, mapped_triplet


def load_lsoie_to_hitl(input_file, hitl):
    for sen, triplet in load_and_map_lsoie(input_file, hitl.extractor):
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
        format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
        force=True,
    )
    logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    console.print("initializing HITL session")
    hitl = HITLManager(extractor_type="ud")
    console.print(f"loading LSOIE data from {args.input_file}")
    load_lsoie_to_hitl(args.input_file, hitl)

    console.print(f"saving HITL session to {args.state_file}")
    hitl.save(args.state_file)


if __name__ == "__main__":
    main()
