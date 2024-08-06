import argparse
import logging
import os
from collections import defaultdict

from rich.console import Console
from tuw_nlp.common.utils import ensure_dir

from newpotato.datasets.food_disease import load_and_map_fd
from newpotato.datasets.lsoie import load_and_map_lsoie
from newpotato.extractors.graph_extractor import GraphBasedExtractor
from newpotato.evaluate.evaluate import Evaluator
from newpotato.hitl import HITLManager


class ExtractorEvaluator(Evaluator):
    def __init__(self, extractor, gold_data):
        super(ExtractorEvaluator, self).__init__()
        self.extractor = extractor
        self.gold_data = gold_data

    def infer_triplets(self, sen):
        preds = self.extractor.infer_triplets(sen)
        return preds

    def gen_texts_with_gold_triplets(self):
        for sen, triplet_list in self.gold_data.items():
            yield sen, [triplet for triplet, is_true in triplet_list if is_true]


def split_data(gold_dict, test_size=None):
    gold_items = list(gold_dict.items())
    test_size = len(gold_items) // 10 if test_size is None else test_size
    train_items, val_items = gold_items[:-test_size], gold_items[-test_size:]
    return dict(train_items), dict(val_items)


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--data_type", default=None, type=str)
    parser.add_argument("-s", "--test_size", default=None, type=int)
    parser.add_argument("-i", "--input_file", default=None, type=str)
    parser.add_argument("-l", "--load_state", default=None, type=str)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o", "--log_dir", default=None, type=str)
    parser.add_argument("-r", "--which_rel", default=None, type=str)
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
        force=True,
    )
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    ensure_dir(args.log_dir)
    print(f'logging to directory {args.log_dir}')

    console = Console()
    if args.load_state is None:
        extractor = GraphBasedExtractor(default_relation=args.which_rel)
        logging.warning(f"loading gold data from {args.input_file=}...")
        if args.data_type == "fd":
            gold_data = {
                sen: [(triplet, True) for triplet in triplets]
                for sen, triplets in load_and_map_fd(
                    args.input_file, extractor, args.which_rel
                )
            }
        elif args.data_type == "lsoie":
            gold_data = defaultdict(list)
            for sen, triplet in load_and_map_lsoie(args.input_file, extractor):
                gold_data[sen].append((triplet, True))
        else:
            raise ValueError(f"unknown data type: {args.data_type}")
    else:
        logging.warning(f"loading data from HITL state file {args.load_state}...")
        hitl = HITLManager.load(args.load_state)
        extractor = hitl.extractor
        gold_data = hitl.text_to_triplets

    train_data, val_data = split_data(gold_data, args.test_size)
    logging.warning(f'{len(train_data)=}, {len(val_data)=}')
    # training
    logging.warning("training...")
    extractor.get_rules(train_data)

    if args.verbose:
        extractor.print_rules(console)

    # evaluation
    logging.warning("evaluating...")
    evaluator = ExtractorEvaluator(extractor, val_data)

    results = evaluator.get_results()

    results_file = os.path.join(args.log_dir, 'results.txt')
    logging.warning(f"writing results to {results_file}...")
    with open(results_file, 'w') as rf:
        for key, value in results.items():
            line = f"{key}: {value}"
            print(line)
            rf.write(f'{line}\n')

    events_file = os.path.join(args.log_dir, 'events.tsv')
    logging.warning(f"writing events to {events_file}...")
    evaluator.write_events_to_file(events_file)


if __name__ == "__main__":
    main()
