import argparse
import logging
import sys

from newpotato.datasets.food_disease import load_and_map_fd
from newpotato.extractors.graph_extractor import GraphBasedExtractor
from newpotato.evaluate.evaluate import Evaluator


class ExtractorEvaluator(Evaluator):
    def __init__(self, extractor, gold_data):
        super(ExtractorEvaluator, self).__init__()
        self.extractor = extractor
        self.gold_data = gold_data

    def infer_triplets(self, sen):
        preds = self.extractor.infer_triplets(sen)
        return preds

    def gen_texts_with_gold_triplets(self):
        yield from self.gold_data.items()


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input_file", default=None, type=str)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-e", "--events_file", default=None, type=str)
    parser.add_argument("-r", "--which_rel", default=None, type=str)
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
        force=True,
    )
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    extractor = GraphBasedExtractor(default_relation=args.which_rel)
    logging.info(f"loading gold data from {args.input_file=}...")
    gold_data = {
        sen: [(triplet, True) for triplet in triplets]
        for sen, triplets in load_and_map_fd(args.input_file, extractor, args.which_rel)
    }
    
    # training
    logging.info("training...")
    extractor.get_rules(gold_data)
    
    extractor.print_rules()

    # evaluation
    logging.info("evaluating...")
    evaluator = ExtractorEvaluator(extractor, gold_data)

    results = evaluator.get_results()
    for key, value in results.items():
        print(f"{key}: {value}")

    if args.events_file is not None:
        if args.events_file == "-":
            evaluator.write_events(sys.stdout)
        else:
            evaluator.write_events_to_file(args.events_file)


if __name__ == "__main__":
    main()
