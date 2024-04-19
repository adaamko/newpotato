import argparse
import logging
import sys

from newpotato.evaluate.evaluate import Evaluator
from newpotato.hitl import HITLManager


class HITLEvaluator(Evaluator):
    def __init__(self, hitl):
        super(HITLEvaluator, self).__init__()
        self.hitl = hitl

    def infer_triplets(self, sen):
        preds = self.hitl.infer_triplets(sen)
        return preds

    def gen_texts_with_gold_triplets(self):
        for sen in self.hitl.text_to_triplets:
            golds = set([triplet for triplet, _ in self.hitl.text_to_triplets[sen]])
            yield sen, golds


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-r", "--relearn", action="store_true")
    parser.add_argument("hitl_state_file")
    parser.add_argument("-e", "--events_file", default=None, type=str)
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

    hitl = HITLManager.load(args.hitl_state_file)
    if args.relearn:
        hitl.get_rules(learn=False)
    evaluator = HITLEvaluator(hitl)
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
