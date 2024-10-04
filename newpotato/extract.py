import argparse
import logging
import sys

from newpotato.datatypes import triplets_to_str
from newpotato.extractors.graph_extractor import GraphBasedExtractor


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input_file", default=None, type=str)
    parser.add_argument("-p", "--patterns_file", default=None, type=str)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o", "--output_file", default=None, type=str)
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

    input_stream = sys.stdin if args.input_file is None else open(args.input_file)
    extractor = GraphBasedExtractor()
    extractor.load_patterns(args.patterns_file)
    for line in input_stream:
        triplets = extractor.infer_triplets(line)
        print(" ".join(triplets_to_str(triplets)))


if __name__ == "__main__":
    main()
