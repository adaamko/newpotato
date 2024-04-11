import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple


from newpotato.datatypes import Triplet
from newpotato.extractors.extractor import Extractor
from newpotato.extractors.graph_extractor import GraphBasedExtractor


class AnnotatedWordsNotFoundError(Exception):
    def __init__(self, words_txt, pattern, sen):
        message = (
            f'Words "{words_txt}" (pattern: "{pattern}") not found in sentence "{sen}"'
        )
        super().__init__(message)

        self.words_txt = words_txt
        self.sen = sen
        self.pattern = pattern


@dataclass
class HITLManager:
    """A class to manage the HITL process and store parsed graphs.

    Attributes:
        parsed_graphs (Dict[str, Dict[str, Any]]): A dict mapping
            sentences to parsed graphs.
        annotated_graphs (Dict[str, List[Hyperedge]]): A dict mapping
            sentences to annotated graphs.
        triplets (Dict[str, List[Tuple]]): A dict mapping sentences to
            triplets.
        latest (Optional[str]): The latest sentence.
        extractor (Extractor): The extractor that uses classifiers to extract triplets from graphs.
    """

    def __init__(self, parser_url: Optional[str] = "http://localhost:7277"):
        self.latest = None
        self.sentences = {}
        self.text_to_triplets = defaultdict(list)
        self.oracle = None
        self.extractor = GraphBasedExtractor()
        logging.info("HITL manager initialized")

    def load_extractor(self, extractor_data):
        self.extractor = Extractor.from_json(extractor_data)

    def load_data(self, sentences, triplet_data, oracle=False):
        self.sentences = sentences
        text_to_triplets = {
            text: [
                (
                    Triplet.from_json(triplet[0]),
                    triplet[1],
                )
                for triplet in triplets
            ]
            for text, triplets in triplet_data.items()
        }

        if oracle:
            self.oracle = text_to_triplets
            self.text_to_triplets = defaultdict(list)
        else:
            self.text_to_triplets = defaultdict(list, text_to_triplets)

    @staticmethod
    def load(fn, oracle=False):
        logging.info(f"loading HITL state from {fn=}")
        with open(fn) as f:
            data = json.load(f)
        return HITLManager.from_json(data, oracle=oracle)

    @staticmethod
    def from_json(
        data: Dict[str, Any], parser_url="http://localhost:7277", oracle=False
    ):
        """
        load HITLManager from saved state

        Args:
            data (dict): the saved state, as returned by the to_json function

        Returns:
            HITLManager: a new HITLManager object with the restored state
        """
        hitl = HITLManager(parser_url)
        hitl.load_data(data["sentences"], data["triplets"], oracle=oracle)
        hitl.load_extractor(data["extractor_data"])
        return hitl

    def to_json(self) -> Dict[str, Any]:
        """
        get the state of the HITLManager so that it can be saved

        Returns:
            dict: a dict with all the HITLManager object's attributes that are relevant to
                its state
        """

        return {
            "sentences": self.sentences,
            "triplets": {
                text: [(triplet[0].to_json(), triplet[1]) for triplet in triplets]
                for text, triplets in self.text_to_triplets.items()
            },
            "extractor_data": self.extractor.to_json(),
        }

    def save(self, fn: str):
        """
        save HITLManager state to a file

        Args:
            fn (str): path of the file to be written (will be overwritten if it exists)
        """
        with open(fn, "w") as f:
            f.write(json.dumps(self.to_json()))

    def get_status(self) -> Dict[str, Any]:
        """
        return basic stats about the HITL state
        """
        n_rules = 0
        if self.extractor.classifier is not None:
            n_rules = len(self.extractor.classifier.rules)

        return {
            "n_sens": len(self.sentences),
            "n_annotated": len(self.text_to_triplets),
            "n_rules": n_rules,
        }

    def add_text(self, text: str):
        self.sentences.update(
            {
                sen: self.extractor.get_tokens(sen)
                for sen in self.extractor.get_sentences(text)
            }
        )

    def get_rules(self, *args, **kwargs):
        return self.extractor.get_rules(self.text_to_triplets, *args, **kwargs)

    def print_rules(self, console):
        return self.extractor.print_rules(console)

    def infer_triplets(self, sen: str, **kwargs) -> List[Triplet]:
        return self.extractor.infer_triplets(sen, **kwargs)

    def triplets_to_str(self, triplets: List[Triplet], sen: str) -> List[str]:
        """
        Returns human-readable versions of triplets for a sentence

        Args:
            triplets (List[Triplet]): the triplets to convert
            sen (str): the sentence that is the source of this triplet

        Returns:
            List[str]: the human-readable form of the triplet
        """
        return [str(triplet) for triplet in triplets]

    def get_true_triplets(self) -> Dict[str, List[Triplet]]:
        """
        Get the triplets, return everything except the latest triplets.

        Returns:
            Dict[str, List[Triplet]]: The triplets.
        """

        return {
            sen: [triplet for triplet, positive in triplets if positive is True]
            for sen, triplets in self.text_to_triplets.items()
            if sen != "latest"
        }

    def delete_triplet(self, text: str, triplet: Triplet):
        """
        Delete the triplet.

        Args:
            text (str): the text to delete the triplet for.
            triplet (Triplet): the triplet to delete
        """

        if text == "latest":
            assert (
                self.latest is not None
            ), "no parsed graphs stored, can't use `latest`"
            return self.delete_triplet(self.latest, triplet)
        assert self.is_parsed(text), f"unparsed text: {text}"
        logging.info(f"deleting from triplets: {text=}, {triplet=}")
        self.text_to_triplets[text].remove((triplet, True))

    def store_triplet(
        self,
        text: str,
        triplet: Triplet,
        positive=True,
    ):
        """
        Store the triplet.

        Args:
            text (str): the text to store the triplet for.
            triplet (Triplet): the triplet to store
            positive (bool): whether to store the triplet as a positive (default) or negative
                example
        """

        if text == "latest":
            assert (
                self.latest is not None
            ), "no parsed graphs stored, can't use `latest`"
            return self.store_triplet(self.latest, triplet, positive)
        logging.info(f"appending to triplets: {text=}, {triplet=}")
        self.text_to_triplets[text].append((triplet, positive))

    def get_toks_from_txt(
        self, words_txt: str, sen: str, ignore_brackets: bool = False
    ) -> Tuple[int, ...]:
        """
        Map a substring of a sentence to its tokens. Used to parse annotations of triplets
        provided as plain text strings of the predicate and the arguments

        Args:
            words_txt (str): the substring of the sentence
            sen (str): the sentence
            ignore_brackets (bool): whether to remove brackets from the text before matching (required for ORE annotation)

        Returns:
            Tuple[int, ...] the tokens of the sentence corresponding to the substring
        """
        logging.debug(f"{words_txt=}, {sen=}")
        if ignore_brackets:
            pattern = re.escape(re.sub('["()]', "", words_txt))
        else:
            pattern = re.escape(words_txt)
        logging.debug(f"{pattern=}")
        if pattern[0].isalpha():
            pattern = r"\b" + pattern
        if pattern[-1].isalpha():
            pattern = pattern + r"\b"
        m = re.search(pattern, sen, re.IGNORECASE)

        if m is None:
            raise AnnotatedWordsNotFoundError(words_txt, pattern, sen)

        start, end = m.span()
        logging.debug(f"span: {(start, end)}")

        tok_i, tok_j = None, None
        tokens = self.get_tokens(sen)
        logging.debug(f"tokens: {tokens}")
        logging.debug(f"tok idxs: {[tok.idx for tok in tokens]}")
        for i, token in enumerate(tokens):
            if token.idx == start:
                tok_i = i
            if token.idx >= end:
                tok_j = i
                break
        if tok_i is None:
            logging.error(
                f'left side of annotation "{words_txt}" does not match the left side of any token in sen "{sen}"'
            )
            raise Exception()
        if tok_j is None:
            tok_j = len(tokens)

        return tuple(range(tok_i, tok_j))

    def get_unannotated_sentences(
        self, max_sens: Optional[int] = None, random_order: bool = False
    ) -> Generator[str, None, None]:
        """
        get a list of sentences that have been added and parsed but not yet annotated

        Args:
            max_sens (int): the maximum number of sentences to return. If None (this is the
                default) or larger than the total number of unannotated sentences, all
                unannotated sentences are returned
            random_order (bool): if False (default), sentences are yielded in the order of
                the self.parsed_graphs dict, which is always the same. If True, a random
                sample is generated, with a new random seed on each function call.

        Returns:
            Generator[str] the unannotated sentences
        """
        sens = [
            sen
            for sen in self.sentences
            if sen != "latest" and sen not in self.text_to_triplets
        ]
        n_graphs = len(sens)
        max_n = min(max_sens, n_graphs) if max_sens is not None else n_graphs

        if random_order:
            random.seed()
            logging.debug(f"sampling {max_n} indices from {n_graphs}")
            indices = set(random.sample(range(n_graphs), max_n))
            logging.debug(f"sample indices: {indices}")
            yield from (
                sen for i, sen in enumerate(sens) if i in indices and sen != "latest"
            )
        else:
            yield from sens[:max_n]
