import logging

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from graphbrain.parsers import create_parser


@dataclass
class TextParser:
    """A class to handle text parsing using Graphbrain."""

    lang: str = "en"
    parser: Optional[Any] = field(default=None, init=False)

    def parse(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse the given text using Graphbrain and return the parsed edges.

        Args:
            text (str): The text to parse.

        Returns:
            List[Dict[str, Any]]: The parsed edges.
        """
        if not self.parser:
            self.parser = create_parser(lang=self.lang)
        paragraphs = text.split("\n\n")
        graphs = []

        for paragraph in paragraphs:
            parses = self.parser.parse(paragraph)["parses"]
            graphs.extend(parses)

        return graphs


@dataclass
class HITLManager:
    """A class to manage the HITL process and store parsed graphs."""

    parsed_graphs: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    triplets: Dict[str, List[Tuple]] = field(default_factory=lambda: defaultdict(list))
    latest: str = None

    def is_parsed(self, text: str) -> bool:
        return text in self.parsed_graphs

    def get_tokens(self, text: str) -> List[str]:
        return [tok for tok in self.parsed_graphs[text][0]["spacy_sentence"]]

    def get_patterns(self):
        return []

    def store_parsed_graphs(self, text: str, parsed_graphs: List[Dict[str, Any]]):
        """
        Store the parsed graphs.

        Args:
            parsed_graphs (List[Dict[str, Any]]): The parsed graphs to store.
        """
        self.latest = text
        self.parsed_graphs["latest"] = parsed_graphs
        self.parsed_graphs[text] = parsed_graphs

    def store_triplet(self, text: str, pred: int, args: List[int]):
        if text == "latest":
            assert self.latest is not None, "no parsed graphs stored, can't use `latest`"
            return self.store_triplet(self.latest, pred, args)
        assert self.is_parsed(text), f"unparsed text: {text}"
        logging.info(f'appending to triplets: {pred}, {args}')
        self.triplets[text].append((pred, args))

    def get_triplets(self):
        return {
            sen: triplets for sen, triplets in self.triplets.items() if sen != "latest"
        }
