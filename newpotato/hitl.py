import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from graphbrain.hyperedge import Hyperedge, hedge
from graphbrain.learner.classifier import Classifier
from graphbrain.learner.rule import Rule
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

            # for each graph, add word2atom from atom2word
            # only storing the id of the word, not the word itself
            for graph in parses:
                # atom2word is a dict of atom: (word, word_id)
                atom2word = graph["atom2word"]

                word2atom = {word[1]: str(atom) for atom, word in atom2word.items()}
                graph["word2atom"] = word2atom

            graphs.extend(parses)

        return graphs


@dataclass
class HITLManager:
    """A class to manage the HITL process and store parsed graphs.

    Attributes:
        parsed_graphs (Dict[str, List[Dict[str, Any]]]): A dict mapping
            sentences to parsed graphs.
        annotated_graphs (Dict[str, List[Hyperedge]]): A dict mapping
            sentences to annotated graphs.
        triplets (Dict[str, List[Tuple]]): A dict mapping sentences to
            triplets.
        latest (Optional[str]): The latest sentence.
        classifier (Optional[Classifier]): The classifier that will learn the rules.
    """

    parsed_graphs: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    triplets: Dict[str, List[Tuple]] = field(default_factory=lambda: defaultdict(list))
    latest: Optional[str] = field(default=None)
    classifier: Optional[Classifier] = field(default=None)

    def is_parsed(self, text: str) -> bool:
        """
        Check if the given text is parsed.
        """

        return text in self.parsed_graphs

    def get_tokens(self, text: str) -> List[str]:
        """
        Get the tokens of the given text.
        """
        return [tok for tok in self.parsed_graphs[text][0]["spacy_sentence"]]

    def get_triplets(self):
        """
        Get the triplets.
        """

        return {
            sen: triplets for sen, triplets in self.triplets.items() if sen != "latest"
        }

    def get_rules(self) -> List[Rule]:
        """
        Get the rules.
        """
        if self.classifier is None:
            return []
        return [rule.pattern for rule in self.classifier.rules]

    def get_annotated_graphs(self) -> List[str]:
        """
        Get the annotated graphs.
        """
        assert self.classifier is not None, "classifier not initialized"
        return [str(rule[0]) for rule in self.classifier.cases]

    def annotate_graphs_with_triplets(self):
        """
        Annotate the graphs with the triplets.
        This function iterates over the triplets and adds the cases to the classifier.
        """
        classifier = Classifier()
        for text, triplets in self.triplets.items():
            if text == "latest":
                continue
            graphs = self.parsed_graphs[text]
            for graph in graphs:
                annotated_graph = graph["main_edge"]
                for triplet in triplets:
                    pred, args = triplet
                    pred_atom = graph["word2atom"][pred]
                    args_atoms = [graph["word2atom"][arg] for arg in args]
                    variables = {
                        "REL": hedge(pred_atom),
                        "ARG1": hedge(args_atoms[0]),
                        "ARG2": hedge(args_atoms[1]),
                    }

                    # positive means whether we want to treat it as a positive or negative example
                    # this helps graphbrain to learn the rules
                    classifier.add_case(
                        annotated_graph, positive=True, variables=variables
                    )

        self.classifier = classifier

    def extract_rules(self):
        """
        Extract the rules from the annotated graphs.
        """
        assert self.classifier is not None, "classifier not initialized"
        self.classifier.extract_patterns()

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
        """
        Store the triplet.

        Args:
            text (str): The text to store the triplet for.
            pred (int): The predicate.
            args (List[int]): The arguments.
        """

        if text == "latest":
            assert (
                self.latest is not None
            ), "no parsed graphs stored, can't use `latest`"
            return self.store_triplet(self.latest, pred, args)
        assert self.is_parsed(text), f"unparsed text: {text}"
        logging.info(f"appending to triplets: {pred}, {args}")
        self.triplets[text].append((pred, args))
