import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fastcoref import spacy_component
import spacy

from graphbrain.hyperedge import Hyperedge
from graphbrain.learner.classifier import Classifier
from graphbrain.learner.rule import Rule
from graphbrain.parsers import create_parser

from newpotato.utils import get_variables


@dataclass
class TextParser:
    """A class to handle text parsing using Graphbrain."""

    def __init__(self, lang: str = "en", parser: Optional[Any] = None):
        self.lang = lang
        self.parser = parser

        self.coref_nlp = spacy.load(
            "en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"]
        )
        self.coref_nlp.add_pipe("fastcoref")

    def resolve_coref(self, text: str) -> str:
        """
        Run coreference resolution and return text with resolved coreferences

        Args:
            text (str): The text to resolve

        Returns:
            str: The resolved text
        """

        doc = self.coref_nlp(text, component_cfg={"fastcoref": {"resolve_text": True}})
        return doc._.resolved_text

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
            resolved_text = self.resolve_coref(paragraph)
            parses = self.parser.parse(resolved_text)["parses"]

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
            words = [tok.text for tok in self.get_tokens(text)]
            for graph in graphs:
                main_edge = graph["main_edge"]
                annotated_graph = graph["main_edge"]
                for triplet in triplets:
                    variables = get_variables(main_edge, words, triplet)
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
        # self.classifier.extract_patterns()
        self.classifier.learn()

    def store_parsed_graphs(self, text: str, parsed_graphs: List[Dict[str, Any]]):
        """
        Store the parsed graphs.

        Args:
            parsed_graphs (List[Dict[str, Any]]): The parsed graphs to store.
        """
        self.latest = text
        self.parsed_graphs["latest"] = parsed_graphs
        self.parsed_graphs[text] = parsed_graphs

    def store_triplet(self, text: str, pred: Tuple[int, ...], args: List[Tuple[int, ...]]):
        """
        Store the triplet.

        Args:
            text (str): The text to store the triplet for.
            pred (Tuple[int, ...]): The predicate.
            args (List[Tuple[int, ...]]): The arguments.
        """

        if text == "latest":
            assert (
                self.latest is not None
            ), "no parsed graphs stored, can't use `latest`"
            return self.store_triplet(self.latest, pred, args)
        assert self.is_parsed(text), f"unparsed text: {text}"
        logging.info(f"appending to triplets: {pred}, {args}")
        self.triplets[text].append((pred, args))

    def classify(self, graph: Hyperedge) -> List[Dict[str, Any]]:
        """
        Classify the graph.

        Args:
            graph (Hyperedge): The graph to classify.

        Returns:
            List[Dict[str, Any]]: The matches in a format of [{"REL": "pred", "ARG1": "arg1", "ARG2": "arg2"}]
        """
        assert self.classifier is not None, "classifier not initialized"
        matches = self.classifier.classify(graph)
        logging.info(f"classifier matches: {matches}")

        return matches
