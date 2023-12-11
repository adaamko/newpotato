import json
import logging
import random

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import spacy
from fastcoref import spacy_component
from graphbrain.hyperedge import Hyperedge
from graphbrain.learner.classifier import Classifier
from graphbrain.learner.classifier import from_json as classifier_from_json
from graphbrain.learner.rule import Rule
from graphbrain.parsers import create_parser

from newpotato.datatypes import GraphParse, Triplet
from newpotato.utils import get_variables


@dataclass
class TextParser:
    """A class to handle text parsing using Graphbrain."""

    @staticmethod
    def from_params(params):
        if params is None:
            return TextParser()
        else:
            return TextParser(**params)

    def __init__(self, lang: str = "en", corefs: bool = True):
        self.lang = lang
        self.parser = create_parser(lang=self.lang)
        self.corefs = corefs

        if corefs:
            self.coref_nlp = spacy.load(
                "en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"]
            )
            self.coref_nlp.add_pipe("fastcoref")

    def get_params(self):
        return {"lang": self.lang, "corefs": self.corefs}

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

        paragraphs = text.split("\n\n")
        graphs = []

        for paragraph in paragraphs:
            resolved_text = self.resolve_coref(paragraph) if self.corefs else paragraph

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
class Extractor:
    """A class to extract triplets from graphs, texts, and annotated graphs.

    Attributes:
        classifier (Optional[Classifier]): The classifier to use for extraction.
    """

    classifier: Optional[Classifier] = field(default=None)

    @staticmethod
    def from_json(classifier_data):
        extractor = Extractor()
        extractor.classifier = classifier_from_json(classifier_data)
        return extractor

    def to_json(self) -> Dict[str, List]:
        if self.classifier is None:
            return None
        return self.classifier.to_json()

    def get_rules(self) -> List[Rule]:
        """
        Get the rules.
        """
        if self.classifier is None:
            return []
        return [rule.pattern for rule in self.classifier.rules]

    def extract_rules(self):
        """
        Extract the rules from the annotated graphs.
        """
        assert self.classifier is not None, "classifier not initialized"
        # self.classifier.extract_patterns()
        self.classifier.learn()

    def get_annotated_graphs_from_classifier(self) -> List[str]:
        """
        Get the annotated graphs

        Returns:
            List[str]: The annotated graphs. An annotated graph is a hyperedge that has been annotated with variables. e.g. "REL(ARG1, ARG2)"
        """
        assert self.classifier is not None, "classifier not initialized"
        return [str(rule[0]) for rule in self.classifier.cases]

    def add_cases(
        self,
        parsed_graphs: Dict[str, Dict[str, Any]],
        text_to_triplets: Dict[str, List[Triplet]],
    ):
        """
        Add cases to the classifier.

        Args:
            parsed_graphs (List[Dict[str, Any]]): The parsed graphs.
            triplets (List[Tuple]): The triplets.
        """
        classifier = Classifier()
        for text, triplets in text_to_triplets.items():
            graph = parsed_graphs[text]
            words = [tok.text for tok in graph["spacy_sentence"]]
            annotated_graph = graph["main_edge"]
            for triplet, positive in triplets:
                variables = get_variables(annotated_graph, words, triplet)
                logging.info("adding case:")
                logging.info(f"graph: {annotated_graph}")
                logging.info(
                    f"triplet: {triplet}, variables: {variables}, positive: {positive}"
                )

                # positive means whether we want to treat it as a positive or negative example
                # this helps graphbrain to learn the rules
                classifier.add_case(
                    annotated_graph, positive=positive, variables=variables
                )

        self.classifier = classifier

    def classify(self, graph: Hyperedge) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Classify the graph.

        Args:
            graph (Hyperedge): The graph to classify.

        Returns:
            Tuple[List[Dict[str, Any]], List[str]]: The matches and the rules triggered.
        """
        assert self.classifier is not None, "classifier not initialized"

        try:
            matches = self.classifier.classify(graph)
            rule_ids_triggered = self.classifier.rules_triggered(graph)
            rules_triggered = [
                str(self.classifier.rules[rule_id - 1].pattern)
                for rule_id in rule_ids_triggered
            ]
        except AttributeError as err:
            logging.error(f"Graphbrain classifier threw exception:\n{err}")
            matches, rules_triggered = [], []

        logging.info(f"classifier matches: {matches}")
        logging.info(f"classifier rules triggered: {rules_triggered}")

        return matches, rules_triggered


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
        text_parser (TextParser): The text parser that parses text into graphs.
    """

    def __init__(
        self,
        parsed_graphs=None,
        triplets=None,
        latest=None,
        extractor_data=None,
        parser_params=None,
    ):
        self.parsed_graphs = {} if parsed_graphs is None else parsed_graphs
        self.text_to_triplets = defaultdict(list) if triplets is None else triplets
        self.latest = latest
        self.extractor = Extractor(extractor_data)
        self.text_parser = TextParser.from_params(parser_params)

    @staticmethod
    def load(fn):
        with open(fn) as f:
            data = json.load(f)
        return HITLManager.from_json(data)

    @staticmethod
    def from_json(hitl_data):
        return HITLManager(**hitl_data)

    def to_json(self):

        return {
            "parsed_graphs": {
                text: graph.to_json() for text, graph in self.parsed_graphs.items()
            },
            "triplets": {
                text: [(triplet[0].to_json(), triplet[1]) for triplet in triplets]
                for text, triplets in self.text_to_triplets.items()
            },
            "extractor_data": self.extractor.to_json(),
            "parser_params": self.text_parser.get_params(),
        }

    def save(self, fn):
        with open(fn, "w") as f:
            f.write(json.dumps(self.to_json()))

    def parse_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse the given text.

        Args:
            text (str): The text to parse.

        Returns:
            List[Dict[str, Any]]: The parsed graphs.
        """
        return self.text_parser.parse(text)

    def get_rules(self) -> List[Rule]:
        """
        Get the rules.
        """

        _ = self.get_annotated_graphs()
        self.extractor.extract_rules()

        return self.extractor.get_rules()

    def get_annotated_graphs(self) -> List[str]:
        """
        Get the annotated graphs.
        """

        self.extractor.add_cases(self.parsed_graphs, self.text_to_triplets)

        return self.extractor.get_annotated_graphs_from_classifier()

    def add_text_to_graphs(self, text: str) -> None:
        """Add the given text to the graphs.

        Args:
            text (str): The text to add to the graphs.

        Returns:
            None
        """
        self.get_graphs(text)

    def is_parsed(self, text: str) -> bool:
        """
        Check if the given text is parsed.
        """

        return text in self.parsed_graphs

    def get_tokens(self, text: str) -> List[str]:
        """
        Get the tokens of the given text.
        """
        return [tok for tok in self.parsed_graphs[text]["spacy_sentence"]]

    def get_true_triplets(
        self,
    ) -> Dict[str, List[Triplet]]:
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

    def get_graphs(self, text: str) -> List[Dict[str, Any]]:
        """
        Get graphs for text, parsing it if necessary

        Args:
            text (str): the text to get the graphs for
            graphs (List[Dict[str, Any]]): the graphs corresponding to the text
        """
        if text in self.parsed_graphs:
            return [self.parsed_graphs[text]]

        graphs = self.parse_text(text)
        for graph in graphs:
            self.latest = text
            self.parsed_graphs[graph["text"]] = GraphParse(graph)
            self.parsed_graphs["latest"] = GraphParse(graph)

        return graphs

    def store_triplet(
        self,
        text: str,
        pred: Tuple[int, ...],
        args: List[Tuple[int, ...]],
        positive=True,
    ):
        """
        Store the triplet.

        Args:
            text (str): The text to store the triplet for.
            pred (Tuple[int, ...]): The predicate.
            args (List[Tuple[int, ...]]): The arguments.
            positive (bool): whether to store the triplet as a positive (default) or negative
                example
        """

        if text == "latest":
            assert (
                self.latest is not None
            ), "no parsed graphs stored, can't use `latest`"
            return self.store_triplet(self.latest, pred, args)
        assert self.is_parsed(text), f"unparsed text: {text}"
        logging.info(f"appending to triplets: {pred}, {args}")
        self.text_to_triplets[text].append((Triplet(pred, args), positive))

    def get_unannotated_sentences(self, max_sens=None, random_order=False):
        n_graphs = len(self.parsed_graphs)
        max_n = min(max_sens, n_graphs) if max_sens is not None else n_graphs

        if random_order:
            random.seed()
            logging.debug(f"sampling {max_n} indices from {n_graphs}")
            indices = set(random.sample(range(n_graphs), max_n))
            logging.debug(f"sample indices: {indices}")
            yield from (
                sen
                for i, sen in enumerate(self.parsed_graphs.keys())
                if i in indices and sen != "latest"
            )
        else:
            yield from list(self.parsed_graphs.keys())[:max_n]

    def match_rules(self, sen):
        graph = self.parsed_graphs[sen]
        main_graph = graph["main_edge"]
        matches, _ = self.extractor.classify(main_graph)
        return matches

    def extract_triplets_from_text(
        self, text: str, convert_to_text: bool = False
    ) -> Dict[str, Any]:
        """
        Extract the triplets from the given text with the Extractor.
        First the text is parsed into graphs, then the graphs are classified by the Extractor.

        Args:
            text (str): The text to extract triplets from.

        Returns:
            Dict[str, Any]: The matches and rules triggered. The matches are a list of dicts, where each dict is a triplet. The rules triggered are a list of strings, where each string is a rule.
        """

        graphs = self.get_graphs(text)
        matches_by_text = {graph["text"]: {} for graph in graphs}

        for graph in graphs:
            matches, rules_triggered = self.extractor.classify(graph["main_edge"])
            if convert_to_text:
                matches = [
                    {k: v.label() for k, v in match.items()} for match in matches
                ]

            logging.info(f"matches: {matches}")

            matches_by_text[graph["text"]]["matches"] = matches
            matches_by_text[graph["text"]]["rules_triggered"] = rules_triggered

        return matches_by_text
