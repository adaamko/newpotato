import json
import logging
import random
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import spacy
from fastcoref import spacy_component
from graphbrain.hyperedge import Hyperedge
from graphbrain.learner.classifier import Classifier
from graphbrain.learner.classifier import from_json as classifier_from_json
from graphbrain.learner.rule import Rule
from graphbrain.parsers import create_parser

from newpotato.datatypes import GraphParse, Triplet
from newpotato.utils import get_variables, matches2triplets

assert spacy_component  # silence flake8


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
class TextParser:
    """A class to handle text parsing using Graphbrain."""

    @staticmethod
    def from_params(params: Dict[str, Any]):
        if params is None:
            return TextParser()
        else:
            return TextParser(**params)

    def __init__(self, lang: str = "en", corefs: bool = True):
        self.lang = lang
        self.corefs = corefs
        self.init_parser()

    def init_parser(self):
        self.parser = create_parser(lang=self.lang)
        if self.corefs:
            self.coref_nlp = spacy.load(
                "en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"]
            )
            self.coref_nlp.add_pipe("fastcoref")

    def get_params(self) -> Dict[str, Any]:
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
    def from_json(classifier_data: Dict[str, Any]):
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

    def extract_rules(self, learn: bool = False):
        """
        Extract the rules from the annotated graphs.
        """
        assert self.classifier is not None, "classifier not initialized"
        if learn:
            self.classifier.learn()
        else:
            self.classifier.extract_patterns()
            self.classifier._index_rules()

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
            parsed_graphs (Dict[str, Dict[str, Any]]): The parsed graphs.
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
            logging.debug(f"{self.classifier.rules=}")
            logging.debug(f"{rule_ids_triggered=}")
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
        parser (TextParser): The text parser that parses text into graphs.
        parser_params (Dict): parameters to be used to initialize a TextParser object
    """

    def __init__(
        self,
        parsed_graphs: Dict[str, Dict[str, Any]] = None,
        triplets: Dict[str, List[Tuple]] = None,
        latest: Optional[str] = None,
        extractor_data: Optional[Extractor] = None,
        parser_params: Optional[Dict[str, Any]] = None,
        parser: Optional[TextParser] = None,
    ):
        self.parsed_graphs = {} if parsed_graphs is None else parsed_graphs
        self.text_to_triplets = (
            defaultdict(list) if triplets is None else defaultdict(list, triplets)
        )
        self.latest = latest

        if extractor_data is None:
            self.extractor = Extractor()
        else:
            self.extractor = Extractor.from_json(extractor_data)

        if parser is None:
            self.text_parser = TextParser.from_params(parser_params)
        else:
            assert (
                parser_params is None
            ), "parser and parser_params cannot both be specified"
            self.text_parser = parser

    @staticmethod
    def load(fn):
        with open(fn) as f:
            data = json.load(f)
        return HITLManager.from_json(data)

    @staticmethod
    def from_json(data: Dict[str, Any]):
        """
        load HITLManager from saved state

        Args:
            data (dict): the saved state, as returned by the to_json function

        Returns:
            HITLManager: a new HITLManager object with the restored state
        """
        parser = TextParser.from_params(data["parser_params"])
        spacy_vocab = parser.parser.nlp.vocab
        parsed_graphs = {
            text: GraphParse.from_json(graph_dict, spacy_vocab)
            for text, graph_dict in data["parsed_graphs"].items()
        }
        triplets = {
            text: [(Triplet.from_json(triplet[0]), triplet[1]) for triplet in triplets]
            for text, triplets in data["triplets"].items()
        }

        hitl = HITLManager(
            parsed_graphs=parsed_graphs,
            triplets=triplets,
            extractor_data=data["extractor_data"],
            parser=parser,
        )
        return hitl

    def to_json(self) -> Dict[str, Any]:
        """
        get the state of the HITLManager so that it can be saved

        Returns:
            dict: a dict with all the HITLManager object's attributes that are relevant to
                its state
        """

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
            "n_sens": len(self.parsed_graphs),
            "n_annotated": len(self.text_to_triplets),
            "n_rules": n_rules,
        }

    def parse_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse the given text.

        Args:
            text (str): The text to parse.

        Returns:
            List[Dict[str, Any]]: The parsed graphs.
        """
        return self.text_parser.parse(text)

    def get_rules(self, learn: bool = False) -> List[Rule]:
        """
        Get the rules.

        Args:
            learn (bool): whether to run graphbrain classifier's learn function.
                If False (default), only extract_patterns is called
        """

        _ = self.get_annotated_graphs()
        self.extractor.extract_rules(learn=learn)

        return self.extractor.get_rules()

    def infer_triplets(self, sen: str) -> List[Triplet]:
        """
        match rules against sentence and return triplets corresponding to the matches

        Args:
            sen (str): the sentence to perform inference on

        Returns:
            List[Triple]: list of triplets inferred
        """
        logging.debug(f'inferring triplets for: "{sen}"')
        graph = self.parsed_graphs[sen]
        logging.debug(f'graph: "{graph}"')
        matches = self.match_rules(sen)
        logging.debug(f'matches: "{matches}"')
        triplets = matches2triplets(matches, graph)
        logging.debug(f'triplets: "{triplets}"')

        return triplets

    def triplets_to_str(self, triplets: List[Triplet], sen: str) -> List[str]:
        """
        Returns human-readable versions of triplets for a sentence

        Args:
            triplets (List[Triplet]): the triplets to convert
            sen (str): the sentence that is the source of this triplet

        Returns:
            List[str]: the human-readable form of the triplet
        """
        return [self.triplet_to_str(triplet, sen) for triplet in triplets]

    def triplet_to_str(self, triplet: Triplet, sen: str) -> str:
        """
        Returns a human-readable version of a triplet by retrieving the words and phrases from the sentence

        Args:
            triplet (Triplet): the triplet to display
            sen (str): the sentence that is the source of this triplet

        Returns:
            str: the human-readable form of the triplet
        """
        pred, args = triplet.pred, triplet.args
        toks = self.get_tokens(sen)
        pred_phrase = "_".join(toks[a].text for a in pred)
        args_str = ", ".join("_".join(toks[a].text for a in phrase) for phrase in args)
        return f"{pred_phrase}({args_str})"

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

    def store_triplets_from_annotation(self, data: Dict[str, Any]):
        """
        Store triplets from annotation.

        Args:
            data (dict): A dictionary with the keys "sen" and "triplets".
                Triplet annotations must be provided as a list of dictionaries with the keys
                "rel" and "args", each of which must be a substring of the sentence.
        """
        sen, triplets = self.get_triplets_from_annotation(data)
        for pred, args in triplets:
            self.store_triplet(sen, pred, args)

    def get_triplets_from_annotation(self, data: Dict[str, Any]):
        """
        Get annotated triplets.

        Args:
            data (dict): A dictionary with the keys "sen" and "triplets".
                Triplet annotations must be provided as a list of dictionaries with the keys
                "rel" and "args", each of which must be a substring of the sentence.
        Returns:
            Tuple[str, List[Tuple[Tuple, List[Tuple]]]: the sentence (after parsing)
                and the list of triplets, as required by store_triplet
        """
        logging.debug(f"getting triplets from annotation: {data}")
        graphs = self.get_graphs(data["sen"])
        if len(graphs) > 1:
            print("sentence split into two:", data["sen"])
            print([graph["text"] for graph in graphs])
            raise Exception()
        sen = graphs[0]["text"]
        triplets = []
        for triplet in data["triplets"]:
            try:
                pred = self.get_toks_from_txt(triplet["rel"], sen)
                args = [
                    self.get_toks_from_txt(arg_txt, sen) for arg_txt in triplet["args"]
                ]
                triplets.append((pred, args))
            except AnnotatedWordsNotFoundError as e:
                logging.warning(f"skipping triplet: {e}")

        return sen, triplets

    def get_toks_from_txt(self, words_txt: str, sen: str) -> Tuple[int, ...]:
        """
        Map a substring of a sentence to its tokens. Used to parse annotations of triplets
        provided as plain text strings of the predicate and the arguments

        Args:
            words_txt (str): the substring of the sentence
            sen (str): the sentence

        Returns:
            Tuple[int, ...] the tokens of the sentence corresponding to the substring
        """
        logging.debug(f"words_txt: {words_txt}, sen: {sen}")

        pattern = re.escape(re.sub("[()]", "", words_txt))
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
            for sen in self.parsed_graphs
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

    def match_rules(self, sen: str) -> List[Dict]:
        """
        match rules against sentence by passing the sentence's graph to the extractor

        Args:
            sen (str): the sentence to be matched against

        Returns:
            List[Dict] a list of hypergraphs corresponding to the matches
        """
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
