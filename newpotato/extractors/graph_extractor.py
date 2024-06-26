import logging
from collections import Counter, defaultdict
from itertools import chain
from typing import Any, Dict, List, Optional

from newpotato.datatypes import GraphMappedTriplet, Triplet
from newpotato.extractors.extractor import Extractor
from newpotato.extractors.graph_parser_client import GraphParserClient

from tuw_nlp.graph.ud_graph import UDGraph
from tuw_nlp.graph.utils import GraphFormulaPatternMatcher


class GraphBasedExtractor(Extractor):
    @staticmethod
    def from_json(data: Dict[str, Any]):
        extractor = GraphBasedExtractor()
        extractor.text_parser.check_params(data["parser_params"])

        extractor.parsed_graphs = {
            text: UDGraph.from_json(graph_dict)
            for text, graph_dict in data["parsed_graphs"].items()
        }

        return extractor

    def to_json(self) -> Dict[str, Any]:
        # TODO learned rules are not yet saved
        data = {
            "extractor_type": "graph",
            "parsed_graphs": {
                text: graph.to_json() for text, graph in self.parsed_graphs.items()
            },
            "parser_params": self.text_parser.get_params(),
        }

        return data

    def __init__(
        self,
        parser_url: Optional[str] = "http://localhost:7277",
        default_relation: Optional[str] = None,
    ):
        super(GraphBasedExtractor, self).__init__()
        self.text_parser = GraphParserClient(parser_url)
        self.default_relation = default_relation
        self.n_rules = 0

    def _parse_text(self, text: str):
        """
        Parse the given text.

        Args:
            text (str): The text to parse.

        Returns:
            TODO
        """
        graphs = self.text_parser.parse(text)
        for graph in graphs:
            yield graph.text, graph

    def get_tokens(self, sen) -> List[str]:
        """
        Get the tokens of the given text.
        """
        return self.parsed_graphs[sen].tokens

    def get_lemmas(self, sen) -> List[str]:
        """
        Get the lemmas of the given text.
        """
        return [w.lemma for w in self.parsed_graphs[sen].stanza_sen.words]

    def get_rules(self, text_to_triplets, **kwargs):
        pred_graphs = Counter()
        triplet_graphs = Counter()
        arg_graphs = defaultdict(Counter)
        for text, triplets in text_to_triplets.items():
            # toks = self.get_tokens(text)
            logging.debug(f"{text=}")
            graph = self.parsed_graphs[text]
            logging.debug(graph.to_dot())
            lemmas = self.get_lemmas(text)
            for triplet, positive in triplets:
                logging.debug(f"{triplet=}")
                if triplet.pred is not None:
                    logging.debug(f"{triplet.pred_graph=}")
                    pred_graphs[triplet.pred_graph] += 1
                    pred_lemmas = tuple(lemmas[i] for i in triplet.pred)
                    triplet_toks = set(chain(triplet.pred, *triplet.args))
                else:
                    pred_lemmas = (self.default_relation,)
                    triplet_toks = set(chain(*triplet.args))

                for arg_graph in triplet.arg_graphs:
                    logging.debug(f"{arg_graph=}")
                    arg_graphs[pred_lemmas][arg_graph] += 1

                logging.debug(f"{triplet_toks=}")

                triplet_graph = graph.subgraph(
                    triplet_toks, handle_unconnected="shortest_path"
                )
                triplet_graphs[triplet_graph] += 1

                logging.debug(f"{triplet_graph=}")

                if triplet.pred is None:
                    inferred_pred_toks = set(
                        node
                        for node in triplet_graph.G.nodes()
                        if node not in triplet_toks
                    )
                    logging.debug(f"{inferred_pred_toks=}")
                    inferred_pred_graph = graph.subgraph(
                        inferred_pred_toks, handle_unconnected="shortest_path"
                    )
                    logging.debug(f"{inferred_pred_graph=}")
                    pred_graphs[inferred_pred_graph] += 1

        self.pred_graphs = pred_graphs
        self.arg_graphs = arg_graphs
        self.triplet_graphs = triplet_graphs
        self.n_rules = len(self.pred_graphs)

        patterns = []
        threshold = 1
        label = "LABEL"
        for pred_graph, freq in self.pred_graphs.most_common():
            if freq < threshold:
                break
            patterns.append(((pred_graph.G,), (), label))

        self.pred_matcher = GraphFormulaPatternMatcher(
            patterns, converter=None, case_sensitive=False
        )
        self._is_trained = True
        return [graph for graph, freq in self.pred_graphs.most_common(20)]

    def print_rules(self, console):
        console.print("[bold green]Extracted Rules:[/bold green]")
        console.print(f"{self.pred_graphs=}")
        console.print(f"{self.arg_graphs=}")
        console.print(f"{self.triplet_graphs=}")

    def get_n_rules(self):
        return self.n_rules

    def extract_triplets_from_text(self, text, **kwargs):
        matches_by_text = {}
        for sen, triplets_and_subgraphs in self._infer_triplets(text):
            matches_by_text[sen] = {
                "matches": [],
                "rules_triggered": [],
                "triplets": [],
            }
            for triplet, subgraph in triplets_and_subgraphs:
                matches_by_text[sen]["rules_triggered"].append(subgraph)
                matches_by_text[sen]["triplets"].append(triplet)
                matches_by_text[sen]["matches"].append(
                    {"REL": None, "ARG0": None, "ARG1": None}
                )

        return matches_by_text

    def map_triplet(self, triplet, sentence, **kwargs):
        graph = self.parsed_graphs[sentence]
        print(f"mapping triplet: {triplet.pred=}, {triplet.args=}")
        logging.debug(f"mapping triplet to {graph=}")
        pred_subgraph = (
            graph.subgraph(triplet.pred, handle_unconnected="shortest_path")
            if triplet.pred is not None
            else None
        )

        logging.debug(f"triplet mapped: {pred_subgraph=}")

        arg_subgraphs = [
            graph.subgraph(arg, handle_unconnected="shortest_path")
            if len(arg) > 0
            else None
            for arg in triplet.args
        ]
        logging.debug(f"triplet mapped: {arg_subgraphs=}")
        return GraphMappedTriplet(triplet, pred_subgraph, arg_subgraphs)

    def _infer_triplets(self, text: str):
        for sen, sen_graph in self.parse_text(text):
            triplets_and_subgraphs = []
            for key, i, subgraphs in self.pred_matcher.match(
                sen_graph.G, return_subgraphs=True
            ):
                for subgraph in subgraphs:
                    logging.debug(f"MATCH: {sen=}")
                    logging.debug(f"MATCH: {subgraph.graph=}")
                    ud_subgraph = sen_graph.subgraph(subgraph.nodes)
                    pred_indices = tuple(
                        idx
                        for idx, token in enumerate(ud_subgraph.tokens)
                        if token is not None
                    )
                    triplet = Triplet(pred_indices, ())
                    mapped_triplet = self.map_triplet(triplet, sen)
                    triplets_and_subgraphs.append((mapped_triplet, ud_subgraph))
            yield sen, triplets_and_subgraphs

    def infer_triplets(self, text: str, **kwargs) -> List[Triplet]:
        triplets = sorted(
            set(
                triplet
                for sen, triplets_and_subgraphs in self._infer_triplets(text)
                for triplet, _ in triplets_and_subgraphs
            )
        )
        return triplets
