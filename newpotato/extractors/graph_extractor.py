import json
import logging
from collections import Counter, defaultdict
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from tuw_nlp.graph.ud_graph import UDGraph
from tuw_nlp.graph.utils import GraphFormulaPatternMatcher
from tuw_nlp.text.utils import tuple_if_list

from newpotato.datatypes import GraphMappedTriplet, Triplet
from newpotato.extractors.extractor import Extractor
from newpotato.extractors.graph_parser_client import GraphParserClient


class GraphBasedExtractor(Extractor):
    @staticmethod
    def from_json(data: Dict[str, Any]):
        extractor = GraphBasedExtractor()
        extractor.text_parser.check_params(data["parser_params"])

        extractor.parsed_graphs = {
            tuple_if_list(item["text"]): UDGraph.from_json(item["graph"])
            for item in data["parsed_graphs"]
        }

        return extractor

    def to_json(self) -> Dict[str, Any]:
        # TODO learned rules are not yet saved
        return {
            "extractor_type": "graph",
            "parsed_graphs": [
                {"text": text, "graph": graph.to_json()}
                for text, graph in self.parsed_graphs.items()
            ],
            "parser_params": self.text_parser.get_params(),
        }

    def save(self, fn: str):
        with open(fn, "w") as f:
            f.write(json.dumps(self.to_json()))

    def _triplet_patterns_to_json(self, graphs):
        return [
            {
                "pattern": pattern[0].to_penman(),
                "arg_roots": pattern[1],
                "inferred_nodes": pattern[2],
                "count": count,
            }
            for pattern, count in graphs.items()
        ]

    def _patterns_to_json(self, graphs):
        return {graph.to_penman(): count for graph, count in graphs.most_common()}

    def patterns_to_json(self):
        return {
            "pred_graphs": self._patterns_to_json(self.pred_graphs),
            "all_arg_graphs": self._patterns_to_json(self.all_arg_graphs),
            "arg_graphs_by_pred": {
                " ".join(pred): self._patterns_to_json(arg_graphs)
                for pred, arg_graphs in self.arg_graphs_by_pred.items()
            },
            "triplet_graphs": self._triplet_patterns_to_json(self.triplet_graphs),
            "triplet_graphs_by_pred": {
                " ".join(pred): self._triplet_patterns_to_json(tr_graphs)
                for pred, tr_graphs in self.triplet_graphs_by_pred.items()
            },
        }

    def save_patterns(self, fn: str):
        with open(fn, "w") as f:
            f.write(json.dumps(self.patterns_to_json(), indent=4))

    def __init__(
        self,
        parser_url: Optional[str] = "http://localhost:7277",
        default_relation: Optional[str] = None,
    ):
        super(GraphBasedExtractor, self).__init__()
        self.text_parser = GraphParserClient(parser_url)
        self.default_relation = default_relation
        self.n_rules = 0

    def _parse_sen_tuple(self, sen_tuple: Tuple):
        """
        Parse pretokenized sentence.

        Args:
            sen_tuple (Tuple): The pretokenized sentence.

        Returns:
            TODO
        """
        graph = self.text_parser.parse_pretokenized(sen_tuple)
        return sen_tuple, graph

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

    def _get_patterns(self, text_to_triplets):
        pred_graphs = Counter()
        triplet_graphs = Counter()
        triplet_graphs_by_pred = defaultdict(Counter)
        arg_graphs_by_pred = defaultdict(Counter)
        all_arg_graphs = Counter()
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
                    triplet_toks = set(chain(triplet.pred, triplet.arg_roots))
                else:
                    pred_lemmas = (self.default_relation,)
                    triplet_toks = set(triplet.arg_roots)

                for arg_graph in triplet.arg_graphs:
                    logging.debug(f"{arg_graph=}")
                    arg_graphs_by_pred[pred_lemmas][arg_graph] += 1
                    all_arg_graphs[arg_graph] += 1

                logging.debug(f"{triplet_toks=}")

                triplet_graph = graph.subgraph(
                    triplet_toks, handle_unconnected="shortest_path"
                )

                # the list of arg roots is stored to map nodes to arguments
                # inferred nodes are stored so they can be ignored at matching time
                # both are stored by lextop indices
                pattern_key = (
                    triplet_graph,
                    tuple(triplet_graph.index_nodes(triplet.arg_roots)),
                    tuple(triplet_graph.index_inferred_nodes()),
                )
                triplet_graphs[pattern_key] += 1
                triplet_graphs_by_pred[pred_lemmas][pattern_key] += 1

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
        self.all_arg_graphs = all_arg_graphs
        self.arg_graphs_by_pred = arg_graphs_by_pred
        self.triplet_graphs = triplet_graphs
        self.triplet_graphs_by_pred = triplet_graphs_by_pred

    def _get_matcher_from_graphs(self, graphs, label, threshold):
        patterns = []
        for graph, freq in graphs.most_common():
            if freq < threshold:
                break
            if isinstance(graph, UDGraph):
                patterns.append(((graph.G,), (), label))
            else:
                patterns.append(((graph[0].G,), (), label))

        matcher = GraphFormulaPatternMatcher(
            patterns, converter=None, case_sensitive=False
        )
        return matcher

    def _get_triplet_matchers(self):
        return Counter(
            {
                (
                    self._get_matcher_from_graphs(
                        Counter({graph: count}), label="TRI", threshold=1
                    ),
                    arg_root_indices,
                    inferred_node_indices,
                ): count
                for (
                    graph,
                    arg_root_indices,
                    inferred_node_indices,
                ), count in self.triplet_graphs.most_common()
            }
        )

    def _get_triplet_matchers_by_pred(self):
        return {
            pred_lemmas: Counter(
                {
                    (
                        self._get_matcher_from_graphs(
                            Counter({graph: count}), label="TRI", threshold=1
                        ),
                        arg_root_indices,
                        inferred_node_indices,
                    ): count
                    for (
                        graph,
                        arg_root_indices,
                        inferred_node_indices,
                    ), count in triplet_graph_counter.most_common()
                }
            )
            for pred_lemmas, triplet_graph_counter in self.triplet_graphs_by_pred.items()
        }

    def get_rules(self, text_to_triplets, **kwargs):
        logging.info("collecting patterns...")
        self._get_patterns(text_to_triplets)
        logging.info("getting rules...")
        self.pred_matcher = self._get_matcher_from_graphs(
            self.pred_graphs, label="PRED", threshold=1
        )
        self.arg_matcher = self._get_matcher_from_graphs(
            self.all_arg_graphs, label="ARG", threshold=1
        )
        self.triplet_matchers = self._get_triplet_matchers()
        self.triplet_matchers_by_pred = self._get_triplet_matchers_by_pred()
        self.n_rules = len(self.pred_matcher.patts)

        self._is_trained = True
        return [graph for (graph, _, __), ___ in self.triplet_graphs.most_common(20)]

    def print_rules(self, console):
        console.print("[bold green]Extracted Rules:[/bold green]")
        console.print(f"{self.pred_graphs.most_common(50)=}")
        console.print(f"{self.all_arg_graphs.most_common(50)=}")
        console.print(f"{self.triplet_graphs.most_common(50)=}")

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

    def _match(self, matcher, sen_graph, attrs):
        for key, i, subgraphs in matcher.match(
            sen_graph.G, return_subgraphs=True, attrs=attrs
        ):
            for subgraph in subgraphs:
                ud_subgraph = sen_graph.subgraph(subgraph.nodes)
                indices = frozenset(
                    idx
                    for idx, token in enumerate(ud_subgraph.tokens)
                    if token is not None
                )
                logging.debug(f"MATCH: {indices=}, {ud_subgraph=}")
                yield indices, ud_subgraph

    def _get_arg_cands(self, sen_graph):
        arg_cands = {
            indices: subgraph
            for indices, subgraph in self._match(
                self.arg_matcher, sen_graph, attrs=("upos",)
            )
        }
        arg_roots_to_arg_cands = {
            arg_graph.root: (indices, arg_graph)
            for indices, arg_graph in arg_cands.items()
        }
        arg_cand_root_set = set(arg_roots_to_arg_cands.keys())

        return arg_cands, arg_roots_to_arg_cands, arg_cand_root_set

    def _gen_raw_triplets_for_preds_lexical(self, sen, sen_graph, pred_cands):
        arg_cands, arg_roots_to_arg_cands, arg_cand_root_set = self._get_arg_cands(
            sen_graph
        )

        for pred_cand in pred_cands:
            logging.debug(f"{pred_cand=}")
            pred_lemmas = tuple(sen_graph.G.nodes[i]["name"] for i in pred_cand)
            logging.debug(f"{pred_lemmas=}")
            if pred_lemmas not in self.triplet_matchers_by_pred:
                logging.debug("unknown pred lemmas, skipping")
                continue
            triplet_matchers = self.triplet_matchers_by_pred[pred_lemmas]

            for (
                triplet_matcher,
                arg_root_indices,
                inferred_node_indices,
            ), freq in triplet_matchers.most_common():
                triplet_cands = set(
                    indices
                    for indices in self._match(
                        triplet_matcher, sen_graph, attrs=("upos",)
                    )
                )
                for triplet_cand, triplet_graph in triplet_cands:
                    inferred_nodes = set(
                        triplet_graph.nodes_by_lextop(inferred_node_indices)
                    )
                    arg_roots = triplet_graph.nodes_by_lextop(arg_root_indices)
                    logging.debug("==========================")
                    logging.debug(f"{triplet_cand=}")
                    logging.debug(f"{triplet_graph=}")
                    logging.debug(f"{inferred_nodes=}")
                    logging.debug(f"{arg_roots=}")
                    if pred_cand.issubset(triplet_cand):
                        arg_roots_to_cover = {
                            node for node in triplet_cand - pred_cand - inferred_nodes
                        }
                        logging.debug(f"{arg_roots_to_cover=}")
                        logging.debug(f"{arg_cand_root_set=}")
                        if arg_roots_to_cover.issubset(arg_cand_root_set):
                            logging.debug("FOUND ONE")
                            # we have a winner
                            if arg_roots_to_cover:
                                args = [
                                    arg_roots_to_arg_cands[arg_root][0]
                                    for arg_root in arg_roots_to_cover
                                ]
                            else:
                                args = []

                            yield pred_cand, args

    def _gen_raw_triplets_for_preds_non_lexical(self, sen, sen_graph, pred_cands):
        arg_cands, arg_roots_to_arg_cands, arg_cand_root_set = self._get_arg_cands(
            sen_graph
        )

        for (
            triplet_matcher,
            arg_root_indices,
            inferred_node_indices,
        ), freq in self.triplet_matchers.most_common():
            triplet_cands = set(
                indices
                for indices in self._match(triplet_matcher, sen_graph, attrs=("upos",))
            )
            for triplet_cand, triplet_graph in triplet_cands:
                inferred_nodes = set(
                    triplet_graph.nodes_by_lextop(inferred_node_indices)
                )
                arg_roots = triplet_graph.nodes_by_lextop(arg_root_indices)
                logging.debug("==========================")
                logging.debug(f"{triplet_cand=}")
                logging.debug(f"{triplet_graph=}")
                logging.debug(f"{inferred_nodes=}")
                logging.debug(f"{arg_roots=}")
                for pred_cand in pred_cands:
                    logging.debug(f"{pred_cand=}")
                    if pred_cand.issubset(triplet_cand):
                        arg_roots_to_cover = {
                            node for node in triplet_cand - pred_cand - inferred_nodes
                        }
                        logging.debug(f"{arg_roots_to_cover=}")
                        logging.debug(f"{arg_cand_root_set=}")
                        if arg_roots_to_cover.issubset(arg_cand_root_set):
                            logging.debug("FOUND ONE")
                            # we have a winner
                            if arg_roots_to_cover:
                                args = [
                                    arg_roots_to_arg_cands[arg_root][0]
                                    for arg_root in arg_roots_to_cover
                                ]
                            else:
                                args = []

                            yield pred_cand, args

    def _infer_triplets(self, text: str, lexical=True):
        for sen, sen_graph in self.parse_text(text):
            logging.debug("==========================")
            logging.debug("==========================")
            logging.debug(f"{sen=}")
            pred_cands = {
                indices: subgraph
                for indices, subgraph in self._match(
                    self.pred_matcher, sen_graph, attrs=None
                )
            }
            logging.debug(f"{pred_cands=}")

            if lexical:
                raw_triplet_generator = self._gen_raw_triplets_for_preds_lexical(
                    sen, sen_graph, pred_cands
                )
            else:
                raw_triplet_generator = self._gen_raw_triplets_for_preds_non_lexical(
                    sen, sen_graph, pred_cands
                )

            for pred_cand, args in raw_triplet_generator:
                triplet = Triplet(pred_cand, args, toks=sen_graph.tokens)
                try:
                    mapped_triplet = self.map_triplet(triplet, sen)
                    yield sen, mapped_triplet
                except (
                    KeyError,
                    nx.exception.NetworkXPointlessConcept,
                ):
                    logging.error(f"error mapping triplet: {triplet=}, {sen=}")
                    logging.error("skipping")

    def infer_triplets(self, text: str, **kwargs) -> List[Triplet]:
        triplets = sorted(set(triplet for sen, triplet in self._infer_triplets(text)))
        return triplets
