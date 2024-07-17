import logging
from collections import Counter, defaultdict
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

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
                # inferred nodes are stored are stored so they can be ignored at matching time
                # both are stored by lextop indices
                triplet_graphs[
                    (
                        triplet_graph,
                        tuple(triplet_graph.index_nodes(triplet.arg_roots)),
                        tuple(triplet_graph.index_inferred_nodes()),
                    )
                ] += 1

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

    def _get_matcher_from_graphs(self, graphs, label, threshold, pos_only):
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

    def get_rules(self, text_to_triplets, **kwargs):
        logging.info("collecting patterns...")
        self._get_patterns(text_to_triplets)
        logging.info("getting rules...")
        self.pred_matcher = self._get_matcher_from_graphs(
            self.pred_graphs, label="PRED", threshold=1, pos_only=False
        )
        self.arg_matcher = self._get_matcher_from_graphs(
            self.all_arg_graphs, label="ARG", threshold=1, pos_only=False
        )
        self.triplet_matchers = Counter(
            {
                (
                    self._get_matcher_from_graphs(
                        Counter({graph: count}),
                        label="TRI",
                        threshold=1,
                        pos_only=False,
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
        self.n_rules = len(self.pred_matcher.patts)

        self._is_trained = True
        return [graph for (graph, _, __), ___ in self.triplet_graphs.most_common(20)]

    def print_rules(self, console):
        console.print("[bold green]Extracted Rules:[/bold green]")
        console.print(f"{self.pred_graphs=}")
        console.print(f"{self.all_arg_graphs=}")
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

    def _match(self, matcher, sen_graph):
        for key, i, subgraphs in matcher.match(sen_graph.G, return_subgraphs=True):
            for subgraph in subgraphs:
                logging.debug(f"MATCH: {sen_graph=}")
                logging.debug(f"MATCH: {subgraph.graph=}")
                ud_subgraph = sen_graph.subgraph(subgraph.nodes)
                indices = frozenset(
                    idx
                    for idx, token in enumerate(ud_subgraph.tokens)
                    if token is not None
                )
                yield indices, ud_subgraph

    def _infer_triplets(self, text: str):
        for sen, sen_graph in self.parse_text(text):
            logging.info("==========================")
            logging.info("==========================")
            logging.info(f"{sen=}")
            pred_cands = {
                indices: subgraph
                for indices, subgraph in self._match(self.pred_matcher, sen_graph)
            }
            logging.info(f"{pred_cands=}")
            arg_cands = {
                indices: subgraph
                for indices, subgraph in self._match(self.arg_matcher, sen_graph)
            }
            arg_roots_to_arg_cands = {
                arg_graph.root: (indices, arg_graph)
                for indices, arg_graph in arg_cands.items()
            }
            arg_cand_root_set = set(arg_roots_to_arg_cands.keys())

            for (
                triplet_matcher,
                arg_root_indices,
                inferred_node_indices,
            ), freq in self.triplet_matchers.most_common():
                triplet_cands = set(
                    indices for indices in self._match(triplet_matcher, sen_graph)
                )
                for triplet_cand, triplet_graph in triplet_cands:
                    inferred_nodes = set(
                        triplet_graph.nodes_by_lextop(inferred_node_indices)
                    )
                    arg_roots = triplet_graph.nodes_by_lextop(arg_root_indices)
                    logging.info("==========================")
                    logging.info(f"{triplet_cand=}")
                    logging.info(f"{triplet_graph=}")
                    logging.info(f"{inferred_nodes=}")
                    logging.info(f"{arg_roots=}")
                    for pred_cand in pred_cands:
                        logging.info(f"{pred_cand=}")
                        if pred_cand.issubset(triplet_cand):
                            arg_roots_to_cover = {
                                node
                                for node in triplet_cand - pred_cand - inferred_nodes
                            }
                            logging.info(f"{arg_roots_to_cover=}")
                            logging.info(f"{arg_cand_root_set=}")
                            if arg_roots_to_cover.issubset(arg_cand_root_set):
                                logging.info("FOUND ONE")
                                # we have a winner
                                if arg_roots_to_cover:
                                    args = [
                                        arg_roots_to_arg_cands[arg_root][0]
                                        for arg_root in arg_roots
                                    ]
                                else:
                                    args = []
                                triplet = Triplet(
                                    pred_cand, args, toks=sen_graph.tokens
                                )
                                mapped_triplet = self.map_triplet(triplet, sen)
                                yield sen, mapped_triplet

    def infer_triplets(self, text: str, **kwargs) -> List[Triplet]:
        triplets = sorted(set(triplet for sen, triplet in self._infer_triplets(text)))
        return triplets
