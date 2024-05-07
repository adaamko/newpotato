import logging
from collections import Counter, defaultdict
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

from newpotato.datatypes import Triplet
from newpotato.extractors.extractor import Extractor
from newpotato.extractors.graph_parser_client import GraphParserClient

from tuw_nlp.graph.graph import Graph
from tuw_nlp.graph.utils import GraphFormulaPatternMatcher


class GraphMappedTriplet(Triplet):
    def __init__(self, triplet: Triplet, pred_graph: Graph, arg_graphs: Tuple[Graph]):
        super(GraphMappedTriplet, self).__init__(triplet.pred, triplet.args)
        self.pred_graph = pred_graph
        self.arg_graphs = arg_graphs
        self.mapped = True


class GraphBasedExtractor(Extractor):
    @staticmethod
    def from_json(data: Dict[str, Any]):
        raise NotImplementedError

    def to_json(self) -> Dict[str, Any]:
        raise NotImplementedError

    def __init__(
        self,
        parser_url: Optional[str] = "http://localhost:7277",
        default_relation: Optional[str] = None,
    ):
        super(GraphBasedExtractor, self).__init__()
        self.text_parser = GraphParserClient(parser_url)
        self.default_relation = default_relation

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

    def get_rules(self, text_to_triplets):
        pred_graphs = Counter()
        triplet_graphs = Counter()
        arg_graphs = defaultdict(Counter)
        for text, triplets in text_to_triplets.items():
            # toks = self.get_tokens(text)
            logging.debug(f"{text=}")
            graph = self.parsed_graphs[text]
            logging.debug(graph.to_dot())
            lemmas = self.get_lemmas(text)
            for triplet in triplets:
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
                    triplet_toks, handle_unconnected="shortest_path_infer"
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
                        inferred_pred_toks, handle_unconnected="shortest_path_infer"
                    )
                    logging.debug(f"{inferred_pred_graph=}")
                    pred_graphs[inferred_pred_graph] += 1

        self.pred_graphs = pred_graphs
        self.arg_graphs = arg_graphs
        self.triplet_graphs = triplet_graphs
        
        patterns = []
        threshold = 1
        label = 'LABEL'
        for pred_graph, freq in self.pred_graphs.most_common():
            if freq < threshold:
                break
            patterns.append(((pred_graph.G,), (), label))

        self.pred_matcher = GraphFormulaPatternMatcher(patterns, converter=None, case_sensitive=False)

    def print_rules(self):
        print(f"{self.pred_graphs=}")
        print(f"{self.arg_graphs=}")
        print(f"{self.triplet_graphs=}")

    def extract_triplets_from_text(self, text, **kwargs):
        raise NotImplementedError

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
            for arg in triplet.args
        ]
        logging.debug(f"triplet mapped: {arg_subgraphs=}")
        return GraphMappedTriplet(triplet, pred_subgraph, arg_subgraphs)

    def infer_triplets(self, sen: str, **kwargs) -> List[Triplet]:
        triplets = []
        for _, sen_graph in self.parse_text(sen):
            for key, i, subgraphs in self.pred_matcher.match(sen_graph.G, return_subgraphs=True):
                for subgraph in subgraphs:
                    logging.debug(f'MATCH: {sen=}')
                    logging.debug(f'MATCH: {subgraph.graph=}')
                    pred_indices = tuple(idx for idx, token in enumerate(subgraph.graph['tokens']) if token is not None)
                    triplet = Triplet(pred_indices, ())
                    mapped_triplet = self.map_triplet(triplet, sen)
                    triplets.append(mapped_triplet)

        return triplets
