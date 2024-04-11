import itertools
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from graphbrain.hyperedge import Hyperedge, hedge
from graphbrain.learner.classifier import Classifier
from graphbrain.learner.classifier import from_json as classifier_from_json
from graphbrain.learner.rule import Rule

from newpotato.constants import NON_ATOM_WORDS, NON_WORD_ATOMS
from newpotato.datatypes import Triplet
from newpotato.extractors.extractor import Extractor
from newpotato.extractors.graphbrain_parser import GraphbrainParserClient, GraphParse


def edge2toks(edge: Hyperedge, graph: Dict[str, Any]):
    """
    find IDs of tokens covered by an edge of a graph
    If some atom names match more than one token, candidate token sequences are disambiguated
    based on length and the shortest sequence (i.e. the one with the fewest gaps) is returned

    Args:
        edge (Hyperedge): the Graphbrain Hyperedge to be mapped to token IDs
        graph (Dict[str, Any]): the Graphbrain Hypergraph of the full utterance

    Returns:
        Tuple[int, ...]: tuple of token IDs covered by the subedge
    """

    logging.debug(f"edge2toks\n{edge=}\n{graph=}")

    toks = set()
    strs_to_atoms = defaultdict(list)
    for atom, word in graph["atom2word"].items():
        strs_to_atoms[atom.to_str()].append(atom)

    to_disambiguate = []
    for atom in edge.all_atoms():
        atom_str = atom.to_str()
        if atom_str not in strs_to_atoms:
            assert (
                str(atom) in NON_WORD_ATOMS
            ), f"no token corresponding to {atom=} in {strs_to_atoms=}"
        else:
            cands = strs_to_atoms[atom_str]
            if len(cands) == 1:
                toks.add(graph["atom2word"][cands[0]][1])
            else:
                to_disambiguate.append([graph["atom2word"][cand][1] for cand in cands])

    if len(to_disambiguate) > 0:
        logging.debug(f"edge2toks disambiguation needed: {toks=}, {to_disambiguate=}")
        hyp_sets = []
        for cand in itertools.product(*to_disambiguate):
            hyp_toks = sorted(toks | set(cand))
            hyp_length = hyp_toks[-1] - hyp_toks[0]
            hyp_sets.append((hyp_length, hyp_toks))

        shortest_hyp = sorted(hyp_sets)[0][1]
        logging.debug(f"{shortest_hyp=}")
        return set(shortest_hyp)

    return tuple(sorted(toks))


def matches2triplets(matches: List[Dict], graph: Dict[str, Any]) -> List[Triplet]:
    """
    convert graphbrain matches on a sentence to triplets of the tokens of the sentence

    Args:
        matches (List[Dict]): a list of hypergraphs corresponding to the matches
        graphs (Dict[str, Any]]): The hypergraph of the sentence

    Returns:
        List[Triplet] the list of triplets corresponding to the matches
    """
    triplets = []
    for triple_dict in matches:
        pred = []
        args = []
        for key, edge in triple_dict.items():
            if key == "REL":
                pred = edge2toks(edge, graph)
            else:
                args.append((int(key[-1]), edge2toks(edge, graph)))

        sorted_args = [arg[1] for arg in sorted(args)]
        triplets.append(Triplet(pred, sorted_args))

    return triplets


class UnmappableTripletError(Exception):
    pass


def _toks2subedge(
    edge: Hyperedge,
    toks_to_cover: Tuple[int],
    all_toks: Tuple[int],
    words_to_i: Dict[str, Set[int]],
) -> Tuple[Hyperedge, Set[str]]:
    """
    recursive helper function of toks2subedge

    Args:
        edge (Hyperedge): the Graphbrain Hyperedge in which to look for the subedge
        words (tuple): the tokens to be covered by the subedge
        all_toks (tuple): all tokens in the sentence
        words_to_i (dict): words mapped to token indices

    Returns:
        Hyperedge: the best matching subedge
        set: tokens covered by the matching hyperedge
        set: additional tokens in the matching hyperedge
    """
    words_to_cover = [all_toks[i] for i in toks_to_cover]
    logging.debug(f"_toks2subedge got: {edge=}, {words_to_cover=}")
    if edge.is_atom():
        lowered_word = edge.label().lower()
        if lowered_word not in words_to_i:
            assert (
                lowered_word in NON_WORD_ATOMS
            ), f"no token corresponding to edge label {lowered_word} and it is not listed as a non-word atom"

        toks = words_to_i[lowered_word]
        relevant_toks = toks & toks_to_cover
        if len(relevant_toks) > 0:
            return edge, relevant_toks, set()
        else:
            if lowered_word in NON_ATOM_WORDS:
                return edge, set(), set()
            else:
                return edge, set(), toks

    relevant_toks, irrelevant_toks = set(), set()
    subedges_to_keep = []
    for i, subedge in enumerate(edge):
        s_edge, subedge_relevant_toks, subedge_irrelevant_toks = _toks2subedge(
            subedge, toks_to_cover, all_toks, words_to_i
        )
        if subedge_relevant_toks == toks_to_cover:
            # a subedge covering everything, search can stop
            logging.debug(
                f"_toks2subedge: subedge covers everything, returning {s_edge=}, {subedge_relevant_toks=}, {subedge_irrelevant_toks=}"
            )
            return s_edge, subedge_relevant_toks, subedge_irrelevant_toks

        if len(subedge_relevant_toks) > 0 or (i == 0 and subedge.atom):
            # the first subedge must be kept, to connect the rest (unless it is not atomic)
            subedges_to_keep.append(s_edge)
            relevant_toks |= subedge_relevant_toks
            irrelevant_toks |= subedge_irrelevant_toks

    if len(relevant_toks) == 0:
        # no words covered
        logging.debug(
            f"_toks2subedge: no words covered, returning {edge=}, {relevant_toks=}, {irrelevant_toks=}"
        )
        return edge, relevant_toks, irrelevant_toks
    else:
        pruned_edge = hedge(subedges_to_keep)
        logging.debug(
            f"_toks2subedge: returning {pruned_edge=}, {relevant_toks=}, {irrelevant_toks=}"
        )
        return pruned_edge, relevant_toks, irrelevant_toks


def toks2subedge(
    edge: Hyperedge,
    toks: Tuple[int],
    all_toks: Tuple[int],
    words_to_i: Dict[str, Set[int]],
) -> Hyperedge:
    """
    find subedge in edge corresponding to the phrase in text.
    Based on graphbrain.learner.text2subedge, but keeps track of the set of words covered
    by partial results.
    If an exact match is not possible, returns the smallest subedge that covers all words.
    Raises ValueError if input edge does not contain all words.

    Args:
        edge (Hyperedge): the Graphbrain Hyperedge in which to look for the subedge
        words (set): the words to be covered by the subedge

    Returns:
        Hyperedge: the best matching hyperedge
        bool: whether the matching edge is exact (contains all the words and no other words)
    """
    try:
        toks_to_cover = {
            tok for tok in toks if all_toks[tok].lower() not in NON_ATOM_WORDS
        }
    except IndexError:
        logging.warning(f"some token IDs out of range: {toks=}, {all_toks=}")
        raise UnmappableTripletError()
    subedge, relevant_toks, irrelevant_toks = _toks2subedge(
        edge, toks_to_cover, all_toks, words_to_i
    )
    logging.debug(f"toks2subedge: {subedge=}, {relevant_toks=}, {irrelevant_toks=}")

    if toks_to_cover == relevant_toks:
        if len(irrelevant_toks) == 0:
            return subedge, relevant_toks, True
        logging.warning(f"returning incomplete match: {irrelevant_toks=}")
        return subedge, relevant_toks, False
    else:
        words = [all_toks[t] for t in toks_to_cover]
        logging.warning(
            f"hyperedge {edge} does not contain all words in {words}\n{toks_to_cover=}, {relevant_toks=}"
        )
        raise UnmappableTripletError()


class GraphbrainMappedTriplet(Triplet):
    def __init__(
        self, mapped_pred, mapped_args, toks=None, variables=None, sen_graph=None
    ):
        super(GraphbrainMappedTriplet, self).__init__(mapped_pred, mapped_args)
        self.toks = toks
        self.variables = variables
        self.sen_graph = sen_graph


class GraphbrainExtractor(Extractor):
    """A class to extract triplets from Semantic Hypergraphs.

    Attributes:
        classifier (Optional[Classifier]): The classifier to use for extraction.
        parser_params (Dict): parameters to be used to initialize a TextParser object
    """

    def __init__(
        self,
        classifier: Optional[Classifier] = None,
        parser_url: Optional[str] = "http://localhost:7277",
    ):
        super(GraphbrainExtractor, self).__init__()
        self.classifier = classifier
        self.text_parser = GraphbrainParserClient(parser_url)
        self.spacy_vocab = self.text_parser.get_vocab()

    @staticmethod
    def from_json(data: Dict[str, Any]):
        extractor = GraphbrainExtractor()
        extractor.text_parser.check_params(data["parser_params"])
        if data["classifier"] is not None:
            extractor.classifier = classifier_from_json(data["classifier"])

        extractor.parsed_graphs = {
            text: GraphParse.from_json(graph_dict, extractor.spacy_vocab)
            for text, graph_dict in data["parsed_graphs"].items()
        }
        return extractor

    def to_json(self) -> Dict[str, List]:
        data = {
            "extractor_type": "graphbrain",
            "parsed_graphs": {
                text: graph.to_json() for text, graph in self.parsed_graphs.items()
            },
            "parser_params": self.text_parser.get_params(),
            "classifier": None,
        }
        if self.classifier is not None:
            data["classifier"] = self.classifier.to_json()

        return data

    def parse_text(self, text: str) -> List[GraphParse]:
        """
        Parse the given text.

        Args:
            text (str): The text to parse.

        Returns:
            Generator[Dict[str, Dict[str, Any]]]: The parsed graphs.
        """
        graphs = self.text_parser.parse(text)
        for graph in graphs:
            yield graph['text'], graph

    def get_annotated_graphs_from_classifier(self) -> List[str]:
        """
        Get the annotated graphs

        Returns:
            List[str]: The annotated graphs. An annotated graph is a hyperedge that has been annotated with variables. e.g. "REL(ARG1, ARG2)"
        """
        assert self.classifier is not None, "classifier not initialized"
        return [str(rule[0]) for rule in self.classifier.cases]

    def get_rules(self, text_to_triplets=None, learn: bool = True) -> List[Rule]:
        """
        Get the rules.

        Args:
            learn (bool): whether to run graphbrain classifier's learn function.
                If False (default), only extract_patterns is called
        """
        if text_to_triplets is not None:
            self.add_cases(text_to_triplets)
            self.extract_rules(learn=learn)

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

    def print_rules(self, console):
        annotated_graphs = self.get_annotated_graphs_from_classifier()
        console.print("[bold green]Annotated Graphs:[/bold green]")
        console.print(annotated_graphs)

        rules = self.get_rules()
        console.print("[bold green]Extracted Rules:[/bold green]")
        console.print(rules)

    def add_cases(
        self,
        text_to_triplets: Dict[str, List[Triplet]],
    ):
        """
        Add cases to the classifier.

        Args:
            parsed_graphs (Dict[str, Dict[str, Any]]): The parsed graphs.
            text_to_triplets (Dict[str, List[Tuple]]): The texts and corresponding triplets.
        """
        classifier = Classifier()
        for text, triplets in text_to_triplets.items():
            graph = self.parsed_graphs[text]
            main_edge = graph["main_edge"]
            for triplet, positive in triplets:
                if not isinstance(triplet, GraphbrainMappedTriplet):
                    logging.warning(
                        f"trying to map unmapped triplet {triplet} to {graph}"
                    )
                    mapped_triplet = self.map_triplet(triplet, text)
                    if not mapped_triplet:
                        logging.warning("failed to map triplet, skipping")
                        continue
                else:
                    mapped_triplet = triplet

                logging.info("adding case:")
                logging.info(f"{text=}, {main_edge=}")
                logging.info(
                    f"{mapped_triplet=}, {mapped_triplet.variables=}, positive: {positive}"
                )

                # positive means whether we want to treat it as a positive or negative example
                # this helps graphbrain to learn the rules
                classifier.add_case(
                    main_edge, positive=positive, variables=mapped_triplet.variables
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

    def get_tokens(self, text: str) -> List[str]:
        """
        Get the tokens of the given text.
        """
        return [tok.text for tok in self.parsed_graphs[text]["spacy_sentence"]]

    def add_text_to_graphs(self, text: str) -> None:
        """Add the given text to the graphs.

        Args:
            text (str): The text to add to the graphs.

        Returns:
            None
        """
        self.get_graphs(text)

    def match_rules(self, sen: str) -> List[Dict]:
        """
        match rules against sentence by passing the sentence's graph to the extractor

        Args:
            sen (str): the sentence to be matched against

        Returns:
            List[Dict] a list of hypergraphs corresponding to the matches
        """
        graphs = self.get_graphs(sen)
        all_matches = []
        for graph in graphs:
            main_graph = graph["main_edge"]
            matches, _ = self.classify(main_graph)
            all_matches += matches
        return all_matches

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
        matches_by_text = {
            graph["text"]: {"matches": [], "rules_triggered": [], "triplets": []}
            for graph in graphs
        }

        for graph in graphs:
            matches, rules_triggered = self.classify(graph["main_edge"])
            logging.info(f"matches: {matches}")
            triplets = matches2triplets(matches, graph)
            mapped_triplets = [
                self.map_triplet(triplet, graph["text"]) for triplet in triplets
            ]
            logging.info(f"triplets: {mapped_triplets}")

            if convert_to_text:
                matches = [
                    {k: v.label() for k, v in match.items()} for match in matches
                ]

            matches_by_text[graph["text"]]["matches"] = matches
            matches_by_text[graph["text"]]["rules_triggered"] = rules_triggered
            matches_by_text[graph["text"]]["triplets"] = mapped_triplets

        return matches_by_text

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

    def map_to_subgraphs(self, triplet, sen_graph, strict=True):
        """
        helper function for map_to_subgraphs
        """
        all_toks = tuple(tok.text for tok in sen_graph["spacy_sentence"])
        words_to_i = defaultdict(set)
        for i, word in enumerate(all_toks):
            words_to_i[word.lower()].add(i)

        edge = sen_graph["main_edge"]
        variables = {}

        if triplet.pred is not None:
            rel_edge, relevant_toks, exact_match = toks2subedge(
                edge, triplet.pred, all_toks, words_to_i
            )
            if not exact_match and strict:
                logging.warning(
                    f"cannot map pred {triplet.pred} to subedge of {edge} (closest: {rel_edge}"
                )
                raise UnmappableTripletError
            variables["REL"] = rel_edge
            mapped_pred = tuple(sorted(relevant_toks))
        else:
            mapped_pred = None

        mapped_args = []
        for i in range(len(triplet.args)):
            arg_edge, relevant_toks, exact_match = toks2subedge(
                edge, triplet.args[i], all_toks, words_to_i
            )
            if not exact_match and strict:
                logging.warning(
                    f"cannot map arg {triplet.args[i]} to subedge of {edge} (closest: {arg_edge}"
                )
                raise UnmappableTripletError()
            variables[f"ARG{i}"] = arg_edge
            mapped_args.append(tuple(sorted(relevant_toks)))

        return mapped_pred, mapped_args, variables

    def map_triplet(self, triplet, sentence, strict=True):
        """
        map predicate and arguments of a triplet (each a tuple of token indices) to
        corresponding subgraphs (Hyperedges). The mapping may change the indices, since words
        not showing up in the hypergraph (e.g. punctuation) are not to be considered part of the triplet
        """
        sen_graph = self.parsed_graphs[sentence]
        try:
            mapped_pred, mapped_args, variables = self.map_to_subgraphs(
                triplet, sen_graph, strict=strict
            )
        except UnmappableTripletError:
            logging.warning(
                f"could not map triplet ({triplet.pred=}, {triplet.args=} to {sen_graph=}"
            )
            return False
        else:
            toks = self.get_tokens(sentence)
            return GraphbrainMappedTriplet(
                mapped_pred, tuple(mapped_args), toks, variables, sen_graph
            )
