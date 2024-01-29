import logging
import requests
from typing import Any, Dict, List, Optional


from fastcoref import spacy_component
from graphbrain.parsers import create_parser
import spacy
from spacy.vocab import Vocab

from newpotato.datatypes import GraphParse


assert spacy_component  # silence flake8


class TextParser:
    """A class to handle text parsing using Graphbrain."""

    @staticmethod
    def from_params(params: Dict[str, Any]):
        if params is None:
            return TextParser()
        else:
            return TextParser(**params)

    def __init__(
        self,
        lang: str = "en",
        corefs: bool = True,
        spacy_vocab_path: Optional[str] = "spacy_vocab",
    ):
        self.lang = lang
        self.corefs = corefs
        self.init_parser()
        self.get_vocab().to_disk(spacy_vocab_path)

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

    def parse(self, text: str) -> List[GraphParse]:
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
                graphs.append(GraphParse(graph))

        return graphs

    def get_vocab(self):
        return self.parser.nlp.vocab


class TextParserClient:
    """client to access TextParserServer"""

    def __init__(
        self,
        parser_url: Optional[str] = "http://localhost:7277",
        spacy_vocab_path: Optional[str] = "spacy_vocab",
    ):
        self.vocab = Vocab().from_disk(spacy_vocab_path)
        logging.info(f"loaded spacy vocab from {spacy_vocab_path=}")
        self.url = parser_url
        parser_params = self.get_params()
        logging.info(f"connected to parser, {parser_url=}, {parser_params=}")

    def get_params(self):
        response = requests.request("GET", f"{self.url}/get_params")
        return response.json()["params"]

    def check_params(self, params):
        response = requests.request("POST", f"{self.url}/check_params", json={"params": params})
        if response.status_code == 200:
            return True
        return False

    def get_vocab(self):
        return self.vocab

    def parse(self, text):
        response = requests.request("POST", f"{self.url}/parse", json={"text": text})
        json_graphs = response.json()["graphs"]
        graphs = [GraphParse.from_json(graph, self.vocab) for graph in json_graphs]
        return graphs


def test_parser():
    import sys

    url = sys.argv[1]
    client = TextParserClient(url)
    while True:
        text = input()
        graphs = client.parse(text)
        print(f"{graphs=}")


if __name__ == "__main__":
    test_parser()
