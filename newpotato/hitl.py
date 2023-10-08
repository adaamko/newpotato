from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
            graphs.extend(parses)

        return graphs


@dataclass
class HITLManager:
    """A class to manage the HITL process and store parsed graphs."""

    parsed_graphs: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def store_parsed_graphs(self, parsed_graphs: List[Dict[str, Any]]):
        """
        Store the parsed graphs.

        Args:
            parsed_graphs (List[Dict[str, Any]]): The parsed graphs to store.
        """
        self.parsed_graphs["latest"] = parsed_graphs
