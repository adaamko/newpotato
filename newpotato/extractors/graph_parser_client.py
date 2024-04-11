import logging
import requests
from typing import Optional

from tuw_nlp.graph.ud_graph import UDGraph


def get_graph_cls(graph_type):
    if graph_type == "UD":
        return UDGraph
    else:
        raise ValueError(f"unsupported graph type: {graph_type}")


class GraphParserClient:
    """client to access GraphbrainParserServer"""

    def __init__(
        self,
        parser_url: Optional[str] = "http://localhost:7277",
    ):
        self.url = parser_url
        parser_params = self.get_params()
        logging.info(f"connected to parser, {parser_url=}, {parser_params=}")

    def get_params(self):
        response = requests.request("GET", f"{self.url}/get_params")
        return response.json()["params"]

    def check_params(self, params):
        response = requests.request(
            "POST", f"{self.url}/check_params", json={"params": params}
        )
        if response.status_code == 200:
            return True
        return False

    def get_vocab(self):
        return self.vocab

    def parse(self, text):
        response = requests.request(
            "POST", f"{self.url}/parse", json={"text": text}
        ).json()
        graph_cls = get_graph_cls(response["graph_type"])
        json_graphs = response["graphs"]
        graphs = [graph_cls.from_json(graph) for graph in json_graphs]
        return graphs
