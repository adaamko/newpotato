import logging


class Triplet:
    """A class to handle triplets.

    A triplet consists  of a predicate and a list of arguments.
    """

    def __init__(self, pred, args, toks=None):
        logging.debug(f"triple init got: pred: {pred}, args: {args}")
        self.pred = None if pred is None else tuple(int(i) for i in pred)
        self.args = tuple(tuple(int(i) for i in arg) for arg in args)
        self.toks = toks
        self.mapped = False

    @staticmethod
    def from_json(data):
        return Triplet(
            data["pred"],
            data["args"],
        )

    def to_json(self):
        return {
            "pred": self.pred,
            "args": self.args,
        }

    def __eq__(self, other):
        return (
            isinstance(other, Triplet)
            and self.pred == other.pred
            and self.args == other.args
        )

    def __hash__(self):
        return hash((self.pred, self.args))

    def to_str(self, toks):
        pred_phrase = (
            "" if self.pred is None else "_".join(toks[a] for a in self.pred)
        )
        args_str = ", ".join(
            "_".join(toks[a] for a in phrase) for phrase in self.args
        )
        return f"{pred_phrase}({args_str})"

    def __str__(self):
        if self.toks:
            return self.to_str(self.toks)
        else:
            return f"{self.pred=}, {self.args=}"

    def __repr__(self):
        return str(self)
