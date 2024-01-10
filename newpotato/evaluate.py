import logging
from collections import Counter, defaultdict


class Evaluator:
    def __init__(self):
        self.events = []
        self.counts = None
        self.results = None

    def add(self, golds, preds):
        logging.debug(f'eval event: golds: {golds}, preds: {preds}')
        gold_types = [type(t) for t in golds]
        pred_types = [type(t) for t in preds]
        logging.debug(f'types: golds: {gold_types}, preds: {pred_types}')
        self.events.append((set(golds), set(preds)))

    def count(self):
        c = Counter()
        for gold_set, pred_set in self.events:
            c["docs"] += 1
            c["gold"] += len(gold_set)
            c["pred"] += len(pred_set)
            c["tp"] += len(gold_set & pred_set)
            c["fn"] += len(gold_set - pred_set)
            c["fp"] += len(pred_set - gold_set)
            if gold_set == pred_set:
                c["docs_corr"] += 1
        self.counts = c
        self.results = None

    def f1(self, p, r):
        return 0.0 if p + r == 0 else (2 * p * r) / (p + r)

    def _get_results(self):
        c = self.counts
        res = defaultdict(int)
        res["n_docs"] = c["docs"]
        res["n_gold"] = c["gold"]
        res["n_pred"] = c["pred"]
        res["n_docs"] = c["docs"]
        res["docs_corr"] = c["docs_corr"]
        res["docs_acc"] = c["docs_corr"] / c["docs"]
        if c["pred"] > 0:
            res["precision"] = c["tp"] / c["pred"]
        if c["gold"] > 0:
            res["recall"] = c["tp"] / c["gold"]
        if res["precision"] > 0 and res["recall"] > 0:
            res["f"] = self.f1(res["precision"], res["recall"])

        return res

    def get_results(self):
        if self.results is None:
            self.results = self._get_results()

        return self.results
