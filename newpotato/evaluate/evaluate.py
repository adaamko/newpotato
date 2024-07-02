import logging
from collections import Counter, defaultdict

from newpotato.datatypes import triplets_to_str


class Evaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.events = None
        self.counts = None
        self.results = None

    def infer_triplets(self, sen):
        raise NotImplementedError

    def gen_texts_with_gold_triplets(self):
        raise NotImplementedError

    def _get_counts(self):
        c = Counter()
        for sen, gold_set, pred_set in self.get_events():
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

    def get_counts(self):
        if self.counts is None:
            self._get_counts()
        return self.counts

    def _get_events(self):
        self.events = []
        for sen, gold_list in self.gen_texts_with_gold_triplets():
            golds = set(gold_list)
            preds = set(self.infer_triplets(sen))
            self.events.append((sen, golds, preds))

    def get_events(self):
        if self.events is None:
            self._get_events()
        yield from self.events

    def get_event_type(self, golds, preds):
        fn = len(golds - preds)
        fp = len(preds - golds)
        if fn == 0:
            if fp == 0:
                return "C"
            return "FP"
        if fp == 0:
            return "FN"
        return "B"

    def write_events(self, stream):
        for sen, golds, preds in self.get_events():
            e_type = self.get_event_type(golds, preds)
            logging.debug(
                "golds: " + ", ".join(f"{(t.pred, t.args)}" for t in golds)
            )
            logging.debug("preds: " + ", ".join(f"{(t.pred, t.args)}" for t in preds))
            golds_txt = " ".join(triplets_to_str(golds))
            preds_txt = " ".join(triplets_to_str(preds))
            stream.write(f"{e_type}\t{sen}\t{golds_txt}\t{preds_txt}\n")

    def write_events_to_file(self, fn):
        with open(fn, "w") as f:
            self.write_events(f)

    def f1(self, p, r):
        return 0.0 if p + r == 0 else (2 * p * r) / (p + r)

    def _get_results(self):
        c = self.get_counts()
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
