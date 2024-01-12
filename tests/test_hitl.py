import tempfile
import unittest

from newpotato.hitl import HITLManager


def dict_eq(d1, d2):
    assert d1.keys() == d2.keys()
    for key, value in d1.items():
        if isinstance(value, dict):
            assert dict_eq(value, d2[key])
        else:
            assert d2[key] == value
    return True


class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hitl = HITLManager()

    def test_save_load(self):
        sen = "John loves Mary"
        self.hitl.get_graphs(sen)[0]
        self.hitl.store_triplet(sen, (1,), ((0,), (2,)))
        self.hitl.get_annotated_graphs()
        rules = self.hitl.get_rules()

        fn = tempfile.NamedTemporaryFile(delete="false").name
        self.hitl.save(fn)
        hitl2 = HITLManager.load(fn)
        rules2 = hitl2.get_rules()
        assert rules == rules2

    def test_toks_from_txt(self):
        sen = "The AssetId property of the DriveType is unique."
        word = "unique"

        self.hitl.get_graphs(sen)
        assert self.hitl.get_toks_from_txt(word, sen) == (7,)
