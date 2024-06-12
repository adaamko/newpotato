import tempfile
import unittest

from newpotato.datatypes import Triplet
from newpotato.hitl import HITLManager
from newpotato.utils import get_toks_from_txt


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
        cls.hitl = HITLManager(extractor_type='ud')

    def test_save_load(self):
        sen = "John loves Mary"
        self.hitl.extractor.get_graphs(sen)[sen]
        triplet = Triplet((1,), ((0,), (2,)))
        mapped_triplet = self.hitl.extractor.map_triplet(triplet, sen)
        self.hitl.store_triplet(sen, mapped_triplet)
        rules = self.hitl.get_rules()

        fn = tempfile.NamedTemporaryFile(delete="false").name
        self.hitl.save(fn)
        hitl2 = HITLManager.load(fn)
        rules2 = hitl2.get_rules()
        assert rules == rules2

    def test_toks_from_txt(self):
        sen = "The AssetId property of the DriveType is unique."
        word = "unique"

        graph = self.hitl.extractor.get_graphs(sen)[sen]
        assert get_toks_from_txt(word, graph.stanza_sen) == (7,)
