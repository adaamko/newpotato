import tempfile

from newpotato.hitl import HITLManager


def dict_eq(d1, d2):
    assert d1.keys() == d2.keys()
    for key, value in d1.items():
        if isinstance(value, dict):
            assert dict_eq(value, d2[key])
        else:
            assert d2[key] == value
    return True


def test_save_load():
    sen = "John loves Mary"
    hitl = HITLManager()
    hitl.get_graphs(sen)[0]
    hitl.store_triplet(sen, (1,), ((0,), (2,)))
    hitl.get_annotated_graphs()
    rules = hitl.get_rules()

    fn = tempfile.NamedTemporaryFile(delete="false").name
    hitl.save(fn)
    hitl2 = HITLManager.load(fn)
    rules2 = hitl2.get_rules()
    assert rules == rules2
