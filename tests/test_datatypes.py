from newpotato.datatypes import Triplet


def test_triplet():
    pred = (1, 2, 3)
    args = ((4, 5), (6, 7))
    triplet = Triplet(pred, args)
    assert triplet.pred == pred
    assert triplet.args == args
