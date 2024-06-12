from newpotato.datatypes import Triplet
from newpotato.hitl import HITLManager


def test_annotation():
    # Initialize FastAPI, HITLManager
    hitl_manager = HITLManager(extractor_type="ud")

    sen = "The alarm was cleared after 3 min but reoccurred again at 8h51."
    hitl_manager.extractor.get_graphs(sen, 0)
    pred, args = (1,), ((), ())
    toks = hitl_manager.extractor.get_tokens(sen)
    triplet = Triplet(pred, args, toks=toks)
    mapped_triplet = hitl_manager.extractor.map_triplet(triplet, sen)
    hitl_manager.store_triplet(sen, mapped_triplet, True)
    hitl_manager.get_rules()

    parsed_graphs = hitl_manager.extractor.parsed_graphs
    sentences = [sen for sen in parsed_graphs.keys() if sen != "latest"]

    for sentence in sentences:
        hitl_manager.extractor.extract_triplets_from_text(
            sentence, convert_to_text=True
        )


if __name__ == "__main__":
    # Set up logging
    import logging

    logging.basicConfig(
        format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
        level=logging.INFO,
        force=True,
    )
    test_annotation()
