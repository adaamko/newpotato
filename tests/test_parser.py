from newpotato.hitl import TextParser


def test_parser():
    input_text = (
        "It was a bright cold day in April, and the clocks were striking thirteen."
    )
    parser = TextParser()
    graphs = parser.parse(input_text)
    # assert parse_text(input_text) == {"status": "ok"}
    return graphs


if __name__ == "__main__":
    print(test_parser())
