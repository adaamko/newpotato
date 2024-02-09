from newpotato.parser import TextParser


def test_parser():
    input_text = (
        "It was a bright cold day in April, and the clocks were striking thirteen."
    )
    parser = TextParser()
    graph = parser.parse(input_text)[0]
    # assert parse_text(input_text) == {"status": "ok"}
    assert graph['text'] == input_text


if __name__ == "__main__":
    print(test_parser())
