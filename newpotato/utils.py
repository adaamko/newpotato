import collections
import operator
import logging
import re
from functools import reduce
from typing import Tuple

from stanza.models.common.doc import Sentence

from newpotato.datatypes import Triplet


class AnnotatedWordsNotFoundError(Exception):
    pass


def print_tokens(sentence, extractor, console):
    tokens = extractor.get_tokens(sentence)
    console.print("[bold cyan]Tokens:[/bold cyan]")
    console.print(" ".join(f"{i}_{tok}" for i, tok in enumerate(tokens)))


def _get_single_triplet_from_user(console):
    annotation = input("> ")
    if annotation == "":
        return None

    if annotation == "O":
        return "O"

    try:
        phrases = [
            tuple(int(n) for n in ids.split("_")) if ids else None
            for ids in annotation.split(",")
        ]
        pred = phrases[0]
        args = [arg if arg is not None else () for arg in phrases[1:]]
        return pred, args

    except ValueError:
        console.print("[bold red]Could not parse this:[/bold red]", annotation)
        return False


def get_single_triplet_from_user(sentence, extractor, console, expect_mappable=True):
    console.print(
        """
        [bold cyan] Enter comma-separated list of predicate and args, with token IDs in each separated by underscores, e.g.: 0_The 1_boy 2_has 3_gone 4_to 5_school -> 2_3,0_1,4_5
        Alternatively, type O to get triplet from oracle (if loaded).
        Press ENTER to finish.
        
        [/bold cyan]"""
    )
    while True:
        raw_triplet = _get_single_triplet_from_user(console)
        if raw_triplet is None:
            # user is done
            return None
        if raw_triplet is False:
            # syntax error
            continue
        if raw_triplet == "O":
            # oracle
            return "O"

        pred, args = raw_triplet
        triplet = Triplet(pred, args)
        mapped_triplet = extractor.map_triplet(triplet, sentence)
        if expect_mappable and mapped_triplet is False:
            console.print(
                f"[bold red] Could not map annotation {triplet} to subedges, please provide alternative (or press ENTER to skip)[/bold red]"
            )
            continue

        return mapped_triplet


def get_triplets_from_user(sentence, hitl, console):
    print_tokens(sentence, hitl.extractor, console)

    while True:
        triplet = get_single_triplet_from_user(sentence, hitl.extractor, console)

        if triplet is None:
            break

        elif isinstance(triplet, Triplet):
            hitl.store_triplet(sentence, triplet, True)

        elif triplet in ("O", "A"):
            if hitl.oracle is None:
                console.print("[bold red]The Oracle is not loaded.[/bold red]")
            elif sentence not in hitl.oracle:
                console.print(
                    "[bold red]The Oracle does not annotate this sentence.[/bold red]"
                )
            else:
                for o_triplet, is_true in hitl.oracle[sentence]:
                    console.print(f"[bold cyan]Oracle says: {o_triplet}[/bold cyan]")
                    if triplet == "A":
                        hitl.store_triplet(sentence, o_triplet, is_true)
        else:
            raise ValueError(
                'get_single_triplet_from_user must return Triplet, None, "O", or "A"'
            )


def get_triplet_from_annotation(
    pred, args, sen, sen_graph, extractor, console, ask_user=True
):
    toks = extractor.get_tokens(sen)
    unmapped_triplet = Triplet(pred, args, toks)
    triplet = extractor.map_triplet(unmapped_triplet, sen)
    if triplet is False:
        console.print(
            f"[bold red]Could not map annotation {str(triplet)} to subedges)[/bold red]"
        )
        if not ask_user:
            console.print("[bold red]Returning unmapped triplet[/bold red]")
            return unmapped_triplet

        console.print(
            "[bold red]Please provide alternative (or press ENTER to skip)[/bold red]"
        )
        print_tokens(sen, extractor, console)
        triplet = get_single_triplet_from_user(sen, extractor, console)
        if triplet is None:
            console.print("[bold red]No triplet returned[/bold red]")

    return triplet


def get_toks_from_txt(
    words_txt: str, sen: Sentence, ignore_brackets: bool = False
) -> Tuple[int, ...]:
    """
    Map a substring of a sentence to its tokens. Used to parse annotations of triplets
    provided as plain text strings of the predicate and the arguments

    Args:
        words_txt (str): the substring of the sentence
        sen (Sentence): stanza sentence
        ignore_brackets (bool): whether to remove brackets from the text before matching (required for ORE annotation)

    Returns:
        Tuple[int, ...] the tokens of the sentence corresponding to the substring
    """
    logging.debug(f"{words_txt=}, {sen.text=}")
    logging.debug(
        f"enumerated tokens: {[(i, tok.text) for i, tok in enumerate(sen.tokens)]}"
    )
    if ignore_brackets:
        pattern = re.escape(re.sub('["()]', "", words_txt))
    else:
        pattern = re.escape(words_txt)
    logging.debug(f"{pattern=}")
    if pattern[0].isalpha():
        pattern = r"\b" + pattern
    if pattern[-1].isalpha():
        pattern = pattern + r"\b"
    m = re.search(pattern, sen.text, re.IGNORECASE)

    if m is None:
        logging.warning(
            f'Words "{words_txt}" (pattern: "{pattern}") not found in sentence "{sen.text}"'
        )
        raise AnnotatedWordsNotFoundError()

    start, end = m.span()
    logging.debug(f"span: {(start, end)}")

    tok_i, tok_j = None, None
    for i, token in enumerate(sen.tokens):
        if token.start_char == start:
            tok_i = i
        if token.start_char >= end:
            tok_j = i
            break
    if tok_i is None:
        logging.warning(
            f'left side of annotation "{words_txt}" does not match the left side of any token in sen "{sen.text}"'
        )
        raise AnnotatedWordsNotFoundError()
    if tok_j is None:
        tok_j = len(sen.tokens)

    tok_ids_to_return = tuple(range(tok_i, tok_j))
    logging.debug(f"{tok_ids_to_return=}")
    return tok_ids_to_return


def is_power_of_two(n):
    """Returns True iff n is a power of two.  Assumes n > 0."""
    return (n & (n - 1)) == 0


def eliminate_subsets(sequence_of_sets):
    """Return a list of the elements of `sequence_of_sets`, removing all
    elements that are subsets of other elements.  Assumes that each
    element is a set or frozenset and that no element is repeated.
    Based on https://stackoverflow.com/a/14107790/2753770"""
    # The code below does not handle the case of a sequence containing
    # only the empty set, so let's just handle all easy cases now.
    if len(sequence_of_sets) <= 1:
        return list(sequence_of_sets)
    # We need an indexable sequence so that we can use a bitmap to
    # represent each set.
    if not isinstance(sequence_of_sets, collections.abc.Sequence):
        sequence_of_sets = list(sequence_of_sets)
    # For each element, construct the list of all sets containing that
    # element.
    sets_containing_element = {}
    for i, s in enumerate(sequence_of_sets):
        for element in s:
            try:
                sets_containing_element[element] |= 1 << i
            except KeyError:
                sets_containing_element[element] = 1 << i
    # For each set, if the intersection of all of the lists in which it is
    # contained has length != 1, this set can be eliminated.
    out = [
        s
        for s in sequence_of_sets
        if s
        and is_power_of_two(
            reduce(operator.and_, (sets_containing_element[x] for x in s))
        )
    ]
    return out
