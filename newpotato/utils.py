from newpotato.datatypes import Triplet


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
            tuple(int(n) for n in ids.split("_")) for ids in annotation.split(",")
        ]
        return phrases[0], phrases[1:]
    except ValueError:
        console.print("[bold red]Could not parse this:[/bold red]", annotation)
        return False


def get_single_triplet_from_user(sentence, hitl, console, expect_mappable=True):
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
        mapped_triplet = hitl.extractor.map_triplet(triplet, sentence)
        if expect_mappable and mapped_triplet is False:
            console.print(
                f"[bold red] Could not map annotation {triplet} to subedges, please provide alternative (or press ENTER to skip)[/bold red]"
            )
            continue

        return mapped_triplet


def get_triplets_from_user(sentence, hitl, console):
    print_tokens(sentence, hitl.extractor, console)

    while True:
        triplet = get_single_triplet_from_user(sentence, hitl, console)
        
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
            raise ValueError('get_single_triplet_from_user must return Triplet, None, "O", or "A"')


def get_triplet_from_annotation(
    pred, args, sen, sen_graph, hitl, console, ask_user=True
):
    triplet = Triplet(pred, args, sen_graph)
    if not triplet.mapped:
        console.print(
            f"[bold red]Could not map annotation {str(triplet)} to subedges)[/bold red]"
        )
        if not ask_user:
            console.print("[bold red]Returning unmapped triplet[/bold red]")
            return triplet

        console.print(
            "[bold red]Please provide alternative (or press ENTER to skip)[/bold red]"
        )
        print_tokens(sen, hitl.extractor, console)
        triplet = get_single_triplet_from_user(sen, hitl, console)
        if triplet is None:
            console.print("[bold red]No triplet returned[/bold red]")
    return triplet
