import argparse
import json
import random
from typing import List

from quiver import (
        IceQuiver,
        random_quiver,
        frame_quiver,
        mutate_ice_quiver,
        load_quiver,
        format_edgelist,
        format_exchange_matrix,
    )



def _parse_mutation_sequence(s: str) -> List[int]:
    if not s.strip():
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate framed quivers and mutate ice quivers per Rules.md."
    )
    p.add_argument("--n", type=int, help="Number of mutable vertices (for generation mode).")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--p", type=float, help="Edge probability among mutable pairs (generation mode).")
    g.add_argument("--edges", type=int, help="Number of base edges among mutable vertices.")

    p.add_argument("--input-file", type=str, default="", help="Path to an existing ice quiver.")
    p.add_argument(
        "--input-format",
        choices=["json", "edgelist", "exmat"],
        default="json",
        help="Format of --input-file.",
    )
    p.add_argument(
        "--allow-multi-arrows",
        action="store_true",
        help="Generation mode only: allow multiple arrows with the same source and target.",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        help="Generation mode: max multiplicity per chosen base edge direction.",
    )
    p.add_argument(
        "--mutate",
        type=str,
        default="",
        help="Comma-separated sequence of mutable vertices to mutate at, e.g. '2,1,3'.",
    )
    p.add_argument("--seed", type=int, default=None, help="PRNG seed for generation mode.")
    p.add_argument("--format", choices=["json", "edgelist", "exmat"], default="json", help="Output format.")
    p.add_argument("--output-file", type=str, default="", help="If set, write output to this file.")
    args = p.parse_args()

    if args.input_file:
        # --allow-multi-arrows only affects random generation, not mutation on input quivers.
        q = load_quiver(args.input_file, args.input_format)
    else:
        if args.n is None:
            raise ValueError("Generation mode requires --n")
        if (args.p is None) == (args.edges is None):
            raise ValueError("Generation mode requires exactly one of --p or --edges")
        rng = random.Random(args.seed)
        base = random_quiver(
            args.n,
            rng=rng,
            edge_probability=args.p,
            edges=args.edges,
            allow_multi_arrows=args.allow_multi_arrows,
            max_parallel=args.max_parallel,
        )
        # q = frame_quiver_old(args.n, base)
        q = frame_quiver(IceQuiver(vertices=list(range(1,args.n)), frozen=[], arrow_counts=base))

    for k in _parse_mutation_sequence(args.mutate):
        q = mutate_ice_quiver(q, k)

    if args.format == "json":
        out = json.dumps(q.to_dict(), indent=2, sort_keys=True) + "\n"
    elif args.format == "edgelist":
        out = format_edgelist(q)
    else:
        out = format_exchange_matrix(q)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(out)
    else:
        print(out, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())