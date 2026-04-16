import argparse
import json
from pathlib import Path
from typing import Dict, List

from quiver import frame_quiver, is_red_framed, load_quiver, mutate_ice_quiver


def _mutable_vertices(q) -> List[int]:
    return [v for v in q.vertices if v not in q.frozen]


def _max_edge_mult(q) -> int:
    return max(q.arrow_counts.values(), default=0)


def _exmat_one_line(q) -> str:
    mat = q.to_exmat()
    rows = [" ".join(str(int(x)) for x in row) for row in mat.tolist()]
    return " ; ".join(rows)


def enumerate_sequences(
    input_file: Path,
    input_format: str,
    max_depth: int,
) -> Dict[str, List[Dict]]:
    base = load_quiver(str(input_file), input_format)
    q0 = frame_quiver(base)
    mutables = _mutable_vertices(q0)

    groups: Dict[str, List[Dict]] = {
        "leq5_red_false": [],
        "leq5_all": [],
        "eq5_red_false": [],
        "eq5_all": [],
    }

    def push_record(rec: Dict, ever_mutated_red: bool, depth: int) -> None:
        groups["leq5_all"].append(rec)
        if depth == max_depth:
            groups["eq5_all"].append(rec)
        if not ever_mutated_red:
            groups["leq5_red_false"].append(rec)
            if depth == max_depth:
                groups["eq5_red_false"].append(rec)

    def dfs(
        q,
        depth: int,
        sequence: List[int],
        last_mut: int,
        ever_mutated_red: bool,
        path_max_mult: int,
    ) -> None:
        rec = {
            "depth": depth,
            "sequence": sequence[:],
            "ever_mutated_red": ever_mutated_red,
            "path_max_edge_multiplicity": path_max_mult,
            "final_state_exmat": _exmat_one_line(q),
            "final_state_max_edge_multiplicity": _max_edge_mult(q),
        }
        push_record(rec, ever_mutated_red, depth)

        if depth == max_depth:
            return

        for k in mutables:
            if last_mut is not None and k == last_mut:
                continue
            red_now = is_red_framed(q, k)
            q2 = mutate_ice_quiver(q, k)
            next_path_max = max(path_max_mult, _max_edge_mult(q2))
            dfs(
                q2,
                depth + 1,
                sequence + [k],
                k,
                ever_mutated_red or red_now,
                next_path_max,
            )

    dfs(
        q=q0,
        depth=0,
        sequence=[],
        last_mut=None,
        ever_mutated_red=False,
        path_max_mult=_max_edge_mult(q0),
    )

    return groups


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate framed-quiver mutation paths up to depth D with adjacent "
            "mutations constrained to be different, and export 4 sequence files "
            "by (ever_mutated_red true/false) x (depth <= D / depth == D)."
        )
    )
    parser.add_argument("--input-file", type=str, default="K5.exmat")
    parser.add_argument("--input-format", choices=["json", "edgelist", "exmat"], default="exmat")
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--leq-red-false-file", type=str, default="sequences_depth_leq_max_red_false.jsonl")
    parser.add_argument("--leq-all-file", type=str, default="sequences_depth_leq_max_all.jsonl")
    parser.add_argument("--eq-red-false-file", type=str, default="sequences_depth_eq_max_red_false.jsonl")
    parser.add_argument("--eq-all-file", type=str, default="sequences_depth_eq_max_all.jsonl")
    args = parser.parse_args()

    if args.max_depth < 0:
        raise ValueError("--max-depth must be >= 0")

    input_path = Path(args.input_file)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = enumerate_sequences(
        input_file=input_path,
        input_format=args.input_format,
        max_depth=args.max_depth,
    )

    path_map = {
        "leq5_red_false": output_dir / args.leq_red_false_file,
        "leq5_all": output_dir / args.leq_all_file,
        "eq5_red_false": output_dir / args.eq_red_false_file,
        "eq5_all": output_dir / args.eq_all_file,
    }

    for key, out_path in path_map.items():
        with out_path.open("w", encoding="utf-8") as f:
            for item in groups[key]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Exported: {path_map['leq5_red_false']}")
    print(f"Exported: {path_map['leq5_all']}")
    print(f"Exported: {path_map['eq5_red_false']}")
    print(f"Exported: {path_map['eq5_all']}")
    print(
        "Counts: "
        f"leq_red_false={len(groups['leq5_red_false'])}, "
        f"leq_all={len(groups['leq5_all'])}, "
        f"eq_red_false={len(groups['eq5_red_false'])}, "
        f"eq_all={len(groups['eq5_all'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
