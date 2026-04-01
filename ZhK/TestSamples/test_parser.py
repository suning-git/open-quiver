import numpy as np
import re


def parse_test_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    vertices = None
    mutable = None
    sequence = []
    input_lines = []
    output_lines = []

    mode = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith("vertices:"):
            vertices = int(re.search(r"\d+", line).group())
            continue

        if line.startswith("mutable:"):
            mutable = int(re.search(r"\d+", line).group())
            continue

        if line.startswith("sequence:"):
            sequence = [int(x.strip()) for x in line.split(":")[1].split(",")]
            continue

        if line.startswith("input_matrix"):
            mode = "input"
            continue

        if line.startswith("output_matrix"):
            mode = "output"
            continue

        if mode == "input":
            input_lines.append(line)
        elif mode == "output":
            output_lines.append(line)

    def parse_matrix(lines):
        return np.array([
            [int(x) for x in row.split()]
            for row in lines
        ], dtype=int)

    if vertices is None or mutable is None:
        raise ValueError("Missing 'vertices' or 'mutable' in test file")

    input_mat = parse_matrix(input_lines)
    output_mat = parse_matrix(output_lines)

    # checking
    if input_mat.shape != (mutable, vertices):
        raise ValueError(
            f"Input matrix shape mismatch: expected ({mutable}, {vertices}), got {input_mat.shape}"
        )

    if output_mat.shape != (mutable, vertices):
        raise ValueError(
            f"Output matrix shape mismatch: expected ({mutable}, {vertices}), got {output_mat.shape}"
        )

    return {
        "vertices": vertices,
        "mutable": mutable,
        "sequence": sequence,
        "input": input_mat,
        "output": output_mat,
    }