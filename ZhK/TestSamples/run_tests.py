import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from quiver import ice_quiver_from_exmat, mutate_ice_quiver
from test_parser import parse_test_file


def run_test(path: str) -> bool:
    data = parse_test_file(path)

    q = ice_quiver_from_exmat(data["input"])

    for k in data["sequence"]:
        q = mutate_ice_quiver(q, k)

    result = q.to_exmat()

    ok = np.array_equal(result, data["output"])

    if not ok:
        print(f"[FAIL] {path}")
        print("Expected:")
        print(data["output"])
        print("Got:")
        print(result)
    else:
        print(f"[OK] {path}")

    return ok


def run_all_tests(folder: str):
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".txt")
    ]

    total = len(files)
    passed = 0

    for f in sorted(files):
        if run_test(f):
            passed += 1

    print(f"\nPassed {passed}/{total} tests.")


if __name__ == "__main__":
    run_all_tests("TestSamples/Test1")