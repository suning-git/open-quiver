# qme-ng (Quiver Mutation Explorer - Next Generation)

qme-ng is a C/C++ tool by Gregoire Dupont and Matthieu Perotin for exploring mutation classes and maximal green sequences of quivers. This directory provides a patched, Docker-packaged version ready to use.

- Upstream: https://github.com/mp-bull/qme-ng
- Docker Hub: https://hub.docker.com/r/ningsu/qme-ng

## Quick start (Docker)

Pull the pre-built image (~563 MB):

```bash
docker pull ningsu/qme-ng
```

Run:

```bash
# Show help
docker run --rm ningsu/qme-ng --help

# Compute mutation class of D7
docker run --rm ningsu/qme-ng --type D --size 7

# Compute mutation class of A5
docker run --rm ningsu/qme-ng --type A --size 5
```

### Using a local matrix file

Prepare a matrix file (space or semicolon separated, one row per line):

```
0 1 0;
-1 0 1;
0 -1 0;
```

Mount the directory and run:

```bash
docker run --rm -v /path/to/dir:/data ningsu/qme-ng --file /data/matrix.txt
```

**Windows (Git Bash)**: prefix with `MSYS_NO_PATHCONV=1` to prevent path mangling:

```bash
MSYS_NO_PATHCONV=1 docker run --rm -v "C:\path\to\dir:/data" ningsu/qme-ng --file /data/matrix.txt
```

## Commands

### Mutation class enumeration

Compute the number of non-isomorphic quivers reachable by mutation (BFS + nauty):

```bash
docker run --rm ningsu/qme-ng --type D --size 7
# Output: Set Size:246, The mutation class is acyclic !
```

If the mutation class is infinite, qme-ng detects it and reports early.

### Green sequence exploration (exhaustive)

Find all maximal green sequences and their length distribution (DFS):

```bash
docker run --rm ningsu/qme-ng --type A --size 5 --green
```

### Green sequence finder (randomized)

Find one maximal green sequence via Monte Carlo random walk:

```bash
docker run --rm ningsu/qme-ng --file /data/quiver.txt --green --one 100 --max_depth 50
```

- `--one N`: try N random walks
- `--max_depth N`: abort a walk if it exceeds depth N
- `--min_depth N`: only report sequences of length >= N

## Test run

Using `common/games/A2_square_E7.json` as an example (14-vertex quiver):

1. Extract `B_A` into a text file:

```
0 1 0 0 0 0 0 -1 0 0 0 0 0 0;
-1 0 -1 0 0 0 0 0 1 0 0 0 0 0;
0 1 0 1 0 0 1 0 0 -1 0 0 0 0;
0 0 -1 0 -1 0 0 0 0 0 1 0 0 0;
0 0 0 1 0 1 0 0 0 0 0 -1 0 0;
0 0 0 0 -1 0 0 0 0 0 0 0 1 0;
0 0 -1 0 0 0 0 0 0 0 0 0 0 1;
1 0 0 0 0 0 0 0 -1 0 0 0 0 0;
0 -1 0 0 0 0 0 1 0 1 0 0 0 0;
0 0 1 0 0 0 0 0 -1 0 -1 0 0 -1;
0 0 0 -1 0 0 0 0 0 1 0 1 0 0;
0 0 0 0 1 0 0 0 0 0 -1 0 -1 0;
0 0 0 0 0 -1 0 0 0 0 0 1 0 0;
0 0 0 0 0 0 -1 0 0 1 0 0 0 0;
```

2. Run mutation class (reports infinite):

```bash
docker run --rm -v /path/to/dir:/data ningsu/qme-ng --file /data/A2_square_E7.txt
# Output: Mutation class is infinite !
```

3. Find a green sequence:

```bash
docker run --rm -v /path/to/dir:/data ningsu/qme-ng \
  --file /data/A2_square_E7.txt --green --one 50 --max_depth 30
# Output: Found ! Size: 27
```

## CLI reference

| Flag | Description |
|------|-------------|
| `--file <path>` | Input quiver matrix file |
| `--pefile <path>` | Input principal extension matrix file |
| `--type <T>` | Generate quiver by type: `A`, `D`, `E`, `ATILDE`, `DTILDE`, `ETILDE`, `SPORADIQUE`, `UNAMED`, `E_ELIPTIQUE` |
| `--size <n> [n2]` | Vertex count (with `--type`); second arg is orientation for some types |
| `--green` | Enable green sequence exploration |
| `--one <N>` | Randomized search: try N random walks |
| `--max_depth <N>` | Maximum sequence length before aborting a walk |
| `--min_depth <N>` | Minimum sequence length to report |
| `--p <N>` | Threshold parameter for principal extension |
| `--no-iso` | Disable isomorphism pruning (green exploration only) |
| `--dump-class <prefix>` | Dump mutation class to files |
| `--dump-trunk` | Dump truncated quivers |

## Input file format

Plain `n x n` integer matrix. Supported delimiters: space, tab, `,`, `;`, `[`, `]`.

```
0 1 -1;
-1 0 1;
1 -1 0;
```

Also supports qmu format (see upstream docs).

## Build from source (Docker)

To rebuild the image locally:

```bash
cd common/qme-ng
docker build -t ningsu/qme-ng .
```

This clones upstream qme-ng, applies patches (see `patches.sh`), and compiles with GCC 15 + Boost 1.90 on Arch Linux.

## Build from source (native Linux / WSL)

```bash
sudo apt-get install -y build-essential libgmp-dev libboost-program-options-dev git
git clone https://github.com/mp-bull/qme-ng.git
cd qme-ng
bash /path/to/patches.sh
make
./qme-ng --help
```

## Patches applied

The upstream code requires several fixes to compile and run correctly. All patches are in `patches.sh`.

**Compilation fix (root cause of most build failures):**

- **Makefile `-D__cplusplus` removal**: The upstream Makefile defines `-D__cplusplus`, which overrides the compiler's built-in `__cplusplus` macro to `1`. This breaks C++ standard detection in the standard library and Boost headers. Fix: replace with `-std=gnu++17`.

**Bug fixes from upstream code:**

1. **ATILDE out-of-bounds write** (`src/carquois.cpp`): Loop bound `i<nbSommets` changed to `i<nbSommets-1`
2. **Uninitialized pointers in main** (`qme-ng.cpp`): `carquois/gf/explorator/pt` initialized to `NULL`; unconditional `delete pt` changed to conditional
3. **Missing return in `estDansC()`** (`src/mutexplorator.cpp`): Added `return false` at function end
4. **Uninitialized `ret` variable** (`src/greenfinder.cpp`): `int ret` changed to `int ret = -1`
5. **Windows `sys/times.h` missing** (`include/naututil.h`): Added `_WIN32` branch using `clock()`
6. **File parsing CRLF instability** (`src/principalExtension.cpp`): Strip `\r`, add space between concatenated lines, replace `while(!eof())` pattern

**Additional compatibility fix:**

- **`ss.str("")` type mismatch** (`src/greenSizeHash.cpp`): `const char[1]` cannot implicitly convert to `std::string` in C++17; changed to `ss.str(std::string())`
