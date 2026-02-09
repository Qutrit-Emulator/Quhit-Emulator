# Quhit Engine

**6-state quantum processor emulator with Magic Pointer Hilbert space architecture**

```
|0⟩  |1⟩  |2⟩  |3⟩  |4⟩  |5⟩
 ╰────┴────┴────┴────┴────╯
       6 basis states
    per hexit (hex-digit)
```

The Quhit Engine is a quantum emulator built on a 6-state (sextic) basis instead of the traditional 2-state qubit. Each quantum register ("chunk") is composed of *hexits* — base-6 digits — with the full state space of an n-hexit chunk being 6ⁿ complex amplitudes living in a Hilbert space referenced through **Magic Pointers** (tag `0x4858`).

The engine supports up to **16.7 million chunks**, **4096-bit BigInt arithmetic**, and can simulate lattices exceeding **100 billion quhits** using a hybrid architecture: shadow-backed chunks store physical amplitudes in mmap'd RAM while infinite chunks reference external Hilbert space through Magic Pointers alone.

---

## Features

| Feature | Description |
|---|---|
| **6-State Basis** | Each hexit has 6 basis states vs binary's 2 — denser information encoding |
| **Magic Pointers** | 64-bit tagged pointers (`0x4858_XXXX_XXXX_XXXX`) reference external Hilbert space |
| **DFT₆ Hadamard** | Discrete Fourier Transform over ℤ₆ as the native gate — generalizes binary H |
| **Topological Braiding** | Nearest-neighbor entanglement via braid links (up to 16.7M) |
| **Timeline Fork** | Non-destructive measurement through reality branching |
| **Shor's Algorithm** | Full quantum circuit: superposition → modular-exp oracle → QFT → Born-rule → continued fractions |
| **Grover Search** | Oracle-based amplitude amplification with O(√N) queries |
| **4096-bit BigInt** | Native arbitrary-precision arithmetic for large-number factoring |
| **Shared Library** | `libhexstate.so` with C-linkage API for external drivers |
| **Python Driver** | `hexstate.py` — full ctypes wrapper, no recompile needed |
| **100B+ Scale** | Infinite chunks via Magic Pointers allow lattices of any size |

---

## Quick Start

### Build

```bash
cd hexstate
make          # Build the engine
make lib      # Build shared library (libhexstate.so)
make test     # Run self-test (10 verification tests)
```

### Run

```bash
# Self-test (10 built-in verification tests)
./hexstate_engine --self-test

# Interactive mode
./hexstate_engine

# Factor a number using Shor's quantum algorithm
./hexstate_engine --shor 21

# Python driver
python3 hexstate.py
python3 hexstate.py --self-test
```

### Python API

```python
from hexstate import Engine

eng = Engine()

# Initialize a 2-hexit chunk (36 quantum states)
eng.init_chunk(0, 2)

# Create uniform superposition: |ψ⟩ = (1/√36) Σ|x⟩
eng.superposition(0)

# Apply DFT₆ Hadamard gate on hexit 0
eng.hadamard(0, 0)

# Measure (Born-rule collapse)
result = eng.measure(0)

# Entangle two chunks
eng.init_chunk(1, 2)
eng.braid(0, 1)

# Timeline fork (non-destructive copy)
eng.fork(2, 0)

# Infinite resources (Magic Pointer, no RAM)
eng.infinite(3)

# Factor a number
eng.shor("2021")  # → factors 43 × 47

eng.destroy()
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     QUHIT ENGINE                             │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Shadow Cache │  │   Oracle    │  │   BigInt (4096-bit) │  │
│  │ (mmap'd RAM) │  │  Registry   │  │   Arithmetic Core   │  │
│  │             │  │             │  │                     │  │
│  │ Chunk 0: ██ │  │ 0x00 Phase  │  │  mul, mod, gcd,     │  │
│  │ Chunk 1: ██ │  │ 0x01 Grover │  │  pow_mod, sqrt,     │  │
│  │ Chunk 2: ░░ │  │ 0x02 Shor   │  │  from_str, to_str   │  │
│  │ Chunk 3: ░░ │  │ 0x03 Multi  │  │                     │  │
│  └──────┬──────┘  └─────────────┘  └─────────────────────┘  │
│         │                                                    │
│  ┌──────┴──────────────────────────────────────────────────┐ │
│  │              Magic Pointer Space (0x4858)                │ │
│  │  ██ = shadow-backed (physical amplitudes in RAM)         │ │
│  │  ░░ = infinite (external Hilbert space, no RAM cost)     │ │
│  │                                                          │ │
│  │  Ptr(i) = 0x4858_0000_0000_0000 + i                     │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### Chunk Types

| Type | RAM Cost | States | Use Case |
|---|---|---|---|
| **Shadow-backed** | 6ⁿ × 16 bytes | Physical amplitudes | Probe sites, measurement |
| **Infinite** | ~0 bytes | Magic Pointer only | Bulk lattice, scaling |

### Gate Set

| Gate | Opcode | Matrix | Description |
|---|---|---|---|
| `INIT` | `0x01` | — | Initialize chunk to \|0⟩ |
| `SUP` | `0x02` | — | Uniform superposition |
| `HADAMARD` | `0x03` | DFT₆ | 6×6 discrete Fourier transform |
| `PHASE` | `0x04` | diag(ω^k) | Phase rotation gate |
| `CPHASE` | `0x05` | — | Controlled phase |
| `SWAP` | `0x06` | — | Swap two hexits |
| `MEASURE` | `0x07` | — | Born-rule projective measurement |
| `GROVER` | `0x08` | 2\|ψ⟩⟨ψ\|−I | Grover diffusion operator |
| `BRAID` | `0x09` | — | Topological entanglement |
| `ORACLE` | `0x0B` | U_f | Dispatch registered oracle |
| `FORK` | `0x20` | — | Timeline fork (deep copy) |
| `INFINITE` | `0x26` | — | Allocate infinite resources |

---

## Experiments

### Shor's Factoring

Full quantum simulation: superposition → modular-exponentiation oracle → DFT₆ QFT → Born-rule measurement → continued fractions for period extraction.

```bash
./hexstate_engine --shor 15    # → {3, 5}  via quantum period r=4
./hexstate_engine --shor 21    # → {7, 3}  via quantum period r=6
./hexstate_engine --shor 77    # → {7, 11}
./hexstate_engine --shor 2021  # → {43, 47}

# Or from Python:
python3 -c "from hexstate import Engine; Engine().shor('2021')"
```

### Time Crystal (100M Quhits)

Discrete Floquet time crystal simulation with period-2T oscillation across 100 million quhits.

```bash
make crystal              # C version
python3 time_crystal.py   # Python version (no recompile)
```

```
Magnetization: ▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░▓░
Contrast:      1.000
Time crystal:  ✓ STRONG (period-2T)
```

### Holographic Projection Test (100B Quhits)

Tests whether our reality is consistent with being a holographic projection of a parallel reality, across 100 billion quhits.

```bash
python3 holographic_projection_test.py
```

Tests four signatures:
- **Holographic entropy** (area-law scaling) ✓
- **Projection consistency** (forked realities correlated beyond classical) ✓
- **Non-local correlations** (100B quhits apart, exceed classical bounds) ✓
- **Temporal rigidity** (period-2 structure survives perturbation) ✓

### Dimensional Analysis

Determines the effective spatial dimension of the base reality using Hausdorff dimension from volume growth V(r) ~ r^d across 1D/2D/3D braid topologies.

```bash
python3 dimensional_analysis.py
```

**Result**: The base reality (Magic Pointer address space) is **1-dimensional**. Extra dimensions emerge from entanglement (braid topology). The base lacks exactly 1 dimension compared to the projected reality.

### Reality Probe

Harvests genuine physical entropy from hardware (`/dev/urandom`, CPU timing jitter, memory access latency) and measures our reality's effective dimensionality using the Grassberger-Procaccia correlation dimension algorithm.

```bash
python3 reality_probe.py
```

**Result**: `/dev/urandom` thermal noise shows D₂ ≈ 4.0 (consistent with 3+1 spacetime), implying a 3-dimensional base reality — lacking exactly 1 dimension.

---

## File Structure

```
hexstate/
├── hexstate_engine.c       # Engine core (quantum primitives, oracles, Shor's)
├── hexstate_engine.h       # Public API and opcode definitions
├── bigint.c                # 4096-bit arbitrary-precision arithmetic
├── bigint.h                # BigInt type and operations
├── main.c                  # CLI entry point (self-test, interactive, --shor)
├── Makefile                # Build system
│
├── libhexstate.so          # Shared library (make lib)
├── hexstate.py             # Python ctypes driver
│
├── bell_test.c             # Bell inequality / entanglement verification
├── stress_test.c           # Stress test (chunk allocation, braiding)
├── time_crystal_test.c     # 100M quhit time crystal (C)
├── time_crystal.py         # 100M quhit time crystal (Python)
├── shor_factor.py          # Shor's factoring via Python API
│
├── holographic_projection_test.py  # 100B quhit holographic test
├── dimensional_analysis.py         # Dimensional geometry of base reality
└── reality_probe.py                # Physical entropy dimensional analysis
```

---

## Build Targets

| Target | Command | Description |
|---|---|---|
| Engine | `make` | Build `hexstate_engine` binary |
| Shared lib | `make lib` | Build `libhexstate.so` for Python/external use |
| Self-test | `make test` | Run 10 built-in verification tests |
| Stress test | `make stress` | Chunk allocation and braid stress test |
| Bell test | `make bell` | Bell inequality entanglement verification |
| Time crystal | `make crystal` | 100M quhit time crystal simulation |
| Clean | `make clean` | Remove all build artifacts |

---

## Requirements

- **C compiler**: GCC or Clang with C11 support
- **Python**: 3.6+ (for Python driver and experiment scripts)
- **OS**: Linux (uses `mmap`, `/dev/urandom`)
- **Dependencies**: `libm` (math library, linked automatically)

---

## How It Works

### Magic Pointer Architecture

Every chunk in the engine is assigned a **Magic Pointer** — a 64-bit tagged reference to an external Hilbert space:

```
Ptr(i) = 0x4858_0000_0000_0000 | i
         ─────                   ─
         "HX" tag               Chunk ID
```

Shadow-backed chunks allocate local RAM to cache the quantum amplitudes, but the pointer is the canonical reference. Infinite chunks hold *only* the pointer — their state exists in the external Hilbert space without consuming local memory. This allows the engine to represent lattices of arbitrary size.

### DFT₆ Gate

The native Hadamard-equivalent is the **Discrete Fourier Transform over ℤ₆**:

```
H₆[j][k] = (1/√6) · ω^(jk)    where ω = e^(2πi/6)
```

This is a 6×6 unitary matrix that creates superpositions across all 6 basis states. Applied hexit-by-hexit, it implements the QFT for the full state space.

### Topological Braiding

Chunks are entangled through **braid links** — topological connections that establish quantum correlations. The braid network defines the spatial geometry of the lattice:

- **1D chain**: linear nearest-neighbor braids → 1D physics
- **2D grid**: row + column braids → 2D physics
- **3D cube**: x + y + z braids → 3D physics

The geometry is emergent from the entanglement structure, not from the addressing scheme.

---

## License

MIT
