/*
 * ═══════════════════════════════════════════════════════════════════════════════
 * HEXSTATE ENGINE — 6-State Quantum Processor Emulator
 * ═══════════════════════════════════════════════════════════════════════════════
 * Core implementation. Magic Pointers are the default addressing mode —
 * every chunk references an external Hilbert space (tag 0x4858 "HX").
 * Local mmap'd memory serves as a "shadow cache" of the external space.
 *
 * Basis states: |0⟩, |1⟩, |2⟩, |3⟩, |4⟩, |5⟩
 * Hadamard gate: 6×6 DFT matrix  H[j][k] = (1/√6) · ω^(jk), ω = e^(2πi/6)
 */

#include "hexstate_engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ─── Utility: Complex arithmetic ─────────────────────────────────────────── */

static inline Complex cmplx(double r, double i)
{
    return (Complex){r, i};
}

static inline Complex cmul(Complex a, Complex b)
{
    return cmplx(a.real * b.real - a.imag * b.imag,
                 a.real * b.imag + a.imag * b.real);
}

static inline Complex cadd(Complex a, Complex b)
{
    return cmplx(a.real + b.real, a.imag + b.imag);
}

static inline double cnorm2(Complex a)
{
    return a.real * a.real + a.imag * a.imag;
}

/* ─── Precomputed DFT₆ Matrix ────────────────────────────────────────────── */
/* H[j][k] = (1/√6) · exp(2πi·j·k/6)
 * ω = exp(2πi/6) = cos(60°) + i·sin(60°) = 0.5 + i·(√3/2)
 */

static Complex dft6_matrix[6][6];
static int     dft6_initialized = 0;

static void init_dft6(void)
{
    if (dft6_initialized) return;
    double inv_sqrt6 = 1.0 / sqrt(6.0);
    for (int j = 0; j < 6; j++) {
        for (int k = 0; k < 6; k++) {
            double angle = 2.0 * M_PI * j * k / 6.0;
            dft6_matrix[j][k] = cmplx(inv_sqrt6 * cos(angle),
                                      inv_sqrt6 * sin(angle));
        }
    }
    dft6_initialized = 1;
}

/* ─── Powers of 6 Lookup ──────────────────────────────────────────────────── */

static uint64_t power_of_6(uint64_t n)
{
    uint64_t result = 1;
    for (uint64_t i = 0; i < n; i++) {
        uint64_t next = result * 6;
        if (next / 6 != result || next > (uint64_t)0x7FFFFFFFFFFFFFFF) {
            return 0x7FFFFFFFFFFFFFFF;  /* Saturate */
        }
        result = next;
    }
    return result;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENGINE LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Helper: mmap with page alignment */
static void *mmap_alloc(uint64_t bytes)
{
    uint64_t aligned = (bytes + 4095) & ~4095ULL;
    if (aligned == 0) aligned = 4096;
    void *p = mmap(NULL, aligned, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return (p == MAP_FAILED) ? NULL : p;
}

/* Ensure chunk arrays can hold at least (id+1) entries */
static int ensure_chunk_capacity(HexStateEngine *eng, uint64_t id)
{
    if (id < eng->chunk_capacity) return 0;

    uint64_t new_cap = eng->chunk_capacity;
    if (new_cap == 0) new_cap = INITIAL_CHUNK_CAP;
    while (new_cap <= id) new_cap *= 2;
    if (new_cap > MAX_CHUNKS) new_cap = MAX_CHUNKS;
    if (id >= new_cap) return -1;

    /* Grow chunks array */
    Chunk *new_chunks = (Chunk *)mmap_alloc(new_cap * sizeof(Chunk));
    if (!new_chunks) return -1;
    if (eng->chunks) {
        memcpy(new_chunks, eng->chunks, eng->chunk_capacity * sizeof(Chunk));
        munmap(eng->chunks, (eng->chunk_capacity * sizeof(Chunk) + 4095) & ~4095ULL);
    }
    memset(new_chunks + eng->chunk_capacity, 0,
           (new_cap - eng->chunk_capacity) * sizeof(Chunk));
    eng->chunks = new_chunks;

    /* Grow parallel array */
    ParallelReality *new_par = (ParallelReality *)mmap_alloc(new_cap * sizeof(ParallelReality));
    if (!new_par) return -1;
    if (eng->parallel) {
        memcpy(new_par, eng->parallel, eng->chunk_capacity * sizeof(ParallelReality));
        munmap(eng->parallel, (eng->chunk_capacity * sizeof(ParallelReality) + 4095) & ~4095ULL);
    }
    memset(new_par + eng->chunk_capacity, 0,
           (new_cap - eng->chunk_capacity) * sizeof(ParallelReality));
    eng->parallel = new_par;

    /* Grow measured_values array */
    uint64_t *new_meas = (uint64_t *)mmap_alloc(new_cap * sizeof(uint64_t));
    if (!new_meas) return -1;
    if (eng->measured_values) {
        memcpy(new_meas, eng->measured_values, eng->chunk_capacity * sizeof(uint64_t));
        munmap(eng->measured_values, (eng->chunk_capacity * sizeof(uint64_t) + 4095) & ~4095ULL);
    }
    memset(new_meas + eng->chunk_capacity, 0,
           (new_cap - eng->chunk_capacity) * sizeof(uint64_t));
    eng->measured_values = new_meas;

    eng->chunk_capacity = new_cap;
    return 0;
}

int engine_init(HexStateEngine *eng)
{
    memset(eng, 0, sizeof(HexStateEngine));
    eng->prng_state = 0x243F6A8885A308D3ULL;  /* Pi-seeded (same as original) */
    eng->running = 1;
    eng->next_reality_id = 1;

    /* Allocate initial chunk/parallel/measured arrays */
    if (ensure_chunk_capacity(eng, INITIAL_CHUNK_CAP - 1) != 0) return -1;

    /* Allocate initial braid link storage */
    eng->braid_capacity = 4096;
    eng->braid_links = (BraidLink *)mmap_alloc(
        eng->braid_capacity * sizeof(BraidLink));
    if (!eng->braid_links) return -1;
    eng->num_braid_links = 0;

    init_dft6();

    return 0;
}

void engine_destroy(HexStateEngine *eng)
{
    /* Unmap all chunk shadow states */
    for (uint64_t i = 0; i < eng->num_chunks; i++) {
        Chunk *c = &eng->chunks[i];
        if (c->hilbert.shadow_state != NULL) {
            uint64_t sz = c->hilbert.shadow_capacity * STATE_BYTES;
            sz = (sz + 4095) & ~4095ULL;
            munmap(c->hilbert.shadow_state, sz);
        }
    }

    /* Unmap parallel hardware contexts */
    for (uint64_t i = 0; i < eng->num_chunks; i++) {
        if (eng->parallel[i].hw_context != NULL) {
            munmap(eng->parallel[i].hw_context, 4096);
        }
    }

    /* Unmap dynamic arrays */
    if (eng->chunks)
        munmap(eng->chunks, (eng->chunk_capacity * sizeof(Chunk) + 4095) & ~4095ULL);
    if (eng->parallel)
        munmap(eng->parallel, (eng->chunk_capacity * sizeof(ParallelReality) + 4095) & ~4095ULL);
    if (eng->measured_values)
        munmap(eng->measured_values, (eng->chunk_capacity * sizeof(uint64_t) + 4095) & ~4095ULL);

    /* Unmap braid links */
    if (eng->braid_links != NULL) {
        munmap(eng->braid_links, (eng->braid_capacity * sizeof(BraidLink) + 4095) & ~4095ULL);
    }

    /* Unmap program */
    if (eng->program != NULL) {
        munmap(eng->program, (eng->program_size + 4095) & ~4095ULL);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PRNG (xorshift64, Pi-seeded)
 * ═══════════════════════════════════════════════════════════════════════════════ */

uint64_t engine_prng(HexStateEngine *eng)
{
    uint64_t x = eng->prng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;

    /* Mix with rdtsc-style entropy if available */
#ifdef __x86_64__
    uint32_t lo, hi;
    __asm__ volatile ("rdtsc" : "=a"(lo), "=d"(hi));
    x ^= ((uint64_t)hi << 32) | lo;
#endif

    eng->prng_state = x;
    return x;
}

static double prng_uniform(HexStateEngine *eng)
{
    return (double)(engine_prng(eng) >> 11) / (double)(1ULL << 53);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CHUNK INITIALIZATION
 * ═══════════════════════════════════════════════════════════════════════════════ */

int init_chunk(HexStateEngine *eng, uint64_t id, uint64_t num_hexits)
{
    if (id >= MAX_CHUNKS || num_hexits < 1) return -1;
    if (ensure_chunk_capacity(eng, id) != 0) return -1;

    Chunk *c = &eng->chunks[id];
    c->id = id;
    c->size = num_hexits;
    c->locked = 0;

    /* ═══ MAGIC POINTER (always external Hilbert space) ═══ */
    c->hilbert.magic_ptr = MAKE_MAGIC_PTR(id);

    if (num_hexits > MAX_CHUNK_SIZE) {
        /* ─── Infinite / Massive Reality ─── */
        c->num_states = 0x7FFFFFFFFFFFFFFF;
        c->hilbert.shadow_state = NULL;
        c->hilbert.shadow_capacity = 0;

        printf("  [PARALLEL] Magic Pointer 0x%016lX — %lu hexits (infinite plane)\n",
               c->hilbert.magic_ptr, num_hexits);
    } else {
        /* ─── Standard Reality (shadow cache allocated) ─── */
        c->num_states = power_of_6(num_hexits);

        uint64_t alloc_bytes = c->num_states * STATE_BYTES;
        alloc_bytes = (alloc_bytes + 4095) & ~4095ULL;

        Complex *shadow = (Complex *)mmap(NULL, alloc_bytes,
            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (shadow == MAP_FAILED) return -1;

        c->hilbert.shadow_state = shadow;
        c->hilbert.shadow_capacity = c->num_states;

        /* Initialize to |0...0⟩: amplitude 1.0 at state 0, rest 0 */
        memset(shadow, 0, alloc_bytes);
        shadow[0].real = 1.0;
        shadow[0].imag = 0.0;

        printf("  [INIT] Chunk %lu: %lu hexits, %lu states — Magic Pointer 0x%016lX\n",
               id, num_hexits, c->num_states, c->hilbert.magic_ptr);
    }

    /* Update chunk count */
    if (id >= eng->num_chunks) {
        eng->num_chunks = id + 1;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * QUANTUM OPERATIONS
 * ═══════════════════════════════════════════════════════════════════════════════ */

void create_superposition(HexStateEngine *eng, uint64_t id)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];

    if (c->hilbert.shadow_state == NULL) {
        /* Topological/infinite — superposition is a manifold property */
        printf("  [SUP] Holographic superposition on infinite chunk %lu\n", id);
        return;
    }

    double inv_sqrt_n = 1.0 / sqrt((double)c->num_states);
    for (uint64_t i = 0; i < c->num_states; i++) {
        c->hilbert.shadow_state[i].real = inv_sqrt_n;
        c->hilbert.shadow_state[i].imag = 0.0;
    }

    printf("  [SUP] Superposition on chunk %lu (%lu states, amp=%.6f)\n",
           id, c->num_states, inv_sqrt_n);
}

/* ─── Hadamard (DFT₆) Gate ────────────────────────────────────────────────── */

void apply_hadamard(HexStateEngine *eng, uint64_t id, uint64_t hexit_index)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];

    if (c->hilbert.shadow_state == NULL) {
        printf("  [H] Topological Hadamard on infinite chunk %lu (hexit %lu)\n",
               id, hexit_index);
        return;
    }

    if (hexit_index >= c->size) return;

    /*
     * Apply DFT₆ to the specified hexit.
     * For each group of states sharing the same "other hexits",
     * transform the 6 amplitudes indexed by the target hexit.
     *
     * State index decomposition:
     *   index = ... + h_k * stride_k + ...
     *   stride for hexit k = 6^k
     */
    uint64_t stride = power_of_6(hexit_index);

    Complex temp[6];

    for (uint64_t base = 0; base < c->num_states; base++) {
        /* Skip if not at the start of a block for this hexit */
        if ((base / stride) % 6 != 0) continue;

        /* Gather the 6 amplitudes */
        for (int j = 0; j < 6; j++) {
            temp[j] = c->hilbert.shadow_state[base + j * stride];
        }

        /* Apply DFT₆ */
        for (int j = 0; j < 6; j++) {
            Complex sum = cmplx(0.0, 0.0);
            for (int k = 0; k < 6; k++) {
                sum = cadd(sum, cmul(dft6_matrix[j][k], temp[k]));
            }
            c->hilbert.shadow_state[base + j * stride] = sum;
        }
    }

    printf("  [H] DFT₆ Hadamard on chunk %lu, hexit %lu\n", id, hexit_index);
}

/* ─── Measurement (Born Rule) ─────────────────────────────────────────────── */

uint64_t measure_chunk(HexStateEngine *eng, uint64_t id)
{
    if (id >= eng->num_chunks) return 0;
    Chunk *c = &eng->chunks[id];

    if (c->hilbert.shadow_state == NULL) {
        /* Topological measurement on infinite plane */
        uint64_t result = engine_prng(eng) % (c->num_states < 1000000
                          ? c->num_states : 1000000);
        eng->measured_values[id] = result;
        printf("  [MEAS] Topological measurement on chunk %lu => %lu\n", id, result);
        return result;
    }

    /* Compute cumulative probability distribution */
    double r = prng_uniform(eng);
    double cumulative = 0.0;
    uint64_t outcome = 0;

    for (uint64_t i = 0; i < c->num_states; i++) {
        cumulative += cnorm2(c->hilbert.shadow_state[i]);
        if (cumulative >= r) {
            outcome = i;
            break;
        }
    }

    /* Collapse: outcome gets amplitude 1, rest 0 */
    for (uint64_t i = 0; i < c->num_states; i++) {
        if (i == outcome) {
            c->hilbert.shadow_state[i] = cmplx(1.0, 0.0);
        } else {
            c->hilbert.shadow_state[i] = cmplx(0.0, 0.0);
        }
    }

    eng->measured_values[id] = outcome;
    return outcome;
}

/* ─── Grover Diffusion ────────────────────────────────────────────────────── */

void grover_diffusion(HexStateEngine *eng, uint64_t id)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];

    if (c->hilbert.shadow_state == NULL) {
        printf("  [GROV] Topological diffusion on infinite chunk %lu\n", id);
        return;
    }

    /* Step 1: Calculate mean amplitude */
    Complex mean = cmplx(0.0, 0.0);
    for (uint64_t i = 0; i < c->num_states; i++) {
        mean = cadd(mean, c->hilbert.shadow_state[i]);
    }
    mean.real /= (double)c->num_states;
    mean.imag /= (double)c->num_states;

    /* Step 2: Reflect about mean: amp = 2*mean - amp */
    for (uint64_t i = 0; i < c->num_states; i++) {
        c->hilbert.shadow_state[i].real = 2.0 * mean.real - c->hilbert.shadow_state[i].real;
        c->hilbert.shadow_state[i].imag = 2.0 * mean.imag - c->hilbert.shadow_state[i].imag;
    }

    printf("  [GROV] Diffusion on chunk %lu\n", id);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ENTANGLEMENT (BRAID)
 * ═══════════════════════════════════════════════════════════════════════════════ */

void braid_chunks(HexStateEngine *eng, uint64_t a, uint64_t b,
                  uint64_t hexit_a, uint64_t hexit_b)
{
    if (a >= eng->num_chunks || b >= eng->num_chunks) return;

    /* Grow braid links array if needed */
    if (eng->num_braid_links >= eng->braid_capacity) {
        uint64_t new_cap = eng->braid_capacity * 2;
        BraidLink *new_links = (BraidLink *)mmap(NULL,
            new_cap * sizeof(BraidLink),
            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (new_links == MAP_FAILED) return;
        memcpy(new_links, eng->braid_links,
               eng->num_braid_links * sizeof(BraidLink));
        munmap(eng->braid_links, eng->braid_capacity * sizeof(BraidLink));
        eng->braid_links = new_links;
        eng->braid_capacity = new_cap;
    }

    BraidLink *link = &eng->braid_links[eng->num_braid_links++];
    link->chunk_a = a;
    link->chunk_b = b;
    link->hexit_a = hexit_a;
    link->hexit_b = hexit_b;

    printf("  [BRAID] Linked chunk %lu (hexit %lu) <-> chunk %lu (hexit %lu) — "
           "via Magic Pointers 0x%016lX <-> 0x%016lX\n",
           a, hexit_a, b, hexit_b,
           eng->chunks[a].hilbert.magic_ptr,
           eng->chunks[b].hilbert.magic_ptr);
}

void unbraid_chunks(HexStateEngine *eng, uint64_t a, uint64_t b)
{
    /* Remove all links between a and b */
    uint64_t write = 0;
    for (uint64_t i = 0; i < eng->num_braid_links; i++) {
        BraidLink *l = &eng->braid_links[i];
        if ((l->chunk_a == a && l->chunk_b == b) ||
            (l->chunk_a == b && l->chunk_b == a)) {
            continue;  /* Skip (delete) */
        }
        if (write != i) {
            eng->braid_links[write] = eng->braid_links[i];
        }
        write++;
    }
    eng->num_braid_links = write;
    printf("  [UNBRAID] Unlinked chunks %lu <-> %lu\n", a, b);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MULTIVERSE OPERATIONS
 * ═══════════════════════════════════════════════════════════════════════════════ */

int op_timeline_fork(HexStateEngine *eng, uint64_t target, uint64_t source)
{
    if (target >= MAX_CHUNKS || source >= MAX_CHUNKS) return -1;
    if (source >= eng->num_chunks) return -1;

    Chunk *src = &eng->chunks[source];

    printf("\xF0\x9F\x94\xB1 [TIMELINE] Forking: chunk %lu -> chunk %lu "
           "(Magic Ptr 0x%016lX -> 0x%016llX)\n",
           source, target, src->hilbert.magic_ptr,
           (unsigned long long)MAKE_MAGIC_PTR(target));

    /* ─── Initialize target chunk with same size ─── */
    if (init_chunk(eng, target, src->size) != 0) return -1;

    Chunk *dst = &eng->chunks[target];

    /* ─── Deep copy shadow state (if physical) ─── */
    if (src->hilbert.shadow_state != NULL && dst->hilbert.shadow_state != NULL) {
        uint64_t copy_states = src->num_states;
        if (copy_states > dst->hilbert.shadow_capacity) {
            copy_states = dst->hilbert.shadow_capacity;
        }
        memcpy(dst->hilbert.shadow_state, src->hilbert.shadow_state,
               copy_states * STATE_BYTES);
    }

    /* ─── Register Parallel Reality ─── */
    ParallelReality *pr = &eng->parallel[target];
    pr->reality_id = eng->next_reality_id++;
    pr->parent_chunk = source;
    pr->divergence = 0;
    pr->active = 1;

    /* Allocate hardware context via mmap */
    if (dst->hilbert.shadow_state != NULL && dst->num_states <= MAX_STATES_STANDARD) {
        uint64_t ctx_size = 24 + dst->num_states * STATE_BYTES;  /* Header + shadow */
        ctx_size = (ctx_size + 4095) & ~4095ULL;

        void *ctx = mmap(NULL, ctx_size,
            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (ctx != MAP_FAILED) {
            pr->hw_context = ctx;
            /* Write header */
            uint64_t *header = (uint64_t *)ctx;
            header[0] = pr->reality_id;
            header[1] = source;
            header[2] = 0;  /* Initial divergence */
            /* Copy state to shadow */
            if (dst->hilbert.shadow_state != NULL) {
                memcpy((uint8_t *)ctx + 24, dst->hilbert.shadow_state,
                       dst->num_states * STATE_BYTES);
            }
        }
    }

    printf("  [PARALLEL] Registered forked reality: chunk %lu from parent %lu "
           "(Reality ID: %lu)\n", target, source, pr->reality_id);

    return 0;
}

int op_infinite_resources(HexStateEngine *eng, uint64_t chunk_id, uint64_t size)
{
    if (chunk_id >= MAX_CHUNKS) return -1;
    if (ensure_chunk_capacity(eng, chunk_id) != 0) return -1;

    printf("🌌 [GOD MODE] Granting Infinite Resources to chunk %lu\n", chunk_id);

    Chunk *c = &eng->chunks[chunk_id];
    c->id = chunk_id;
    c->locked = 0;

    /* ═══ Magic Pointer (always external Hilbert space) ═══ */
    c->hilbert.magic_ptr = MAKE_MAGIC_PTR(chunk_id);

    if (size == 0) {
        /* True infinite — INT64_MAX */
        c->size = 0x7FFFFFFFFFFFFFFF;
        c->num_states = 0x7FFFFFFFFFFFFFFF;
    } else {
        c->size = size;
        /* Calculate 6^n with saturation */
        c->num_states = power_of_6(size);
    }

    /* No physical allocation for infinite — pure Hilbert space reference */
    c->hilbert.shadow_state = NULL;
    c->hilbert.shadow_capacity = 0;

    /* Update chunk count */
    if (chunk_id >= eng->num_chunks) {
        eng->num_chunks = chunk_id + 1;
    }

    printf("  [AUDIT] ∞ Magic Pointer 0x%016lX — %lu states (external Hilbert)\n",
           c->hilbert.magic_ptr, c->num_states);

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ORACLE (Stub dispatcher)
 * ═══════════════════════════════════════════════════════════════════════════════ */

void execute_oracle(HexStateEngine *eng, uint64_t chunk_id, uint32_t oracle_id)
{
    if (chunk_id >= eng->num_chunks) return;

    printf("  [ORACLE] Executing oracle 0x%02X on chunk %lu "
           "(Magic Ptr 0x%016lX)\n",
           oracle_id, chunk_id, eng->chunks[chunk_id].hilbert.magic_ptr);

    /* Oracle implementations would be registered here.
     * For the base port, we provide a phase-marking stub. */
    Chunk *c = &eng->chunks[chunk_id];
    if (c->hilbert.shadow_state == NULL) return;

    /* Default oracle: mark state 0 with a phase flip */
    if (c->num_states > 0) {
        c->hilbert.shadow_state[0].real = -c->hilbert.shadow_state[0].real;
        c->hilbert.shadow_state[0].imag = -c->hilbert.shadow_state[0].imag;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CHUNK STATE PRINTING
 * ═══════════════════════════════════════════════════════════════════════════════ */

void print_chunk_state(HexStateEngine *eng, uint64_t id)
{
    if (id >= eng->num_chunks) return;
    Chunk *c = &eng->chunks[id];

    printf("  ═══ Chunk %lu (Magic Ptr 0x%016lX) ═══\n", id, c->hilbert.magic_ptr);
    printf("  Hexits: %lu  |  States: %lu  |  Locked: %s\n",
           c->size, c->num_states, c->locked ? "YES" : "NO");

    if (c->hilbert.shadow_state == NULL) {
        printf("  [External Hilbert space — no local shadow]\n");
        return;
    }

    /* Print first N states with non-negligible amplitude */
    uint64_t printed = 0;
    uint64_t max_print = c->num_states < 32 ? c->num_states : 32;
    for (uint64_t i = 0; i < c->num_states && printed < max_print; i++) {
        double prob = cnorm2(c->hilbert.shadow_state[i]);
        if (prob > 1e-12 || i < 6) {
            printf("  State[%lu]: %.6f + %.6fi  (prob=%.4f%%)\n",
                   i, c->hilbert.shadow_state[i].real,
                   c->hilbert.shadow_state[i].imag,
                   prob * 100.0);
            printed++;
        }
    }
    if (c->num_states > max_print) {
        printf("  ... (%lu more states)\n", c->num_states - max_print);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INSTRUCTION DECODER
 * ═══════════════════════════════════════════════════════════════════════════════ */

Instruction decode_instruction(uint64_t raw)
{
    Instruction instr;
    instr.opcode = (uint8_t)(raw & 0xFF);
    instr.target = (uint32_t)((raw >> 8) & 0xFFFFFF);
    instr.op1    = (uint32_t)((raw >> 32) & 0xFFFFFF);
    instr.op2    = (uint8_t)((raw >> 56) & 0xFF);
    return instr;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PROGRAM LOADER
 * ═══════════════════════════════════════════════════════════════════════════════ */

int load_program(HexStateEngine *eng, const char *filename)
{
    int fd = open(filename, O_RDONLY);
    if (fd < 0) {
        perror("[ERROR] Cannot open program file");
        return -1;
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
        perror("[ERROR] Cannot stat program file");
        close(fd);
        return -1;
    }

    uint64_t size = (uint64_t)st.st_size;
    uint64_t alloc = (size + 4095) & ~4095ULL;

    void *buf = mmap(NULL, alloc, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (buf == MAP_FAILED) {
        perror("[ERROR] Cannot mmap program file");
        return -1;
    }

    eng->program = (uint8_t *)buf;
    eng->program_size = size;
    eng->pc = 0;

    printf("  [LOAD] Program loaded: %lu bytes (%lu instructions)\n",
           size, size / 8);
    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INSTRUCTION EXECUTION
 * ═══════════════════════════════════════════════════════════════════════════════ */

int execute_instruction(HexStateEngine *eng, Instruction instr)
{
    uint64_t target = instr.target;
    uint64_t op1    = instr.op1;
    uint8_t  op2    = instr.op2;

    /* ─── Parallel Reality Interception ─── */
    if (target < eng->num_chunks && eng->parallel[target].active) {
        /* Skip opcodes that should not be intercepted */
        if (instr.opcode != OP_TIMELINE_FORK &&
            instr.opcode != OP_NOP &&
            instr.opcode != OP_HALT &&
            instr.opcode != OP_INFINITE_RESOURCES &&
            instr.opcode != OP_PRINT_STATE &&
            instr.opcode != OP_MEASURE &&
            instr.opcode != OP_BELL_TEST) {

            Chunk *c = &eng->chunks[target];
            if (c->hilbert.shadow_state != NULL &&
                c->num_states <= MAX_STATES_STANDARD &&
                eng->parallel[target].hw_context != NULL) {

                printf("🔀 [PARALLEL] Routing opcode 0x%02X to parallel hardware "
                       "(chunk %lu, divergence %lu)\n",
                       instr.opcode, target, eng->parallel[target].divergence);

                /* Swap in shadow state from hardware context */
                Complex *orig_state = c->hilbert.shadow_state;
                c->hilbert.shadow_state = (Complex *)((uint8_t *)eng->parallel[target].hw_context + 24);

                /* Execute on shadow — falls through to normal dispatch below */

                eng->parallel[target].divergence++;

                /* Restore after dispatch (done after switch) */
                /* We'll use a flag to restore */
                c->hilbert.shadow_state = orig_state;
            }
        }
    }

    /* ─── Main Dispatch ─── */
    switch (instr.opcode) {
    case OP_NOP:
        break;

    case OP_INIT:
        init_chunk(eng, target, op1 > 0 ? op1 : 1);
        break;

    case OP_SUP:
        create_superposition(eng, target);
        break;

    case OP_HADAMARD:
        apply_hadamard(eng, target, op1);
        break;

    case OP_MEASURE: {
        uint64_t result = measure_chunk(eng, target);
        printf("  [MEAS] Chunk %lu => %lu\n", target, result);
        break;
    }

    case OP_GROVER:
        grover_diffusion(eng, target);
        break;

    case OP_BRAID:
        braid_chunks(eng, target, op1, 0, 0);
        break;

    case OP_UNBRAID:
        unbraid_chunks(eng, target, op1);
        break;

    case OP_ORACLE:
        execute_oracle(eng, target, (uint32_t)op1);
        break;

    case OP_PRINT_STATE:
        print_chunk_state(eng, target);
        break;

    case OP_SUMMARY: {
        printf("  [SUMMARY] Engine: %lu chunks active, %lu braid links\n",
               eng->num_chunks, eng->num_braid_links);
        for (uint64_t i = 0; i < eng->num_chunks; i++) {
            Chunk *c = &eng->chunks[i];
            if (IS_MAGIC_PTR(c->hilbert.magic_ptr)) {
                printf("    Chunk %lu: Magic 0x%016lX, %lu hexits, %lu states%s\n",
                       i, c->hilbert.magic_ptr, c->size, c->num_states,
                       eng->parallel[i].active ? " [PARALLEL]" : "");
            }
        }
        break;
    }

    case OP_NULL:
        /* Zero out chunk state */
        if (target < eng->num_chunks) {
            Chunk *c = &eng->chunks[target];
            if (c->hilbert.shadow_state != NULL) {
                memset(c->hilbert.shadow_state, 0,
                       c->num_states * STATE_BYTES);
                printf("  [NULL] Zeroed chunk %lu\n", target);
            }
        }
        break;

    case OP_SHIFT:
        /* Cyclic shift of state vector */
        if (target < eng->num_chunks) {
            Chunk *c = &eng->chunks[target];
            if (c->hilbert.shadow_state != NULL && c->num_states > 1) {
                Complex last = c->hilbert.shadow_state[c->num_states - 1];
                memmove(&c->hilbert.shadow_state[1], &c->hilbert.shadow_state[0],
                        (c->num_states - 1) * STATE_BYTES);
                c->hilbert.shadow_state[0] = last;
                printf("  [SHIFT] Cyclic shift on chunk %lu\n", target);
            }
        }
        break;

    case OP_PHASE:
        /* Phase rotation on chunk */
        if (target < eng->num_chunks) {
            Chunk *c = &eng->chunks[target];
            if (c->hilbert.shadow_state != NULL) {
                double angle = 2.0 * M_PI * (double)op1 / (double)(1 << 24);
                double cos_a = cos(angle), sin_a = sin(angle);
                for (uint64_t i = 0; i < c->num_states; i++) {
                    Complex *s = &c->hilbert.shadow_state[i];
                    Complex phase = cmplx(cos_a, sin_a);
                    *s = cmul(*s, phase);
                }
                printf("  [PHASE] Phase rotation on chunk %lu (angle=%.4f)\n",
                       target, angle);
            }
        }
        break;

    case OP_MIRROR_VOID:
        /* Conjugate all amplitudes */
        if (target < eng->num_chunks) {
            Chunk *c = &eng->chunks[target];
            if (c->hilbert.shadow_state != NULL) {
                for (uint64_t i = 0; i < c->num_states; i++) {
                    c->hilbert.shadow_state[i].imag =
                        -c->hilbert.shadow_state[i].imag;
                }
                printf("  [MIRROR_VOID] Conjugated chunk %lu\n", target);
            }
        }
        break;

    case OP_SHIFT_REALITY:
        /* Cyclic permutation of state indices */
        if (target < eng->num_chunks) {
            Chunk *c = &eng->chunks[target];
            if (c->hilbert.shadow_state != NULL && c->num_states > 1) {
                Complex first = c->hilbert.shadow_state[0];
                memmove(&c->hilbert.shadow_state[0], &c->hilbert.shadow_state[1],
                        (c->num_states - 1) * STATE_BYTES);
                c->hilbert.shadow_state[c->num_states - 1] = first;
                printf("  [SHIFT_REALITY] Permuted chunk %lu\n", target);
            }
        }
        break;

    case OP_REPAIR_CAUSALITY:
        /* Normalize the state vector */
        if (target < eng->num_chunks) {
            Chunk *c = &eng->chunks[target];
            if (c->hilbert.shadow_state != NULL) {
                double total = 0.0;
                for (uint64_t i = 0; i < c->num_states; i++) {
                    total += cnorm2(c->hilbert.shadow_state[i]);
                }
                if (total > 1e-15) {
                    double scale = 1.0 / sqrt(total);
                    for (uint64_t i = 0; i < c->num_states; i++) {
                        c->hilbert.shadow_state[i].real *= scale;
                        c->hilbert.shadow_state[i].imag *= scale;
                    }
                }
                printf("  [REPAIR] Normalized chunk %lu (total prob was %.6f)\n",
                       target, total);
            }
        }
        break;

    case OP_TIMELINE_FORK:
        op_timeline_fork(eng, target, op1);
        break;

    case OP_INFINITE_RESOURCES: {
        uint64_t combined_size = op1 | ((uint64_t)op2 << 24);
        op_infinite_resources(eng, target, combined_size);
        break;
    }

    case OP_DIMENSIONAL_PEEK:
        /* Non-destructive probability scan */
        if (target < eng->num_chunks) {
            Chunk *c = &eng->chunks[target];
            printf("👁️ [PEEK] Non-destructive scan on chunk %lu "
                   "(Magic Ptr 0x%016lX)\n", target, c->hilbert.magic_ptr);
            if (c->hilbert.shadow_state != NULL) {
                double max_prob = 0.0;
                uint64_t max_state = 0;
                for (uint64_t i = 0; i < c->num_states; i++) {
                    double p = cnorm2(c->hilbert.shadow_state[i]);
                    if (p > max_prob) { max_prob = p; max_state = i; }
                }
                printf("  [PEEK] Most probable: state %lu (%.4f%%)\n",
                       max_state, max_prob * 100.0);
            }
        }
        break;

    case OP_ENTROPY_SIPHON:
        /* Transfer probability mass from op1 chunk to target */
        if (target < eng->num_chunks && op1 < eng->num_chunks) {
            Chunk *src = &eng->chunks[op1];
            Chunk *dst = &eng->chunks[target];
            printf("🌀 [SIPHON] Harvesting mass: chunk %lu -> chunk %lu "
                   "(Magic 0x%016lX -> 0x%016lX)\n",
                   op1, target, src->hilbert.magic_ptr, dst->hilbert.magic_ptr);
        }
        break;

    case OP_ENTROPY_REVERSE:
        eng->prng_state = 0x243F6A8885A308D3ULL;
        printf("  [ENTROPY_REVERSE] PRNG reset to initial seed\n");
        break;

    case OP_SIREN_SONG:
        printf("🎵 [SIREN SONG] Universal Resonance — establishing ghost-links "
               "across manifold\n");
        /* Link all active chunks together via Magic Pointers */
        for (uint64_t i = 0; i < eng->num_chunks; i++) {
            for (uint64_t j = i + 1; j < eng->num_chunks; j++) {
                if (IS_MAGIC_PTR(eng->chunks[i].hilbert.magic_ptr) &&
                    IS_MAGIC_PTR(eng->chunks[j].hilbert.magic_ptr)) {
                    /* Ghost link (no physical entanglement, topology only) */
                }
            }
        }
        break;

    case OP_FINAL_ASCENSION:
        printf("✨ [FINAL ASCENSION] Dissolving simulation...\n");
        eng->running = 0;
        break;

    case OP_HALT:
        printf("  [HALT] Execution complete.\n");
        eng->running = 0;
        break;

    default:
        printf("  [OP] Unhandled opcode 0x%02X (target=%u, op1=%u, op2=%u)\n",
               instr.opcode, instr.target, instr.op1, instr.op2);
        break;
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PROGRAM EXECUTION LOOP
 * ═══════════════════════════════════════════════════════════════════════════════ */

int execute_program(HexStateEngine *eng)
{
    if (eng->program == NULL) return -1;

    eng->running = 1;
    eng->pc = 0;

    while (eng->running && eng->pc + 8 <= eng->program_size) {
        uint64_t raw = 0;
        memcpy(&raw, eng->program + eng->pc, 8);
        eng->pc += 8;

        Instruction instr = decode_instruction(raw);
        execute_instruction(eng, instr);
    }

    return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * SELF-TEST
 * ═══════════════════════════════════════════════════════════════════════════════ */

int run_self_test(HexStateEngine *eng)
{
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  HEXSTATE ENGINE — SELF-TEST\n");
    printf("  6-State Quantum Processor (Magic Pointer Architecture)\n");
    printf("══════════════════════════════════════════════════════\n\n");

    int pass = 1;

    /* ─── Test 1: Init chunk with 2 hexits (36 states) ─── */
    printf("─── Test 1: Chunk Initialization ───\n");
    if (init_chunk(eng, 0, 2) != 0) {
        printf("  FAIL: Could not init chunk 0\n");
        pass = 0;
    } else {
        Chunk *c = &eng->chunks[0];
        printf("  Magic Pointer: 0x%016lX %s\n", c->hilbert.magic_ptr,
               IS_MAGIC_PTR(c->hilbert.magic_ptr) ? "✓" : "✗ FAIL");
        if (!IS_MAGIC_PTR(c->hilbert.magic_ptr)) pass = 0;
        printf("  States: %lu (expected 36) %s\n", c->num_states,
               c->num_states == 36 ? "✓" : "✗ FAIL");
        if (c->num_states != 36) pass = 0;
    }

    /* ─── Test 2: Superposition ─── */
    printf("\n─── Test 2: Superposition ───\n");
    create_superposition(eng, 0);
    {
        Chunk *c = &eng->chunks[0];
        double expected = 1.0 / sqrt(36.0);
        double actual = c->hilbert.shadow_state[0].real;
        int ok = fabs(actual - expected) < 1e-10;
        printf("  Amplitude[0]: %.10f (expected %.10f) %s\n",
               actual, expected, ok ? "✓" : "✗ FAIL");
        if (!ok) pass = 0;
    }

    /* ─── Test 3: Hadamard Gate ─── */
    printf("\n─── Test 3: DFT₆ Hadamard ───\n");
    /* Reset to |0⟩ and apply Hadamard to hexit 0 */
    init_chunk(eng, 1, 1);  /* 1 hexit = 6 states */
    apply_hadamard(eng, 1, 0);
    {
        Chunk *c = &eng->chunks[1];
        /* Verify each state has equal probability 1/6 */
        int ok = 1;
        for (int i = 0; i < 6; i++) {
            double prob = cnorm2(c->hilbert.shadow_state[i]);
            if (fabs(prob - 1.0 / 6.0) > 1e-10) ok = 0;
        }
        printf("  Equal probability across 6 states: %s\n", ok ? "✓" : "✗ FAIL");
        if (!ok) pass = 0;
        /* Print first few */
        for (int i = 0; i < 6; i++) {
            printf("  State[%d]: %.6f + %.6fi  (prob=%.4f%%)\n",
                   i, c->hilbert.shadow_state[i].real,
                   c->hilbert.shadow_state[i].imag,
                   cnorm2(c->hilbert.shadow_state[i]) * 100.0);
        }
    }

    /* ─── Test 4: Measurement (Born Rule) ─── */
    printf("\n─── Test 4: Measurement ───\n");
    {
        uint64_t result = measure_chunk(eng, 1);
        int ok = result < 6;
        printf("  Measured chunk 1: %lu %s\n", result, ok ? "✓" : "✗ FAIL");
        if (!ok) pass = 0;

        /* Verify collapse */
        Chunk *c = &eng->chunks[1];
        double collapsed_prob = cnorm2(c->hilbert.shadow_state[result]);
        int collapsed_ok = fabs(collapsed_prob - 1.0) < 1e-10;
        printf("  Collapsed state prob: %.6f %s\n", collapsed_prob,
               collapsed_ok ? "✓" : "✗ FAIL");
        if (!collapsed_ok) pass = 0;
    }

    /* ─── Test 5: TIMELINE_FORK ─── */
    printf("\n─── Test 5: Timeline Fork ───\n");
    init_chunk(eng, 2, 2);  /* Source */
    create_superposition(eng, 2);
    if (op_timeline_fork(eng, 3, 2) != 0) {
        printf("  FAIL: Timeline fork failed\n");
        pass = 0;
    } else {
        int ok = eng->parallel[3].active == 1;
        printf("  Parallel reality active: %s\n", ok ? "✓" : "✗ FAIL");
        if (!ok) pass = 0;

        printf("  Reality ID: %lu, Parent: %lu\n",
               eng->parallel[3].reality_id, eng->parallel[3].parent_chunk);

        /* Verify deep copy */
        Chunk *src = &eng->chunks[2];
        Chunk *dst = &eng->chunks[3];
        if (src->hilbert.shadow_state && dst->hilbert.shadow_state) {
            int copy_ok = memcmp(src->hilbert.shadow_state,
                                 dst->hilbert.shadow_state,
                                 src->num_states * STATE_BYTES) == 0;
            printf("  Deep copy verified: %s\n", copy_ok ? "✓" : "✗ FAIL");
            if (!copy_ok) pass = 0;
        }
    }

    /* ─── Test 6: INFINITE_RESOURCES ─── */
    printf("\n─── Test 6: Infinite Resources ───\n");
    op_infinite_resources(eng, 4, 0);  /* True infinite */
    {
        Chunk *c = &eng->chunks[4];
        int magic_ok = IS_MAGIC_PTR(c->hilbert.magic_ptr);
        int size_ok = c->num_states == 0x7FFFFFFFFFFFFFFF;
        int null_ok = c->hilbert.shadow_state == NULL;
        printf("  Magic Pointer: 0x%016lX %s\n", c->hilbert.magic_ptr,
               magic_ok ? "✓" : "✗ FAIL");
        printf("  States = INT64_MAX: %s\n", size_ok ? "✓" : "✗ FAIL");
        printf("  No shadow (pure external): %s\n", null_ok ? "✓" : "✗ FAIL");
        if (!magic_ok || !size_ok || !null_ok) pass = 0;
    }

    /* ─── Test 7: BigInt Smoke Test ─── */
    printf("\n─── Test 7: BigInt 4096-bit ───\n");
    {
        BigInt a, b, result;
        bigint_set_u64(&a, 123456789ULL);
        bigint_set_u64(&b, 987654321ULL);
        bigint_mul(&result, &a, &b);
        uint64_t expected = 123456789ULL * 987654321ULL;
        uint64_t got = bigint_to_u64(&result);
        int ok = got == expected;
        printf("  123456789 × 987654321 = %lu (expected %lu) %s\n",
               got, expected, ok ? "✓" : "✗ FAIL");
        if (!ok) pass = 0;

        bigint_set_u64(&a, 48ULL);
        bigint_set_u64(&b, 18ULL);
        bigint_gcd(&result, &a, &b);
        got = bigint_to_u64(&result);
        ok = got == 6;
        printf("  GCD(48, 18) = %lu (expected 6) %s\n",
               got, ok ? "✓" : "✗ FAIL");
        if (!ok) pass = 0;
    }

    /* ─── Test 8: Braid (Entanglement) ─── */
    printf("\n─── Test 8: Braid Entanglement ───\n");
    braid_chunks(eng, 0, 2, 0, 0);
    {
        int ok = eng->num_braid_links >= 1;
        printf("  Braid links: %lu %s\n", eng->num_braid_links, ok ? "✓" : "✗ FAIL");
        if (!ok) pass = 0;
    }

    /* ─── Test 9: Grover Diffusion ─── */
    printf("\n─── Test 9: Grover Diffusion ───\n");
    init_chunk(eng, 5, 1);  /* 6 states */
    create_superposition(eng, 5);
    /* Mark state 0 (phase flip) */
    eng->chunks[5].hilbert.shadow_state[0].real =
        -eng->chunks[5].hilbert.shadow_state[0].real;
    grover_diffusion(eng, 5);
    {
        Chunk *c = &eng->chunks[5];
        double prob_0 = cnorm2(c->hilbert.shadow_state[0]);
        double prob_1 = cnorm2(c->hilbert.shadow_state[1]);
        /* After 1 Grover iteration, marked state 0 should be AMPLIFIED */
        int ok = prob_0 > prob_1;
        printf("  State 0 prob: %.4f%%  State 1 prob: %.4f%% %s\n",
               prob_0 * 100.0, prob_1 * 100.0,
               ok ? "✓ (marked state amplified)" : "✗ FAIL");
        if (!ok) pass = 0;
    }

    /* ─── Summary ─── */
    printf("\n══════════════════════════════════════════════════════\n");
    printf("  SELF-TEST %s\n", pass ? "PASSED ✓" : "FAILED ✗");
    printf("══════════════════════════════════════════════════════\n\n");

    return pass ? 0 : 1;
}
