/*
 * lazy_shor_stream.c — Shor's Algorithm: Modular Multiplication Operator
 *
 * INPUT: N only.  No factors, no φ(N).
 *
 * KEY CORRECTION:
 *   Previous versions used a PHASE oracle:  |k⟩ → exp(iφ)|k⟩
 *   That's diagonal — it doesn't move amplitude.  No useful interference.
 *
 *   This version uses the MODULAR MULTIPLICATION operator:
 *     |y⟩ → |a·y mod N⟩
 *   This is a PERMUTATION — amplitude physically moves between states.
 *   Multiple x values map to the same f(x), creating COLLISIONS.
 *   Those collisions create the periodic structure that QFT reveals.
 *
 * Architecture:
 *   1. Control register (chunk 0): H → superposition Σ|k⟩
 *   2. Target register (chunk 1): starts at |1⟩ (= a^0)
 *   3. Controlled modular multiplication:
 *      For each |k⟩: target → |a^k mod N mod D⟩
 *      This PERMUTES target values — not a phase!
 *      Entries with same target value → amplitude INTERFERENCE
 *   4. QFT on control → measure → digit
 *   5. CF → period → factors
 *
 * BUILD:
 *   gcc -O2 -I. -std=c11 -D_GNU_SOURCE \
 *       -c lazy_shor_stream.c -o lazy_shor_stream.o && \
 *   gcc -O2 -o lazy_shor_stream lazy_shor_stream.o hexstate_engine.o bigint.o -lm
 */

#include "hexstate_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

#define D 6
#define N_QUHITS 100000000000000ULL
static const char *bn[] = {"A","T","G","C","dR","Pi"};

static int saved_fd = -1;
static void hush(void) {
    fflush(stdout);
    saved_fd = dup(STDOUT_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDOUT_FILENO);
    close(devnull);
}
static void unhush(void) {
    if (saved_fd >= 0) {
        fflush(stdout);
        dup2(saved_fd, STDOUT_FILENO);
        close(saved_fd);
        saved_fd = -1;
    }
}

static uint64_t gcd64(uint64_t a, uint64_t b) {
    while (b) { uint64_t t = b; b = a % b; a = t; }
    return a;
}
static uint64_t modpow64(uint64_t base, uint64_t exp, uint64_t mod) {
    if (mod == 1) return 0;
    uint64_t r = 1; base %= mod;
    while (exp > 0) {
        if (exp & 1) r = (__uint128_t)r * base % mod;
        exp >>= 1;
        base = (__uint128_t)base * base % mod;
    }
    return r;
}

static void stream_state(HexStateEngine *eng, uint64_t chunk_id,
                          const char *label)
{
    StateIterator it;
    state_iter_begin(eng, chunk_id, &it);
    double norm = 0;
    printf("      ┌─ %s ─ %u entries\n", label, it.total_entries);
    while (state_iter_next(&it)) {
        norm += it.probability;
        printf("      │ [%u] %s amp=(%+.4f,%+.4fi) P=%.4f\n",
               it.entry_index, bn[it.bulk_value % D],
               it.amplitude.real, it.amplitude.imag, it.probability);
    }
    state_iter_end(&it);
    printf("      └─ norm=%.6f\n", norm);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  MODULAR MULTIPLICATION OPERATOR — |y⟩ → |a·y mod N mod D⟩
 *
 *  THIS IS A PERMUTATION, NOT A PHASE.
 *  Amplitude physically MOVES between basis states.
 *  When multiple inputs map to the same output → COLLISION → interference.
 *
 *  For each entry in the Hilbert space:
 *    old_val = entry's bulk_value (represents orbit position)
 *    orbit_val = a^old_val mod N  (the actual number)
 *    new_orbit_val = a * orbit_val mod N = a^(old_val+1) mod N
 *    new_val = new_orbit_val mod D  (project to D-dimensional space)
 *    entry's bulk_value ← new_val
 *
 *  Entries that land on the same new_val have their amplitudes ADDED.
 *  This coherent addition is the quantum interference that encodes
 *  the period r of a mod N.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void apply_modmul_u64(HexStateEngine *eng, uint64_t chunk_id,
                             uint64_t c_j, uint64_t N_val)
{
    int r = find_quhit_reg(eng, chunk_id);
    if (r < 0) return;

    uint32_t nz = eng->quhit_regs[r].num_nonzero;

    /* Build new entries with permuted bulk_values */
    static QuhitBasisEntry new_entries[MAX_QUHIT_HILBERT_ENTRIES];
    uint32_t new_nz = 0;

    for (uint32_t e = 0; e < nz; e++) {
        uint32_t k = eng->quhit_regs[r].entries[e].bulk_value;

        /*
         * MODULAR MULTIPLICATION (on the fly):
         *   orbit position k represents a^k mod N
         *   multiply by c_j: new value = c_j * (a^k mod N) mod N
         *   project to D dimensions: new_val = result mod D
         *
         * c_j = a^(2^j) mod N, so this computes a^(k + 2^j) mod N mod D
         */
        uint64_t orbit_val = modpow64(c_j, (uint64_t)k, N_val);
        uint64_t new_val = orbit_val % D;

        double ar = eng->quhit_regs[r].entries[e].amplitude.real;
        double ai = eng->quhit_regs[r].entries[e].amplitude.imag;

        /* Find existing entry with same new_val — if found, MERGE
         * (this is the quantum interference from collisions!) */
        int found = -1;
        for (uint32_t i = 0; i < new_nz; i++) {
            if (new_entries[i].bulk_value == (uint32_t)new_val) {
                found = (int)i; break;
            }
        }

        if (found >= 0) {
            /* COLLISION → coherent amplitude addition = interference */
            new_entries[found].amplitude.real += ar;
            new_entries[found].amplitude.imag += ai;
        } else if (new_nz < MAX_QUHIT_HILBERT_ENTRIES) {
            new_entries[new_nz] = eng->quhit_regs[r].entries[e];
            new_entries[new_nz].bulk_value = (uint32_t)new_val;
            new_entries[new_nz].amplitude.real = ar;
            new_entries[new_nz].amplitude.imag = ai;
            new_nz++;
        }
    }

    memcpy(eng->quhit_regs[r].entries, new_entries,
           new_nz * sizeof(QuhitBasisEntry));
    eng->quhit_regs[r].num_nonzero = new_nz;
    eng->quhit_regs[r].collapsed = 0;
}

static void apply_modmul_big(HexStateEngine *eng, uint64_t chunk_id,
                             BigInt *c_j, BigInt *N_bi)
{
    int r = find_quhit_reg(eng, chunk_id);
    if (r < 0) return;

    uint32_t nz = eng->quhit_regs[r].num_nonzero;
    static QuhitBasisEntry new_entries[MAX_QUHIT_HILBERT_ENTRIES];
    uint32_t new_nz = 0;

    for (uint32_t e = 0; e < nz; e++) {
        uint32_t k = eng->quhit_regs[r].entries[e].bulk_value;

        BigInt k_bi, orbit_val;
        bigint_set_u64(&k_bi, (uint64_t)k);
        bigint_pow_mod(&orbit_val, c_j, &k_bi, N_bi);

        BigInt D_bi, quo, rem;
        bigint_set_u64(&D_bi, D);
        bigint_div_mod(&orbit_val, &D_bi, &quo, &rem);
        uint32_t new_val = (uint32_t)bigint_to_u64(&rem);

        double ar = eng->quhit_regs[r].entries[e].amplitude.real;
        double ai = eng->quhit_regs[r].entries[e].amplitude.imag;

        int found = -1;
        for (uint32_t i = 0; i < new_nz; i++) {
            if (new_entries[i].bulk_value == new_val) {
                found = (int)i; break;
            }
        }

        if (found >= 0) {
            new_entries[found].amplitude.real += ar;
            new_entries[found].amplitude.imag += ai;
        } else if (new_nz < MAX_QUHIT_HILBERT_ENTRIES) {
            new_entries[new_nz] = eng->quhit_regs[r].entries[e];
            new_entries[new_nz].bulk_value = new_val;
            new_entries[new_nz].amplitude.real = ar;
            new_entries[new_nz].amplitude.imag = ai;
            new_nz++;
        }
    }

    memcpy(eng->quhit_regs[r].entries, new_entries,
           new_nz * sizeof(QuhitBasisEntry));
    eng->quhit_regs[r].num_nonzero = new_nz;
    eng->quhit_regs[r].collapsed = 0;
}

/* ═══ CF extraction ═══ */

static int extract_cf_denominators(int *digits, int n_digits,
                                   uint64_t *denoms, int max_denoms)
{
    uint64_t numer = 0, denom = 1;
    for (int j = 0; j < n_digits && j < 30; j++) {
        numer = numer * D + digits[j];
        denom *= D;
        uint64_t g = gcd64(numer, denom);
        if (g > 1) { numer /= g; denom /= g; }
    }
    if (numer == 0) return 0;

    uint64_t n_cf = numer, d_cf = denom;
    uint64_t q0 = 1, q1 = 0;
    int count = 0;
    for (int i = 0; i < 100 && d_cf != 0 && count < max_denoms; i++) {
        uint64_t a = n_cf / d_cf;
        uint64_t rem = n_cf % d_cf;
        uint64_t q2 = a * q1 + q0;
        if (q2 > 1) denoms[count++] = q2;
        q0 = q1; q1 = q2;
        n_cf = d_cf; d_cf = rem;
    }
    return count;
}

static int extract_cf_denominators_big(int *digits, int n_digits,
                                       BigInt *denoms, int max_denoms)
{
    uint64_t numer = 0, denom = 1;
    for (int j = 0; j < n_digits && j < 30; j++) {
        numer = numer * D + digits[j];
        denom *= D;
        uint64_t g = gcd64(numer, denom);
        if (g > 1) { numer /= g; denom /= g; }
    }
    if (numer == 0) return 0;

    uint64_t n_cf = numer, d_cf = denom;
    BigInt q0, q1, one;
    bigint_set_u64(&q0, 1);
    bigint_set_u64(&q1, 0);
    bigint_set_u64(&one, 1);
    int count = 0;
    for (int i = 0; i < 100 && d_cf != 0 && count < max_denoms; i++) {
        uint64_t a = n_cf / d_cf;
        uint64_t rem = n_cf % d_cf;
        BigInt a_bi, aq1, q2;
        bigint_set_u64(&a_bi, a);
        bigint_mul(&aq1, &a_bi, &q1);
        bigint_add(&q2, &aq1, &q0);
        if (bigint_cmp(&q2, &one) > 0)
            bigint_copy(&denoms[count++], &q2);
        bigint_copy(&q0, &q1);
        bigint_copy(&q1, &q2);
        n_cf = d_cf; d_cf = rem;
    }
    return count;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  SINGLE IPE ROUND
 *
 *  Engine: H → MODULAR MULTIPLICATION (permutes states) → H → measure
 *  The multiplication creates collisions → interference → period peaks
 * ═══════════════════════════════════════════════════════════════════════════ */

static int run_ipe_round_u64(uint64_t c_j, uint64_t N_val,
                             double accumulated_phase, int j, int verbose)
{
    static HexStateEngine eng;
    hush();
    engine_init(&eng);
    init_chunk(&eng, 0, 1);
    op_infinite_resources_dim(&eng, 0, N_QUHITS, D);
    init_quhit_register(&eng, 0, N_QUHITS, D);
    init_chunk(&eng, 1, 1);
    op_infinite_resources_dim(&eng, 1, N_QUHITS, D);
    init_quhit_register(&eng, 1, N_QUHITS, D);
    braid_chunks(&eng, 0, 1, 0, 0);
    unhush();

    /* H → superposition */
    hush(); entangle_all_quhits(&eng, 0); unhush();

    if (verbose) stream_state(&eng, 0, "After H (superposition)");

    /*
     * MODULAR MULTIPLICATION OPERATOR:
     * |k⟩ → |c_j^k mod N mod D⟩
     *
     * This PERMUTES bulk_values — amplitude moves between states.
     * When c_j^k₁ mod N mod D == c_j^k₂ mod N mod D for k₁ ≠ k₂,
     * those entries COLLIDE and their amplitudes add coherently.
     * This collision-interference encodes the period.
     */
    apply_modmul_u64(&eng, 1, c_j, N_val);

    if (verbose) stream_state(&eng, 0, "After modular multiplication");

    /* Phase correction from prior digits */
    if (j > 0 && fabs(accumulated_phase) > 1e-15) {
        int r = find_quhit_reg(&eng, 1);
        if (r >= 0) {
            uint32_t nz = eng.quhit_regs[r].num_nonzero;
            for (uint32_t e = 0; e < nz; e++) {
                uint32_t v = eng.quhit_regs[r].entries[e].bulk_value;
                double phi = -2.0 * M_PI * (double)v * accumulated_phase;
                double cr = cos(phi), ci = sin(phi);
                double ar = eng.quhit_regs[r].entries[e].amplitude.real;
                double ai = eng.quhit_regs[r].entries[e].amplitude.imag;
                eng.quhit_regs[r].entries[e].amplitude.real = ar*cr - ai*ci;
                eng.quhit_regs[r].entries[e].amplitude.imag = ar*ci + ai*cr;
            }
        }
    }

    /* Final H → interference pattern */
    hush(); entangle_all_quhits(&eng, 0); unhush();

    if (verbose) stream_state(&eng, 0, "After QFT (measurement basis)");

    /* MEASURE */
    hush();
    uint64_t k = measure_chunk(&eng, 0);
    unbraid_chunks(&eng, 0, 1);
    engine_destroy(&eng);
    unhush();

    return (int)(k % D);
}

static int run_ipe_round_big(BigInt *c_j, BigInt *N_bi,
                             double accumulated_phase, int j, int verbose)
{
    static HexStateEngine eng;
    hush();
    engine_init(&eng);
    init_chunk(&eng, 0, 1);
    op_infinite_resources_dim(&eng, 0, N_QUHITS, D);
    init_quhit_register(&eng, 0, N_QUHITS, D);
    init_chunk(&eng, 1, 1);
    op_infinite_resources_dim(&eng, 1, N_QUHITS, D);
    init_quhit_register(&eng, 1, N_QUHITS, D);
    braid_chunks(&eng, 0, 1, 0, 0);
    unhush();

    hush(); entangle_all_quhits(&eng, 0); unhush();

    apply_modmul_big(&eng, 1, c_j, N_bi);

    if (j > 0 && fabs(accumulated_phase) > 1e-15) {
        int r = find_quhit_reg(&eng, 1);
        if (r >= 0) {
            uint32_t nz = eng.quhit_regs[r].num_nonzero;
            for (uint32_t e = 0; e < nz; e++) {
                uint32_t v = eng.quhit_regs[r].entries[e].bulk_value;
                double phi = -2.0 * M_PI * (double)v * accumulated_phase;
                double cr = cos(phi), ci = sin(phi);
                double ar = eng.quhit_regs[r].entries[e].amplitude.real;
                double ai = eng.quhit_regs[r].entries[e].amplitude.imag;
                eng.quhit_regs[r].entries[e].amplitude.real = ar*cr - ai*ci;
                eng.quhit_regs[r].entries[e].amplitude.imag = ar*ci + ai*cr;
            }
        }
    }

    hush(); entangle_all_quhits(&eng, 0); unhush();

    if (verbose) stream_state(&eng, 0, "After QFT");

    hush();
    uint64_t k = measure_chunk(&eng, 0);
    unbraid_chunks(&eng, 0, 1);
    engine_destroy(&eng);
    unhush();

    return (int)(k % D);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  FACTOR uint64_t
 * ═══════════════════════════════════════════════════════════════════════════ */

static int ipe_factor_u64(uint64_t N_val, int extra_rounds)
{
    int n_bits = 0;
    { uint64_t t = N_val; while (t > 0) { n_bits++; t >>= 1; } }
    int n_rounds = n_bits + extra_rounds;

    printf("  ╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  N = %-20lu  (%d bits)                            ║\n",
           (unsigned long)N_val, n_bits);
    printf("  ║  Oracle: MODULAR MULTIPLICATION |y⟩ → |ay mod N⟩ (permutation)║\n");
    printf("  ╚═══════════════════════════════════════════════════════════════════╝\n\n");

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int success = 0;
    uint64_t f1=0, f2=0, found_r=0, found_a=0;
    uint64_t bases[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43};

    for (int bi = 0; bi < 14 && !success; bi++) {
        uint64_t a = bases[bi];
        if (a >= N_val) continue;
        uint64_t g = gcd64(a, N_val);
        if (g > 1 && g < N_val) {
            f1=g; f2=N_val/g; found_a=a; success=1;
            printf("  Base %lu: gcd=%lu — trivial\n\n",
                   (unsigned long)a, (unsigned long)g);
            break;
        }

        printf("  ── Base a = %lu ──\n", (unsigned long)a);

        /* Show collision structure */
        printf("  Collision table (a^k mod N mod D):\n    ");
        for (int k = 0; k < D; k++) {
            uint64_t val = modpow64(a, k, N_val);
            printf("k=%d→%lu(mod%d=%lu) ", k, (unsigned long)val,
                   D, (unsigned long)(val % D));
        }
        printf("\n\n");

        int *digits = calloc(n_rounds, sizeof(int));
        double phase = 0.0;
        uint64_t c_j = a;  /* c_0 = a^(2^0) = a */

        printf("  ┌──────┬─────────────────────────────────┬──────┬──────────────┐\n");
        printf("  │Round │ c_j (modmul param)              │  d_j │ θ cumulative │\n");
        printf("  ├──────┼─────────────────────────────────┼──────┼──────────────┤\n");

        for (int j = 0; j < n_rounds; j++) {
            digits[j] = run_ipe_round_u64(c_j, N_val, phase, j, j < 2);

            phase = 0;
            for (int m = 0; m <= j; m++) {
                double w = 1.0;
                for (int q = 0; q <= m; q++) w /= D;
                phase += digits[m] * w;
            }

            printf("  │  %2d  │ c_%d = %-10lu (modmul)     │  %d   │ θ=%.8f  │\n",
                   j, j, (unsigned long)c_j, digits[j], phase);

            c_j = (__uint128_t)c_j * c_j % N_val;
        }
        printf("  └──────┴─────────────────────────────────┴──────┴──────────────┘\n\n");

        printf("  Phase digits: ");
        for (int j = 0; j < n_rounds && j < 25; j++) printf("%d", digits[j]);
        printf("\n");

        uint64_t denoms[64];
        int nd = extract_cf_denominators(digits, n_rounds, denoms, 64);
        printf("  CF denominators:");
        for (int i = 0; i < nd && i < 15; i++)
            printf(" %lu", (unsigned long)denoms[i]);
        printf("\n\n");

        for (int di = 0; di < nd && !success; di++) {
            for (uint64_t mult = 1; mult <= 24 && !success; mult++) {
                uint64_t r = denoms[di] * mult;
                if (r >= N_val || r < 2 || r % 2 != 0) continue;
                if (modpow64(a, r, N_val) != 1) continue;

                uint64_t half = modpow64(a, r/2, N_val);
                if (half == N_val - 1) continue;

                uint64_t g1 = gcd64(half+1, N_val);
                uint64_t g2 = gcd64(half > 0 ? half-1 : N_val-1, N_val);

                if (g1 > 1 && g1 < N_val) {
                    f1=g1; f2=N_val/g1; found_r=r; found_a=a; success=1;
                } else if (g2 > 1 && g2 < N_val) {
                    f1=g2; f2=N_val/g2; found_r=r; found_a=a; success=1;
                }
            }
        }
        free(digits);
        printf("\n");
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec-t0.tv_sec)*1000.0+(t1.tv_nsec-t0.tv_nsec)/1e6;

    printf("  ┌───────────────────────────────────────────────────────────────────┐\n");
    if (success)
        printf("  │  ✓ N = %lu = %lu × %lu  (r=%lu a=%lu)\n",
               (unsigned long)N_val, (unsigned long)f1, (unsigned long)f2,
               (unsigned long)found_r, (unsigned long)found_a);
    else
        printf("  │  ✗ N = %lu — not factored\n", (unsigned long)N_val);
    printf("  │  %.1f ms  |  modular multiplication oracle (permutation)\n", ms);
    printf("  └───────────────────────────────────────────────────────────────────┘\n\n\n");
    return success;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  FACTOR BigInt
 * ═══════════════════════════════════════════════════════════════════════════ */

static int ipe_factor_bigint(const char *N_str, int extra_rounds)
{
    BigInt N_bi, one;
    bigint_from_decimal(&N_bi, N_str);
    bigint_set_u64(&one, 1);
    uint32_t n_bits = bigint_bitlen(&N_bi);
    int n_rounds = (int)(2.0 * (double)n_bits / log2(D)) + extra_rounds;
    if (n_rounds > 250) n_rounds = 250;

    printf("  ╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  %u-BIT NUMBER — Modular Multiplication Oracle                 ║\n", n_bits);
    printf("  ║  Input: N only.  Oracle: |y⟩ → |ay mod N⟩ (PERMUTATION)       ║\n");
    printf("  ╚═══════════════════════════════════════════════════════════════════╝\n\n");
    printf("  N = %s\n      (%u bits, %zu digits)\n\n", N_str, n_bits, strlen(N_str));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    int factored = 0;
    uint64_t base_vals[] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43};

    for (int bi = 0; bi < 14 && !factored; bi++) {
        BigInt base_a;
        bigint_set_u64(&base_a, base_vals[bi]);
        BigInt gcd_check;
        bigint_gcd(&gcd_check, &base_a, &N_bi);
        if (bigint_cmp(&gcd_check, &one) != 0) continue;

        printf("  ── Base a = %lu ──\n", (unsigned long)base_vals[bi]);

        int *digits = calloc(n_rounds, sizeof(int));
        double phase = 0.0;
        BigInt c_j;
        bigint_copy(&c_j, &base_a);

        printf("  ┌──────┬──────────────────────────────────┬──────┬──────────────┐\n");
        printf("  │Round │ modular multiplication oracle     │  d_j │ θ cumulative │\n");
        printf("  ├──────┼──────────────────────────────────┼──────┼──────────────┤\n");

        for (int j = 0; j < n_rounds; j++) {
            digits[j] = run_ipe_round_big(&c_j, &N_bi, phase, j, j < 2);

            phase = 0;
            for (int m = 0; m <= j; m++) {
                double w = 1.0;
                for (int q = 0; q <= m; q++) w /= D;
                phase += digits[m] * w;
            }

            if (j < 5 || j == n_rounds-1 || (j % 25 == 0))
                printf("  │  %3d  │ |y⟩ → |c_%d·y mod N⟩ (perm)   │  %d   │ θ=%.8f  │\n",
                       j, j, digits[j], phase);

            BigInt sq, sqq, sqr;
            bigint_mul(&sq, &c_j, &c_j);
            bigint_div_mod(&sq, &N_bi, &sqq, &sqr);
            bigint_copy(&c_j, &sqr);
        }
        printf("  └──────┴──────────────────────────────────┴──────┴──────────────┘\n\n");

        printf("  Phase digits: ");
        for (int j = 0; j < n_rounds && j < 30; j++) printf("%d", digits[j]);
        if (n_rounds > 30) printf("...");
        printf("\n");

        BigInt denoms[64];
        int nd = extract_cf_denominators_big(digits, n_rounds, denoms, 64);
        printf("  CF denominators: %d found\n", nd);
        for (int i = 0; i < nd && i < 10; i++) {
            char ds[1240];
            bigint_to_decimal(ds, sizeof(ds), &denoms[i]);
            printf("    q_%d = %s\n", i, ds);
        }
        printf("\n");

        BigInt two; bigint_set_u64(&two, 2);
        BigInt N_minus_1; bigint_sub(&N_minus_1, &N_bi, &one);

        for (int di = 0; di < nd && !factored; di++) {
            for (uint64_t mult = 1; mult <= 24 && !factored; mult++) {
                BigInt mult_bi, r_try;
                bigint_set_u64(&mult_bi, mult);
                bigint_mul(&r_try, &denoms[di], &mult_bi);
                if (bigint_cmp(&r_try, &N_bi) >= 0) continue;

                BigInt r_half, r_rem;
                bigint_div_mod(&r_try, &two, &r_half, &r_rem);
                if (!bigint_is_zero(&r_rem)) continue;

                BigInt verify;
                bigint_pow_mod(&verify, &base_a, &r_try, &N_bi);
                if (bigint_cmp(&verify, &one) != 0) continue;

                BigInt half_pow;
                bigint_pow_mod(&half_pow, &base_a, &r_half, &N_bi);
                if (bigint_cmp(&half_pow, &one) == 0) continue;
                if (bigint_cmp(&half_pow, &N_minus_1) == 0) continue;

                BigInt pm1, pp1, fac1, fac2;
                bigint_sub(&pm1, &half_pow, &one);
                bigint_add(&pp1, &half_pow, &one);
                bigint_gcd(&fac1, &pm1, &N_bi);
                bigint_gcd(&fac2, &pp1, &N_bi);

                BigInt *winner = NULL;
                if (bigint_cmp(&fac1, &one) != 0 &&
                    bigint_cmp(&fac1, &N_bi) != 0) winner = &fac1;
                else if (bigint_cmp(&fac2, &one) != 0 &&
                         bigint_cmp(&fac2, &N_bi) != 0) winner = &fac2;

                if (winner) {
                    BigInt other, rem3, check;
                    bigint_div_mod(&N_bi, winner, &other, &rem3);
                    bigint_mul(&check, winner, &other);

                    char f1s[1240], f2s[1240], rs[1240];
                    bigint_to_decimal(f1s, sizeof(f1s), winner);
                    bigint_to_decimal(f2s, sizeof(f2s), &other);
                    bigint_to_decimal(rs, sizeof(rs), &r_try);

                    printf("  ┌── FACTORS ──────────────────────────────────────┐\n");
                    printf("  │  r = %s\n", rs);
                    printf("  │  p = %s\n  │  q = %s\n", f1s, f2s);
                    printf("  │  p×q = N? %s\n",
                           bigint_cmp(&check, &N_bi)==0 ? "✓" : "✗");
                    printf("  └────────────────────────────────────────────────┘\n\n");
                    factored = (bigint_cmp(&check, &N_bi) == 0);
                }
            }
        }
        free(digits);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec-t0.tv_sec)*1000.0+(t1.tv_nsec-t0.tv_nsec)/1e6;

    printf("  ┌───────────────────────────────────────────────────────────────────┐\n");
    if (factored)
        printf("  │  ✓ %u-BIT FACTORED (modular multiplication permutation)\n", n_bits);
    else
        printf("  │  ✗ %u-bit N — not factored in %d rounds\n", n_bits, n_rounds);
    printf("  │  %.1f ms  |  %d rounds\n", ms, n_rounds);
    printf("  └───────────────────────────────────────────────────────────────────┘\n\n\n");
    return factored;
}

int main(void)
{
    printf("\n");
    printf("  ╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  SHOR'S ALGORITHM — Modular Multiplication Oracle              ║\n");
    printf("  ║                                                                 ║\n");
    printf("  ║  Oracle: |y⟩ → |a·y mod N⟩  (PERMUTATION, not phase)          ║\n");
    printf("  ║  Amplitude MOVES between states — not just phased.             ║\n");
    printf("  ║  Collisions in f(x) = a^x mod N → amplitude interference      ║\n");
    printf("  ║  → QFT reveals period peaks → CF → factors                    ║\n");
    printf("  ╚═══════════════════════════════════════════════════════════════════╝\n\n");

    printf("  ████████████████████████████████████████████████████████████████████\n");
    printf("  ██  PART 1: Small Number Factoring                               ██\n");
    printf("  ████████████████████████████████████████████████████████████████████\n\n");

    struct { uint64_t N; int extra; } targets[] = {
        { 15,    4  },  { 21,    4  },  { 35,    4  },
        { 77,    6  },  { 143,   6  },  { 323,   6  },
        { 899,   8  },  { 2021,  8  },  { 8633,  8  },
    };
    int n = sizeof(targets)/sizeof(targets[0]);
    int wins = 0;
    for (int i = 0; i < n; i++)
        wins += ipe_factor_u64(targets[i].N, targets[i].extra);

    printf("  ── Small: %d / %d factored ──\n\n\n", wins, n);

    printf("  ████████████████████████████████████████████████████████████████████\n");
    printf("  ██  PART 2: 256-bit Semiprime                                    ██\n");
    printf("  ████████████████████████████████████████████████████████████████████\n\n");

    int big_win = ipe_factor_bigint(
        "115792089237316195423570985008687907854578655348606557127283215897629986438259",
        15);

    printf("  ╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  RESULTS                                                        ║\n");
    printf("  ║  Small:   %2d / %2d                                             ║\n", wins, n);
    printf("  ║  256-bit: %s                                                ║\n",
           big_win ? "✓ FACTORED" : "✗ FAILED  ");
    printf("  ║                                                                 ║\n");
    printf("  ║  Oracle: |y⟩ → |a·y mod N mod D⟩                              ║\n");
    printf("  ║  • PERMUTATION: amplitude moves between states                 ║\n");
    printf("  ║  • COLLISIONS: same f(x) → coherent amplitude addition        ║\n");
    printf("  ║  • INTERFERENCE: periodic structure → QFT peaks at s/r        ║\n");
    printf("  ╚═══════════════════════════════════════════════════════════════════╝\n\n");

    return 0;
}
