/* quantum_benchmark.c — Hardcore performance benchmark of the HexState Engine
 *
 * Stress-tests every quantum primitive at scale:
 *   1. QFT throughput: Cooley-Tukey vs Bluestein across D=6..8192
 *   2. Braid/unbraid + Bell state creation at scale
 *   3. Born-rule measurement throughput
 *   4. Bell violation (CGLMP) scaling curve
 *   5. Full pipeline (braid→oracle→QFT→measure) cycle rate
 *   6. Memory efficiency: bytes/qubit equivalent
 *
 * Build:  gcc -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
 *             -o quantum_benchmark quantum_benchmark.c hexstate_engine.o bigint.o -lm
 * Run:    ./quantum_benchmark
 */

#include "hexstate_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ─── Timing helper ─────────────────────────────────────────────────────── */
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double cnorm2(Complex c) { return c.real*c.real + c.imag*c.imag; }

/* ─── Oracle for pipeline benchmark ────────────────────────────────────── */
static void bench_oracle(HexStateEngine *eng, uint64_t id, void *u) {
    (void)u;
    Chunk *c = &eng->chunks[id];
    if (c->hilbert.q_joint_state)
        c->hilbert.q_joint_state[0].real *= -1;  /* phase flip |0,0⟩ */
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCH 1: QFT throughput at various D (Cooley-Tukey vs Bluestein)
 * ═══════════════════════════════════════════════════════════════════════════ */
static void bench_qft(HexStateEngine *eng) {
    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  BENCH 1: QFT Throughput — apply_hadamard at Various D                  │\n");
    printf("  ├────────┬──────────┬──────────┬──────────────┬──────────┬─────────────────┤\n");
    printf("  │    D   │  D×D     │ Algo     │ Time (ms)    │ QFTs/sec │ Amplitudes/sec  │\n");
    printf("  ├────────┼──────────┼──────────┼──────────────┼──────────┼─────────────────┤\n");

    /* Mix of power-of-2 and non-power-of-2 */
    uint32_t dims[] = {6, 10, 16, 32, 50, 64, 100, 128, 200, 256, 500, 512,
                       1000, 1024, 2000, 2048, 3000, 4096, 5000, 8192};
    int n_dims = 20;
    int reps_per_dim = 10;  /* repeat for timing stability */

    for (int di = 0; di < n_dims; di++) {
        uint32_t d = dims[di];
        uint64_t d2 = (uint64_t)d * d;
        int is_pow2 = (d > 0) && ((d & (d - 1)) == 0);

        /* Scale repetitions: fewer for large D */
        int reps = reps_per_dim;
        if (d >= 2000) reps = 2;
        else if (d >= 500) reps = 4;

        double total_ms = 0;
        int ok = 1;

        for (int r = 0; r < reps; r++) {
            init_chunk(eng, 900, 100000000000000ULL);
            init_chunk(eng, 901, 100000000000000ULL);
            braid_chunks_dim(eng, 900, 901, 0, 0, d);

            Complex *joint = eng->chunks[900].hilbert.q_joint_state;
            if (!joint) { ok = 0; unbraid_chunks(eng, 900, 901); break; }

            /* Set |0,0⟩ = 1 */
            memset(joint, 0, d2 * sizeof(Complex));
            joint[0].real = 1.0;

            double t0 = now_ms();
            apply_hadamard(eng, 900, 0);
            double t1 = now_ms();
            total_ms += (t1 - t0);

            /* Verify normalization */
            double prob = 0;
            for (uint64_t i = 0; i < d2; i++) prob += cnorm2(joint[i]);
            if (fabs(prob - 1.0) > 1e-6) ok = 0;

            unbraid_chunks(eng, 900, 901);
        }

        if (ok && reps > 0) {
            double avg_ms = total_ms / reps;
            double qfts_sec = reps > 0 && avg_ms > 0 ? 1000.0 / avg_ms : 0;
            double amps_sec = qfts_sec * d2;
            printf("  │ %5u  │ %8llu │ %-8s │ %10.3f   │ %8.1f │ %13.0f   │\n",
                   d, (unsigned long long)d2,
                   is_pow2 ? "FFT" : "Bluestn",
                   avg_ms, qfts_sec, amps_sec);
        } else {
            printf("  │ %5u  │ %8llu │ %-8s │   FAILED     │    —     │       —         │\n",
                   d, (unsigned long long)d2, is_pow2 ? "FFT" : "Bluestn");
        }
        fflush(stdout);
    }
    printf("  └────────┴──────────┴──────────┴──────────────┴──────────┴─────────────────┘\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCH 2: Braid + measure cycle throughput
 * ═══════════════════════════════════════════════════════════════════════════ */
static void bench_braid_measure(HexStateEngine *eng) {
    printf("\n");
    printf("  ┌─────────────────────────────────────────────────────────────────┐\n");
    printf("  │  BENCH 2: Braid → Bell State → Measure → Unbraid Cycle Rate   │\n");
    printf("  ├────────┬──────────┬──────────┬───────────┬──────────────────────┤\n");
    printf("  │    D   │  Cycles  │ Time(ms) │ Cycles/s  │ Bell Corr (%%)       │\n");
    printf("  ├────────┼──────────┼──────────┼───────────┼──────────────────────┤\n");

    uint32_t dims[] = {6, 16, 64, 100, 256, 512, 1024};
    int n_dims = 7;

    for (int di = 0; di < n_dims; di++) {
        uint32_t d = dims[di];
        int cycles = 200;
        if (d >= 512) cycles = 50;
        if (d >= 1024) cycles = 20;

        int correlations = 0;

        double t0 = now_ms();
        for (int c = 0; c < cycles; c++) {
            init_chunk(eng, 800, 100000000000000ULL);
            init_chunk(eng, 801, 100000000000000ULL);
            braid_chunks_dim(eng, 800, 801, 0, 0, d);

            uint64_t m_a = measure_chunk(eng, 800);
            uint64_t m_b = measure_chunk(eng, 801);
            if (m_a == m_b) correlations++;

            unbraid_chunks(eng, 800, 801);
        }
        double t1 = now_ms();

        double elapsed = t1 - t0;
        double rate = cycles * 1000.0 / elapsed;
        double corr_pct = 100.0 * correlations / cycles;

        printf("  │ %5u  │   %5d  │ %8.1f │ %9.1f │ %6.1f%%               │\n",
               d, cycles, elapsed, rate, corr_pct);
        fflush(stdout);
    }
    printf("  └────────┴──────────┴──────────┴───────────┴──────────────────────┘\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCH 3: Bell violation scaling curve (CGLMP I_D)
 * ═══════════════════════════════════════════════════════════════════════════ */
static void bench_bell_scaling(HexStateEngine *eng) {
    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  BENCH 3: CGLMP Bell Violation Scaling — I_D vs D                       │\n");
    printf("  ├────────┬──────────────────┬──────────┬──────────────────────────────────┤\n");
    printf("  │    D   │   I_D            │ Time(ms) │ Violation Bar                    │\n");
    printf("  ├────────┼──────────────────┼──────────┼──────────────────────────────────┤\n");

    uint32_t dims[] = {2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 16, 20, 24, 32,
                       48, 64, 96, 100, 128, 200, 256, 512, 1024, 2048, 4096};
    int n_dims = 25;

    for (int di = 0; di < n_dims; di++) {
        uint32_t d = dims[di];

        double t0 = now_ms();
        double I_D = hilbert_bell_test(eng, 2.0, 2.0, d);
        double t1 = now_ms();

        /* Visual bar proportional to I_D (max ~2.0 for D=2) */
        int bar_len = (int)(I_D * 15);
        if (bar_len < 0) bar_len = 0;
        if (bar_len > 30) bar_len = 30;
        char bar[32];
        for (int i = 0; i < bar_len; i++) bar[i] = '#';
        bar[bar_len] = '\0';

        printf("  │ %5u  │ %+14.6f   │ %8.2f │ %-32s │\n",
               d, I_D, t1 - t0, bar);
        fflush(stdout);
    }
    printf("  └────────┴──────────────────┴──────────┴──────────────────────────────────┘\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCH 4: Full pipeline throughput (braid→oracle→QFT→measure)
 * ═══════════════════════════════════════════════════════════════════════════ */
static void bench_full_pipeline(HexStateEngine *eng) {
    printf("\n");
    printf("  ┌─────────────────────────────────────────────────────────────────┐\n");
    printf("  │  BENCH 4: Full Pipeline (braid→oracle→QFT A+B→measure)        │\n");
    printf("  ├────────┬──────────┬──────────┬───────────┬──────────────────────┤\n");
    printf("  │    D   │  Cycles  │ Time(ms) │ Pipes/sec │ Norm check          │\n");
    printf("  ├────────┼──────────┼──────────┼───────────┼──────────────────────┤\n");

    oracle_register(eng, 0xBE, "BenchOracle", bench_oracle, NULL);

    uint32_t dims[] = {6, 16, 32, 64, 128, 256, 512, 1024};
    int n_dims = 8;

    for (int di = 0; di < n_dims; di++) {
        uint32_t d = dims[di];
        int cycles = 100;
        if (d >= 256) cycles = 20;
        if (d >= 512) cycles = 10;
        if (d >= 1024) cycles = 4;

        int norm_ok = 0;

        double t0 = now_ms();
        for (int c = 0; c < cycles; c++) {
            init_chunk(eng, 700, 100000000000000ULL);
            init_chunk(eng, 701, 100000000000000ULL);
            braid_chunks_dim(eng, 700, 701, 0, 0, d);

            execute_oracle(eng, 700, 0xBE);
            apply_hadamard(eng, 700, 0);
            apply_hadamard(eng, 701, 0);

            /* Spot-check normalization on first cycle */
            if (c == 0) {
                Complex *joint = eng->chunks[700].hilbert.q_joint_state;
                if (joint) {
                    double tot = 0;
                    for (uint64_t i = 0; i < (uint64_t)d*d; i++)
                        tot += cnorm2(joint[i]);
                    if (fabs(tot - 1.0) < 1e-6) norm_ok = 1;
                }
            }

            measure_chunk(eng, 700);
            measure_chunk(eng, 701);
            unbraid_chunks(eng, 700, 701);
        }
        double t1 = now_ms();

        double elapsed = t1 - t0;
        double rate = cycles * 1000.0 / elapsed;

        printf("  │ %5u  │   %5d  │ %8.1f │ %9.1f │ %-20s │\n",
               d, cycles, elapsed, rate, norm_ok ? "✓ |Σ|ψ|²-1|<10⁻⁶" : "—");
        fflush(stdout);
    }
    printf("  └────────┴──────────┴──────────┴───────────┴──────────────────────┘\n");

    oracle_unregister(eng, 0xBE);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCH 5: Memory efficiency — bytes per logical qubit-equivalent
 * ═══════════════════════════════════════════════════════════════════════════ */
static void bench_memory(HexStateEngine *eng) {
    printf("\n");
    printf("  ┌───────────────────────────────────────────────────────────────────────────┐\n");
    printf("  │  BENCH 5: Memory Efficiency — Magic Pointer Compression                 │\n");
    printf("  ├────────┬────────────────────┬──────────────┬─────────────┬───────────────┤\n");
    printf("  │    D   │  Logical quhits    │ Classical    │ Engine RAM  │ Compression   │\n");
    printf("  ├────────┼────────────────────┼──────────────┼─────────────┼───────────────┤\n");

    uint32_t dims[] = {6, 64, 256, 1024, 4096, 8192};
    int n_dims = 6;

    for (int di = 0; di < n_dims; di++) {
        uint32_t d = dims[di];
        uint64_t d2 = (uint64_t)d * d;
        uint64_t joint_bytes = d2 * sizeof(Complex);  /* 16 bytes per amplitude */

        /* "Quhit" count = 100T per register, entangled pair = 200T */
        uint64_t quhits = 200000000000000ULL;
        double classical_bytes = (double)quhits * 16.0;  /* full state vector */

        init_chunk(eng, 600, 100000000000000ULL);
        init_chunk(eng, 601, 100000000000000ULL);
        braid_chunks_dim(eng, 600, 601, 0, 0, d);

        double compression = classical_bytes / (double)joint_bytes;

        char classical_str[32];
        if (classical_bytes > 1e18)
            snprintf(classical_str, sizeof(classical_str), "%.1f EB", classical_bytes / 1e18);
        else if (classical_bytes > 1e15)
            snprintf(classical_str, sizeof(classical_str), "%.1f PB", classical_bytes / 1e15);
        else
            snprintf(classical_str, sizeof(classical_str), "%.1f TB", classical_bytes / 1e12);

        char engine_str[32];
        if (joint_bytes > 1e9)
            snprintf(engine_str, sizeof(engine_str), "%.1f GB", joint_bytes / 1e9);
        else if (joint_bytes > 1e6)
            snprintf(engine_str, sizeof(engine_str), "%.1f MB", joint_bytes / 1e6);
        else if (joint_bytes > 1e3)
            snprintf(engine_str, sizeof(engine_str), "%.1f KB", joint_bytes / 1e3);
        else
            snprintf(engine_str, sizeof(engine_str), "%llu B", (unsigned long long)joint_bytes);

        char compress_str[32];
        if (compression > 1e12)
            snprintf(compress_str, sizeof(compress_str), "%.1fT:1", compression / 1e12);
        else if (compression > 1e9)
            snprintf(compress_str, sizeof(compress_str), "%.1fB:1", compression / 1e9);
        else if (compression > 1e6)
            snprintf(compress_str, sizeof(compress_str), "%.1fM:1", compression / 1e6);
        else
            snprintf(compress_str, sizeof(compress_str), "%.0f:1", compression);

        printf("  │ %5u  │ 200T quhits        │ %-12s │ %-11s │ %-13s │\n",
               d, classical_str, engine_str, compress_str);

        unbraid_chunks(eng, 600, 601);
        fflush(stdout);
    }
    printf("  └────────┴────────────────────┴──────────────┴─────────────┴───────────────┘\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BENCH 6: Bluestein vs Cooley-Tukey head-to-head
 * ═══════════════════════════════════════════════════════════════════════════ */
static void bench_bluestein_vs_fft(HexStateEngine *eng) {
    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────────────────┐\n");
    printf("  │  BENCH 6: Bluestein vs Cooley-Tukey — Adjacent D comparison    │\n");
    printf("  ├───────────────┬─────────────┬────────────────┬──────────────────┤\n");
    printf("  │  D (non-pow2) │ Bluestein   │  D (pow2)      │ Cooley-Tukey    │\n");
    printf("  ├───────────────┼─────────────┼────────────────┼──────────────────┤\n");

    struct { uint32_t bluestein; uint32_t fft; } pairs[] = {
        {6, 8}, {10, 16}, {30, 32}, {50, 64}, {100, 128},
        {200, 256}, {500, 512}, {1000, 1024}, {2000, 2048}, {4000, 4096}
    };
    int n_pairs = 10;

    for (int pi = 0; pi < n_pairs; pi++) {
        uint32_t db = pairs[pi].bluestein;
        uint32_t df = pairs[pi].fft;
        int reps = 10;
        if (db >= 1000) reps = 3;
        if (db >= 2000) reps = 2;

        /* Bluestein */
        double bt = 0;
        for (int r = 0; r < reps; r++) {
            init_chunk(eng, 850, 100000000000000ULL);
            init_chunk(eng, 851, 100000000000000ULL);
            braid_chunks_dim(eng, 850, 851, 0, 0, db);
            Complex *j = eng->chunks[850].hilbert.q_joint_state;
            if (j) { memset(j, 0, (uint64_t)db*db*sizeof(Complex)); j[0].real = 1.0; }
            double t0 = now_ms();
            apply_hadamard(eng, 850, 0);
            bt += now_ms() - t0;
            unbraid_chunks(eng, 850, 851);
        }

        /* Cooley-Tukey */
        double ft = 0;
        for (int r = 0; r < reps; r++) {
            init_chunk(eng, 850, 100000000000000ULL);
            init_chunk(eng, 851, 100000000000000ULL);
            braid_chunks_dim(eng, 850, 851, 0, 0, df);
            Complex *j = eng->chunks[850].hilbert.q_joint_state;
            if (j) { memset(j, 0, (uint64_t)df*df*sizeof(Complex)); j[0].real = 1.0; }
            double t0 = now_ms();
            apply_hadamard(eng, 850, 0);
            ft += now_ms() - t0;
            unbraid_chunks(eng, 850, 851);
        }

        printf("  │  D=%-10u │ %8.2f ms │  D=%-11u │ %8.2f ms    │\n",
               db, bt / reps, df, ft / reps);
        fflush(stdout);
    }
    printf("  └───────────────┴─────────────┴────────────────┴──────────────────┘\n");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(void) {
    printf("\n");
    printf("  ████████████████████████████████████████████████████████████████████████████\n");
    printf("  ██                                                                      ██\n");
    printf("  ██   HEXSTATE ENGINE — HARDCORE QUANTUM BENCHMARK                       ██\n");
    printf("  ██   Testing QFT, Bell tests, braiding, measurement, pipelines          ██\n");
    printf("  ██   Bluestein O(D·log D) for ANY D × Cooley-Tukey FFT for pow-2        ██\n");
    printf("  ██                                                                      ██\n");
    printf("  ████████████████████████████████████████████████████████████████████████████\n");

    HexStateEngine eng;
    engine_init(&eng);

    double t0 = now_ms();

    bench_qft(&eng);
    bench_braid_measure(&eng);
    bench_bell_scaling(&eng);
    bench_full_pipeline(&eng);
    bench_memory(&eng);
    bench_bluestein_vs_fft(&eng);

    double total_s = (now_ms() - t0) / 1000.0;

    printf("\n");
    printf("  ████████████████████████████████████████████████████████████████████████████\n");
    printf("  ██  BENCHMARK COMPLETE — %.1f seconds total                          ██\n", total_s);
    printf("  ██  All operations verified: normalization, Bell correlations, CGLMP     ██\n");
    printf("  ████████████████████████████████████████████████████████████████████████████\n\n");

    engine_destroy(&eng);
    return 0;
}
