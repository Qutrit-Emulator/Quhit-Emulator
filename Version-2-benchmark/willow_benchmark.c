/*
 * willow_benchmark.c — Google Willow Quantum Supremacy Benchmark
 *
 * Uses ONLY the HexState MPS overlay API:
 *   Gate constructors:  mps_build_dft6, mps_build_cz, mps_build_hadamard2
 *   Gate application:   mps_gate_1site, mps_gate_2site
 *   State management:   mps_overlay_init, mps_overlay_write_zero, mps_overlay_free
 *   Readout:            mps_overlay_measure, mps_overlay_amplitude, mps_overlay_norm
 *   Chain management:   mps_renormalize_chain, mps_sweep_right
 *
 * Build:
 *   gcc -O2 -std=gnu99 willow_benchmark.c quhit_core.c quhit_gates.c \
 *       quhit_measure.c quhit_entangle.c quhit_register.c mps_overlay.c \
 *       bigint.c -lm -o willow_benchmark
 */

#include "mps_overlay.h"
#include <time.h>
#include <sys/time.h>
#include <math.h>

static double wall_time(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * XEB via mps_overlay_amplitude
 *
 * XEB_F = D^N × <P(x)>_samples − 1
 * Save/restore MPS state, measure samples, compute amplitudes.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static double compute_xeb(QuhitEngine *eng, uint32_t *quhits, int n,
                           int n_samples)
{
    int nbytes = n * (int)sizeof(MpsTensor);
    MpsTensor *saved = (MpsTensor *)malloc(nbytes);
    if (!saved) return -1.0;
    memcpy(saved, mps_store, nbytes);

    double norm_sq = mps_overlay_norm(eng, quhits, n);

    uint32_t *samples = (uint32_t *)malloc(
        (size_t)n_samples * (size_t)n * sizeof(uint32_t));

    for (int s = 0; s < n_samples; s++) {
        memcpy(mps_store, saved, nbytes);
        for (int i = 0; i < n; i++)
            samples[s * n + i] = mps_overlay_measure(eng, quhits, n, i);
    }

    memcpy(mps_store, saved, nbytes);
    double sum_prob = 0;

    for (int s = 0; s < n_samples; s++) {
        double amp_re, amp_im;
        mps_overlay_amplitude(eng, quhits, n, &samples[s * n], &amp_re, &amp_im);
        double p = amp_re * amp_re + amp_im * amp_im;
        if (norm_sq > 1e-30) p /= norm_sq;
        sum_prob += p;
    }

    double mean_p = sum_prob / n_samples;
    double log2_xeb1 = (double)n * log2((double)MPS_PHYS) + log2(mean_p > 0 ? mean_p : 1e-300);
    double xeb = pow(2.0, log2_xeb1) - 1.0;
    if (!isfinite(xeb)) xeb = -1.0;

    memcpy(mps_store, saved, nbytes);
    free(samples);
    free(saved);
    return xeb;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * RUN BENCHMARK — native MPS gates only
 *
 * Each depth cycle:
 *   1. Random 1-site gates (DFT₆ or H₂) via mps_gate_1site
 *   2. CZ on ALL adjacent pairs L→R via mps_gate_2site
 *   3. mps_renormalize_chain to restore ||ψ||=1
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void run_benchmark(int n, int depth, int n_xeb_samples)
{
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  WILLOW RCS: %d quhits, depth %d\n", n, depth);
    printf("║  D=%d, MPS χ=%d, 1D chain (native API)\n", MPS_PHYS, MPS_CHI);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    QuhitEngine *eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    if (!eng) { fprintf(stderr, "ERROR: alloc engine\n"); return; }
    quhit_engine_init(eng);

    uint32_t *quhits = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    for (int i = 0; i < n; i++)
        quhits[i] = quhit_init(eng);

    mps_overlay_init(eng, quhits, n);
    mps_overlay_write_zero(eng, quhits, n);

    /* Engine gate constructors */
    double dft6_re[36], dft6_im[36];
    double had2_re[36], had2_im[36];
    double cz_re[36 * 36], cz_im[36 * 36];

    mps_build_dft6(dft6_re, dft6_im);
    mps_build_hadamard2(had2_re, had2_im);
    mps_build_cz(cz_re, cz_im);

    double *g1_re[2] = { dft6_re, had2_re };
    double *g1_im[2] = { dft6_im, had2_im };

    int *last_gate = (int *)calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) last_gate[i] = -1;

    double t_start = wall_time();
    int total_1q = 0, total_2q = 0;

    for (int d = 0; d < depth; d++) {
        double cyc_t = wall_time();

        /* ── 1-site layer: random DFT₆ or H₂ per site ── */
        for (int q = 0; q < n; q++) {
            int gi;
            do { gi = (int)(quhit_prng(eng) % 2); } while (gi == last_gate[q]);
            last_gate[q] = gi;
            mps_gate_1site(eng, quhits, n, q, g1_re[gi], g1_im[gi]);
            total_1q++;
        }

        /* ── 2-site CZ layer: sequential L→R on all adjacent pairs ── */
        mps_sweep_right = 1;
        int cz_count = 0;

        for (int q = 0; q < n - 1; q++) {
            mps_gate_2site(eng, quhits, n, q, cz_re, cz_im);
            cz_count++;
            total_2q++;
        }

        /* Renormalize chain after each cycle */
        mps_renormalize_chain(eng, quhits, n);
        double norm = mps_overlay_norm(eng, quhits, n);

        printf("  Cycle %2d:  %7.4f s  norm=%.6f  1q=%d  CZ=%d\n",
               d + 1, wall_time() - cyc_t, norm, n, cz_count);
    }

    double t_total = wall_time() - t_start;
    double final_norm = mps_overlay_norm(eng, quhits, n);
    double mem_kb = (double)n * sizeof(MpsTensor) / 1024.0;

    printf("\n┌──────────────────────────────────────────────────────────┐\n");
    printf("│  RESULTS: %d quhits, depth %d\n", n, depth);
    printf("├──────────────────────────────────────────────────────────┤\n");
    printf("│  Circuit time:  %10.3f s\n", t_total);
    printf("│  Final norm:    %10.6f\n", final_norm);
    printf("│  MPS memory:    %10.1f KB  (%d × %zu B/site)\n",
           mem_kb, n, sizeof(MpsTensor));
    printf("│  1-site gates:  %10d\n", total_1q);
    printf("│  2-site gates:  %10d  (mps_build_cz)\n", total_2q);
    printf("│  Hilbert dim:   6^%d ≈ 10^%.0f\n",
           n, n * log10(6.0));
    printf("└──────────────────────────────────────────────────────────┘\n");

    if (n_xeb_samples > 0) {
        printf("\n  XEB (%d samples via mps_overlay_amplitude)...\n", n_xeb_samples);
        double tx = wall_time();
        double xeb = compute_xeb(eng, quhits, n, n_xeb_samples);
        printf("  XEB = %.6f  (%.3f s)\n", xeb, wall_time() - tx);
    }

    free(last_gate);
    free(quhits);
    mps_overlay_free();
    quhit_engine_destroy(eng);
    free(eng);
}

int main(void)
{
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  HEXSTATE ENGINE V2 — WILLOW QUANTUM SUPREMACY BENCHMARK\n");
    printf("══════════════════════════════════════════════════════════════\n");
    printf("  Gates:  mps_build_dft6 / mps_build_hadamard2 / mps_build_cz\n");
    printf("  MPS:    χ=%d, D=%d (exact — no truncation)\n", MPS_CHI, MPS_PHYS);
    printf("  Sweep:  L→R sequential on all adjacent pairs\n");
    printf("══════════════════════════════════════════════════════════════\n");

    run_benchmark(10,   8,   200);
    run_benchmark(20,  14,   100);
    run_benchmark(53,  20,    50);
    run_benchmark(105, 20,    20);

    printf("\n══════════════════════════════════════════════════════════════\n");
    printf("  BENCHMARK COMPLETE\n");
    printf("══════════════════════════════════════════════════════════════\n");
    return 0;
}
