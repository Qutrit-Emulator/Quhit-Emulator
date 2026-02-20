/*
 * willow_lazy.c — Willow Benchmark with Lazy Evaluation
 *
 * Side-by-side comparison: EAGER (original) vs LAZY (deferred gates).
 * Shows how many gates reality could skip if it computes on demand.
 *
 * Build:
 *   gcc -O2 -std=gnu99 willow_lazy.c quhit_core.c quhit_gates.c \
 *       quhit_measure.c quhit_entangle.c quhit_register.c mps_overlay.c \
 *       bigint.c -lm -o willow_lazy
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
 * EAGER BENCHMARK (original — all gates applied immediately)
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void run_eager(int n, int depth, double *out_time, int *out_1q, int *out_2q)
{
    QuhitEngine *eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(eng);

    uint32_t *q = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    for (int i = 0; i < n; i++) q[i] = quhit_init(eng);

    mps_overlay_init(eng, q, n);
    mps_overlay_write_zero(eng, q, n);

    double dft_re[36], dft_im[36], had_re[36], had_im[36];
    double cz_re[36*36], cz_im[36*36];
    mps_build_dft6(dft_re, dft_im);
    mps_build_hadamard2(had_re, had_im);
    mps_build_cz(cz_re, cz_im);
    double *g1_re[2] = {dft_re, had_re};
    double *g1_im[2] = {dft_im, had_im};

    int *last = (int *)calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) last[i] = -1;

    int total_1q = 0, total_2q = 0;
    double t0 = wall_time();

    for (int d = 0; d < depth; d++) {
        for (int i = 0; i < n; i++) {
            int gi;
            do { gi = (int)(quhit_prng(eng) % 2); } while (gi == last[i]);
            last[i] = gi;
            mps_gate_1site(eng, q, n, i, g1_re[gi], g1_im[gi]);
            total_1q++;
        }
        mps_sweep_right = 1;
        for (int i = 0; i < n - 1; i++) {
            mps_gate_2site(eng, q, n, i, cz_re, cz_im);
            total_2q++;
        }
        mps_renormalize_chain(eng, q, n);
    }

    *out_time = wall_time() - t0;
    *out_1q = total_1q;
    *out_2q = total_2q;

    free(last);
    free(q);
    mps_overlay_free();
    quhit_engine_destroy(eng);
    free(eng);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * LAZY BENCHMARK (deferred gates, measurement triggers materialization)
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void run_lazy(int n, int depth, int measure_sites,
                     double *out_queue_time, double *out_meas_time,
                     LazyStats *out_stats)
{
    QuhitEngine *eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(eng);

    uint32_t *q = (uint32_t *)malloc((size_t)n * sizeof(uint32_t));
    for (int i = 0; i < n; i++) q[i] = quhit_init(eng);

    MpsLazyChain *lc = mps_lazy_init(eng, q, n);

    /* Initialize all sites to |0⟩ */
    for (int i = 0; i < n; i++)
        mps_lazy_zero_site(lc, i);

    double dft_re[36], dft_im[36], had_re[36], had_im[36];
    double cz_re[36*36], cz_im[36*36];
    mps_build_dft6(dft_re, dft_im);
    mps_build_hadamard2(had_re, had_im);
    mps_build_cz(cz_re, cz_im);
    double *g1_re[2] = {dft_re, had_re};
    double *g1_im[2] = {dft_im, had_im};

    int *last = (int *)calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) last[i] = -1;

    double t0 = wall_time();

    /* Queue ALL gates — nothing is applied */
    for (int d = 0; d < depth; d++) {
        for (int i = 0; i < n; i++) {
            int gi;
            do { gi = (int)(quhit_prng(eng) % 2); } while (gi == last[i]);
            last[i] = gi;
            mps_lazy_gate_1site(lc, i, g1_re[gi], g1_im[gi]);
        }
        mps_sweep_right = 1;
        for (int i = 0; i < n - 1; i++)
            mps_lazy_gate_2site(lc, i, cz_re, cz_im);
    }

    *out_queue_time = wall_time() - t0;

    /* Now measure — THIS triggers materialization */
    double t1 = wall_time();
    mps_lazy_flush(lc);  /* flush all to measure */

    /* Renormalize after flush */
    mps_renormalize_chain(eng, q, n);

    /* Measure a subset of sites */
    int nbytes = n * (int)sizeof(MpsTensor);
    MpsTensor *saved = (MpsTensor *)malloc(nbytes);
    memcpy(saved, mps_store, nbytes);

    for (int s = 0; s < measure_sites && s < n; s++) {
        memcpy(mps_store, saved, nbytes);
        mps_overlay_measure(eng, q, n, s);
    }

    *out_meas_time = wall_time() - t1;
    free(saved);

    mps_lazy_finalize_stats(lc);
    *out_stats = lc->stats;

    free(last);
    free(q);
    mps_lazy_free(lc);
    quhit_engine_destroy(eng);
    free(eng);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * COMPARISON RUNNER
 * ═══════════════════════════════════════════════════════════════════════════════ */
static void run_comparison(int n, int depth, int meas_sites)
{
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  WILLOW RCS: %d quhits, depth %d    EAGER vs LAZY\n", n, depth);
    printf("║  D=%d, χ=%d, Hilbert=6^%d ≈ 10^%.0f\n",
           MPS_PHYS, MPS_CHI, n, n * log10(6.0));
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    /* Eager */
    double eager_time;
    int eager_1q, eager_2q;
    printf("  EAGER (all gates applied immediately)...\n");
    run_eager(n, depth, &eager_time, &eager_1q, &eager_2q);
    printf("  ✓ Eager: %.3fs  1q=%d  2q=%d  total=%d\n\n",
           eager_time, eager_1q, eager_2q, eager_1q + eager_2q);

    /* Lazy */
    double lazy_q, lazy_m;
    LazyStats ls;
    printf("  LAZY (gates queued, materialized on measure)...\n");
    run_lazy(n, depth, meas_sites, &lazy_q, &lazy_m, &ls);
    printf("  ✓ Lazy:  queue=%.3fs  materialize=%.3fs  total=%.3fs\n\n",
           lazy_q, lazy_m, lazy_q + lazy_m);

    /* Stats */
    lazy_stats_print(&ls);

    /* Comparison table */
    printf("\n  ┌─────────────────────────┬────────────┬────────────┐\n");
    printf("  │ Metric                  │    Eager   │    Lazy    │\n");
    printf("  ├─────────────────────────┼────────────┼────────────┤\n");
    printf("  │ Total time              │ %8.3f s │ %8.3f s │\n",
           eager_time, lazy_q + lazy_m);
    printf("  │ Gate queue time         │     N/A    │ %8.3f s │\n", lazy_q);
    printf("  │ Materialization time    │     N/A    │ %8.3f s │\n", lazy_m);
    printf("  │ Gates queued            │   %7d  │   %7lu  │\n",
           eager_1q + eager_2q, (unsigned long)ls.gates_queued);
    printf("  │ Gates materialized      │   %7d  │   %7lu  │\n",
           eager_1q + eager_2q, (unsigned long)ls.gates_materialized);
    printf("  │ Gates fused             │         0  │   %7lu  │\n",
           (unsigned long)ls.gates_fused);
    printf("  │ MPS memory              │ %6.0f KB  │ %6lu KB  │\n",
           n * sizeof(MpsTensor) / 1024.0,
           (unsigned long)(ls.memory_actual / 1024));
    printf("  │ Hilbert space           │  10^%-5.0f  │  10^%-5.0f  │\n",
           n * log10(6.0), ls.hilbert_log10);
    printf("  └─────────────────────────┴────────────┴────────────┘\n");
}

int main(void)
{
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("  HEXSTATE V2 — WILLOW BENCHMARK: EAGER vs LAZY EVALUATION\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    printf("  Eager: all gates applied immediately (original behavior)\n");
    printf("  Lazy:  gates queued, materialized only when measured\n");
    printf("══════════════════════════════════════════════════════════════════\n");

    /* Warm-up + small scale */
    run_comparison(10,  8,  5);

    /* Willow-equivalent */
    run_comparison(53,  20, 10);
    run_comparison(105, 20, 5);

    /* Beyond Willow */
    run_comparison(500,  20, 3);
    run_comparison(1000, 20, 1);

    printf("\n══════════════════════════════════════════════════════════════════\n");
    printf("  BENCHMARK COMPLETE\n");
    printf("══════════════════════════════════════════════════════════════════\n");
    return 0;
}
