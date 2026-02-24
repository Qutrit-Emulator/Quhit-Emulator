/*
 * mipt_6d.c — 6D Measurement-Induced Phase Transition (CORRECTED)
 *
 * D=6 in 6 spatial dimensions.
 * Finite-size scaling with PROPER STATISTICAL AVERAGING.
 *
 * Previous version used MIPT_STEPS=10 and a single trajectory,
 * producing shot-noise-dominated data. This version fixes:
 *   1. MIPT_STEPS=30 for proper equilibration
 *   2. N_SAMPLES=8 independent trajectories per (L, p) point
 *   3. Reports mean ± standard error
 *
 * L=2:  64  quhits → 6^64  ≈ 10^50
 * L=3: 729  quhits → 6^729 ≈ 10^567
 */

#include "peps6d_overlay.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MIPT_J       1.5
#define MIPT_DTAU    1.0
#define MIPT_STEPS   30       /* equilibration steps (was 10) */
#define N_SAMPLES    8        /* independent trajectories per point */

/* ═══════════════ DFT₆ ═══════════════ */

static double DFT_RE[36], DFT_IM[36];

static void build_dft6(void)
{
    int D = TNS6D_D;
    double norm = 1.0 / sqrt(D);
    for (int j = 0; j < D; j++)
     for (int k = 0; k < D; k++) {
         double phase = 2.0 * M_PI * j * k / D;
         DFT_RE[j*D+k] = norm * cos(phase);
         DFT_IM[j*D+k] = norm * sin(phase);
     }
}

/* ═══════════════ Clock Gate ═══════════════ */

static void build_clock_gate(double *G_re, double *G_im)
{
    int D = TNS6D_D, D2 = D*D, D4 = D2*D2;
    for (int i = 0; i < D4; i++) { G_re[i] = 0; G_im[i] = 0; }
    for (int kA = 0; kA < D; kA++)
     for (int kB = 0; kB < D; kB++) {
         int idx = (kA*D+kB)*D2 + (kA*D+kB);
         double phase = MIPT_J * cos(2.0*M_PI*(kA-kB)/(double)D) * MIPT_DTAU;
         G_re[idx] = cos(phase);
         G_im[idx] = -sin(phase);
     }
}

/* ═══════════════ Measurement ═══════════════ */

static void measure_site(Tns6dGrid *g, int x, int y, int z, int w, int v, int u)
{
    double probs[6];
    tns6d_local_density(g, x, y, z, w, v, u, probs);

    double r = (double)rand() / RAND_MAX;
    double cumul = 0;
    int outcome = 0;
    for (int k = 0; k < 6; k++) {
        cumul += probs[k];
        if (r <= cumul) { outcome = k; break; }
    }

    double P_re[36]={0}, P_im[36]={0};
    P_re[outcome*6 + outcome] = 1.0;
    tns6d_gate_1site(g, x, y, z, w, v, u, P_re, P_im);
    tns6d_normalize_site(g, x, y, z, w, v, u);
}

/* ═══════════════ Diagnostics ═══════════════ */

static double avg_entropy(Tns6dGrid *g)
{
    double total = 0;
    int N = g->Lx * g->Ly * g->Lz * g->Lw * g->Lv * g->Lu;
    for (int u = 0; u < g->Lu; u++)
     for (int v = 0; v < g->Lv; v++)
      for (int w = 0; w < g->Lw; w++)
       for (int z = 0; z < g->Lz; z++)
        for (int y = 0; y < g->Ly; y++)
         for (int x = 0; x < g->Lx; x++) {
             double probs[6];
             tns6d_local_density(g, x, y, z, w, v, u, probs);
             double S = 0;
             for (int k = 0; k < 6; k++)
                 if (probs[k] > 1e-20) S -= probs[k] * log(probs[k]);
             total += S / log(6.0);
         }
    return total / N;
}

static void compress_reg(QuhitEngine *eng, int reg, double thr)
{
    if (reg < 0) return;
    QuhitRegister *r = &eng->registers[reg];
    uint32_t j = 0;
    for (uint32_t i = 0; i < r->num_nonzero; i++) {
        double m = r->entries[i].amp_re * r->entries[i].amp_re +
                   r->entries[i].amp_im * r->entries[i].amp_im;
        if (m > thr) { if (j != i) r->entries[j] = r->entries[i]; j++; }
    }
    r->num_nonzero = j;
}

static void compress_all(Tns6dGrid *g)
{
    int N = g->Lx * g->Ly * g->Lz * g->Lw * g->Lv * g->Lu;
    for (int i = 0; i < N; i++)
        compress_reg(g->eng, g->site_reg[i], 1e-4);
}

/* ═══════════════ Single MIPT Trajectory ═══════════════ */

static double run_mipt_single(int L, double meas_rate, double *G_re, double *G_im)
{
    Tns6dGrid *g = tns6d_init(L, L, L, L, L, L);
    double final_entropy = 0;

    for (int step = 1; step <= MIPT_STEPS; step++) {
        tns6d_gate_1site_all(g, DFT_RE, DFT_IM);
        tns6d_trotter_step(g, G_re, G_im);
        compress_all(g);

        for (int u = 0; u < L; u++)
         for (int v = 0; v < L; v++)
          for (int w = 0; w < L; w++)
           for (int z = 0; z < L; z++)
            for (int y = 0; y < L; y++)
             for (int x = 0; x < L; x++) {
                 if ((double)rand() / RAND_MAX < meas_rate)
                     measure_site(g, x, y, z, w, v, u);
             }

        final_entropy = avg_entropy(g);
    }

    tns6d_free(g);
    return final_entropy;
}

/* ═══════════════ Averaged MIPT Run ═══════════════ */

static void run_mipt_averaged(int L, double meas_rate, double *G_re, double *G_im,
                               double *mean_out, double *stderr_out)
{
    double samples[N_SAMPLES];

    for (int s = 0; s < N_SAMPLES; s++) {
        samples[s] = run_mipt_single(L, meas_rate, G_re, G_im);
    }

    /* Compute mean */
    double mean = 0;
    for (int s = 0; s < N_SAMPLES; s++) mean += samples[s];
    mean /= N_SAMPLES;

    /* Compute standard error = stddev / sqrt(N) */
    double var = 0;
    for (int s = 0; s < N_SAMPLES; s++) var += (samples[s] - mean) * (samples[s] - mean);
    var /= (N_SAMPLES - 1);
    double se = sqrt(var / N_SAMPLES);

    *mean_out = mean;
    *stderr_out = se;
}

/* ═══════════════ Main: Finite-Size Scaling ═══════════════ */

int main(void)
{
    srand((unsigned)time(NULL));
    int D = TNS6D_D, D2 = D*D, D4 = D2*D2;

    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  6D MIPT — FINITE-SIZE SCALING (STAT. AVERAGED)                ║\n");
    printf("  ║  ────────────────────────────────────────────────────────────── ║\n");
    printf("  ║  D=6 in 6 dimensions. 12 neighbors per site.                   ║\n");
    printf("  ║  %d Trotter steps × %d independent trajectories per point     ║\n",
           MIPT_STEPS, N_SAMPLES);
    printf("  ║                                                                ║\n");
    printf("  ║  L=2:  64  quhits (6^64  ≈ 10^50)                             ║\n");
    printf("  ║  L=3: 729  quhits (6^729 ≈ 10^567)                            ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    build_dft6();

    double *G_re = (double *)calloc(D4, sizeof(double));
    double *G_im = (double *)calloc(D4, sizeof(double));
    build_clock_gate(G_re, G_im);

    double rates[] = {0.00,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.80,1.00};
    int n_rates = sizeof(rates)/sizeof(rates[0]);

    int L_values[] = {2, 3};
    int num_L = 2;

    double mean_S[2][13], se_S[2][13];

    for (int li = 0; li < num_L; li++) {
        int L = L_values[li];
        int N = 1; for(int d=0;d<6;d++) N*=L;
        printf("  Running L=%d (%d quhits, 6^%d ≈ 10^%.0f) — %d steps × %d samples...\n",
               L, N, N, N*log10(6.0), MIPT_STEPS, N_SAMPLES);
        fflush(stdout);

        for (int pi = 0; pi < n_rates; pi++) {
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);

            run_mipt_averaged(L, rates[pi], G_re, G_im,
                              &mean_S[li][pi], &se_S[li][pi]);

            clock_gettime(CLOCK_MONOTONIC, &t1);
            double dt = (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;

            printf("    p=%.2f → ⟨S⟩=%.4f ± %.4f  (%.1fs, %d traj)\n",
                   rates[pi], mean_S[li][pi], se_S[li][pi], dt, N_SAMPLES);
            fflush(stdout);
        }
        printf("\n");
    }

    /* ═══════════════ CROSSING TABLE ═══════════════ */
    printf("  ╔══════════════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  FINITE-SIZE SCALING — 6D MIPT (STAT. AVERAGED)                        ║\n");
    printf("  ╠══════════════════════════════════════════════════════════════════════════╣\n");
    printf("  ║                                                                        ║\n");
    printf("  ║   p    |  ⟨S⟩ L=2 ± SE  |  ⟨S⟩ L=3 ± SE  |  Δ ± σ_Δ  | Signal?     ║\n");
    printf("  ║  ──────┼─────────────────┼─────────────────┼───────────┼───────────── ║\n");

    for (int pi = 0; pi < n_rates; pi++) {
        double delta = mean_S[0][pi] - mean_S[1][pi];
        /* Propagated error: σ_Δ = sqrt(σ²_A + σ²_B) */
        double se_delta = sqrt(se_S[0][pi]*se_S[0][pi] + se_S[1][pi]*se_S[1][pi]);

        const char *sig = "";
        /* Signal: is |Δ| > 2σ_Δ ? (2-sigma significance) */
        if (se_delta > 1e-10) {
            double z_score = fabs(delta) / se_delta;
            if (z_score < 1.0) sig = "  noise";
            else if (z_score < 2.0) sig = "  ~1σ";
            else sig = "  >2σ ★";
        }
        /* Check for crossing relative to previous point */
        if (pi > 0) {
            double pd = mean_S[0][pi-1] - mean_S[1][pi-1];
            if ((pd > 0 && delta < 0) || (pd < 0 && delta > 0)) {
                double se_pd = sqrt(se_S[0][pi-1]*se_S[0][pi-1] +
                                    se_S[1][pi-1]*se_S[1][pi-1]);
                if (fabs(pd) > 2*se_pd || fabs(delta) > 2*se_delta)
                    sig = "  ← p_c ★★";
                else
                    sig = "  ← cross?";
            }
        }

        printf("  ║  %.2f  | %.4f ± %.4f | %.4f ± %.4f | %+.4f±%.3f|%s\n",
               rates[pi],
               mean_S[0][pi], se_S[0][pi],
               mean_S[1][pi], se_S[1][pi],
               delta, se_delta, sig);
    }

    printf("  ║                                                                        ║\n");
    printf("  ║  Signal column: ★★ = statistically significant crossing (>2σ)          ║\n");
    printf("  ║                 cross? = sign change but within error bars              ║\n");
    printf("  ║                 noise = Δ within 1σ of zero                             ║\n");
    printf("  ║                                                                        ║\n");
    printf("  ║  Dimensional progression of p_c:                                       ║\n");
    printf("  ║    3D: p_c ≈ 0.05–0.10  (6 neighbors/site)                            ║\n");
    printf("  ║    4D: p_c ≈ 0.10       (8 neighbors/site)                            ║\n");
    printf("  ║    5D: p_c ≈ ???        (10 neighbors/site) — needs re-averaging       ║\n");
    printf("  ║    6D: p_c = see above  (12 neighbors/site)                            ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════════════╝\n\n");

    free(G_re); free(G_im);
    return 0;
}
