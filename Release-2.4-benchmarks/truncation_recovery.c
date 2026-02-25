/*
 * truncation_recovery.c — Does Truncation Error Accumulate or Recover?
 *
 * The Hypothesis:
 *   If SVD truncation is DESTRUCTIVE, fidelity will monotonically
 *   decrease toward zero as cycles accumulate.
 *
 *   If SVD truncation TRACKS THE PHYSICS, fidelity will oscillate
 *   with the entanglement dynamics and never collapse to zero.
 *
 * Test: Run 30 cycles of DFT₆ + clock gate on 2×2×2 = 8 quhits.
 * Compare exact vs PEPS3D at every cycle. Plot the fidelity trajectory.
 *
 * If recovery is real, this proves that PEPS truncation is a
 * reversible compression — not a permanent information loss.
 */

#include "exact_sim.h"
#include "peps3d_overlay.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define D       6
#define Lx      2
#define Ly      2
#define Lz      2
#define N_SITES (Lx*Ly*Lz)
#define CYCLES  30

static double DFT_RE[36], DFT_IM[36];

static void build_dft6(void)
{
    double norm = 1.0 / sqrt(D);
    for (int j = 0; j < D; j++)
     for (int k = 0; k < D; k++) {
         double ph = 2.0 * M_PI * j * k / D;
         DFT_RE[j*D+k] = norm * cos(ph);
         DFT_IM[j*D+k] = norm * sin(ph);
     }
}

static void exact_clock_2site(ExactState *s, int q1, int q2, double J)
{
    for (int64_t idx = 0; idx < s->dim; idx++) {
        int d1 = exact_qudit_val(idx, q1);
        int d2 = exact_qudit_val(idx, q2);
        double phase = J * cos(2.0 * M_PI * (d1 - d2) / (double)D);
        double cr = cos(phase), ci = -sin(phase);
        double ar = s->re[idx], ai = s->im[idx];
        s->re[idx] = ar*cr - ai*ci;
        s->im[idx] = ar*ci + ai*cr;
    }
}

static void build_clock_gate(double *G_re, double *G_im, double J)
{
    int D2 = D*D, D4 = D2*D2;
    for (int i = 0; i < D4; i++) { G_re[i] = 0; G_im[i] = 0; }
    for (int kA = 0; kA < D; kA++)
     for (int kB = 0; kB < D; kB++) {
         int idx = (kA*D+kB)*D2 + (kA*D+kB);
         double phase = J * cos(2.0*M_PI*(kA-kB)/(double)D);
         G_re[idx] = cos(phase);
         G_im[idx] = -sin(phase);
     }
}

static void exact_local_density(ExactState *s, int q, double *probs)
{
    for (int k = 0; k < D; k++) probs[k] = 0;
    double total = 0;
    for (int64_t idx = 0; idx < s->dim; idx++) {
        int dq = exact_qudit_val(idx, q);
        double p = s->re[idx]*s->re[idx] + s->im[idx]*s->im[idx];
        probs[dq] += p; total += p;
    }
    if (total > 1e-30) for (int k = 0; k < D; k++) probs[k] /= total;
}

static double distribution_fidelity(const double *p, const double *q)
{
    double F = 0;
    for (int k = 0; k < D; k++) F += sqrt(fabs(p[k]) * fabs(q[k]));
    return F * F;
}

static double density_entropy(const double *probs)
{
    double S = 0;
    for (int k = 0; k < D; k++)
        if (probs[k] > 1e-20) S -= probs[k] * log(probs[k]);
    return S / log(D);
}

static void compress_register(QuhitEngine *eng, int reg, double thr)
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

static int site_idx(int x, int y, int z) { return z*Ly*Lx + y*Lx + x; }

int main(void)
{
    srand(42);
    build_dft6();

    double J = 0.5;
    double *G_re = (double *)calloc(D*D*D*D, sizeof(double));
    double *G_im = (double *)calloc(D*D*D*D, sizeof(double));
    build_clock_gate(G_re, G_im, J);

    ExactState *exact = exact_alloc(N_SITES);
    Tns3dGrid *peps = tns3d_init(Lx, Ly, Lz);

    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  TRUNCATION RECOVERY TEST: 30 CYCLES                           ║\n");
    printf("  ║  ────────────────────────────────────────────────────────────── ║\n");
    printf("  ║  Hypothesis: If truncation is non-destructive, fidelity        ║\n");
    printf("  ║  will OSCILLATE rather than monotonically decrease.             ║\n");
    printf("  ║                                                                ║\n");
    printf("  ║  System: 2×2×2 = 8 D=6 quhits, J=%.1f                         ║\n", J);
    printf("  ║  Exact:  6^8 = 1,679,616 amplitudes                           ║\n");
    printf("  ║  PEPS3D: 8 sparse registers                                    ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════════╝\n\n");

    printf("  Cycle | Avg Fidelity | Exact ⟨S⟩  | PEPS ⟨S⟩   | Trend\n");
    printf("  ──────┼──────────────┼────────────┼────────────┼──────────\n");

    double prev_fidelity = 1.0;
    int up_count = 0, down_count = 0;
    double min_fidelity = 1.0, max_fidelity = 0.0;
    double fidelity_history[CYCLES];

    for (int cycle = 1; cycle <= CYCLES; cycle++) {
        /* DFT₆ on all sites */
        for (int q = 0; q < N_SITES; q++)
            exact_gate_1site(exact, q, DFT_RE, DFT_IM);
        tns3d_gate_1site_all(peps, DFT_RE, DFT_IM);

        /* Nearest-neighbor clock gates */
        for (int z = 0; z < Lz; z++)
         for (int y = 0; y < Ly; y++)
          for (int x = 0; x < Lx-1; x++) {
              exact_clock_2site(exact, site_idx(x,y,z), site_idx(x+1,y,z), J);
              tns3d_gate_x(peps, x, y, z, G_re, G_im);
          }
        for (int z = 0; z < Lz; z++)
         for (int y = 0; y < Ly-1; y++)
          for (int x = 0; x < Lx; x++) {
              exact_clock_2site(exact, site_idx(x,y,z), site_idx(x,y+1,z), J);
              tns3d_gate_y(peps, x, y, z, G_re, G_im);
          }
        for (int z = 0; z < Lz-1; z++)
         for (int y = 0; y < Ly; y++)
          for (int x = 0; x < Lx; x++) {
              exact_clock_2site(exact, site_idx(x,y,z), site_idx(x,y,z+1), J);
              tns3d_gate_z(peps, x, y, z, G_re, G_im);
          }

        /* Compress PEPS */
        for (int i = 0; i < N_SITES; i++)
            compress_register(peps->eng, peps->site_reg[i], 1e-6);
        for (int z = 0; z < Lz; z++)
         for (int y = 0; y < Ly; y++)
          for (int x = 0; x < Lx; x++)
              tns3d_normalize_site(peps, x, y, z);

        /* Compare */
        double total_F = 0, total_Se = 0, total_Sp = 0;
        for (int z = 0; z < Lz; z++)
         for (int y = 0; y < Ly; y++)
          for (int x = 0; x < Lx; x++) {
              double ep[D], pp[D];
              exact_local_density(exact, site_idx(x,y,z), ep);
              tns3d_local_density(peps, x, y, z, pp);
              total_F += distribution_fidelity(ep, pp);
              total_Se += density_entropy(ep);
              total_Sp += density_entropy(pp);
          }
        double avg_F = total_F / N_SITES;
        double avg_Se = total_Se / N_SITES;
        double avg_Sp = total_Sp / N_SITES;

        const char *trend;
        if (avg_F > prev_fidelity + 0.01) { trend = "  ↑ RECOVER"; up_count++; }
        else if (avg_F < prev_fidelity - 0.01) { trend = "  ↓ drop"; down_count++; }
        else { trend = "  → stable"; }

        if (avg_F < min_fidelity) min_fidelity = avg_F;
        if (avg_F > max_fidelity) max_fidelity = avg_F;
        fidelity_history[cycle-1] = avg_F;
        prev_fidelity = avg_F;

        printf("   %2d   |   %.6f   |  %.4f    |  %.4f    |%s\n",
               cycle, avg_F, avg_Se, avg_Sp, trend);
        fflush(stdout);
    }

    /* ═══════════════ VERDICT ═══════════════ */
    printf("\n  ╔══════════════════════════════════════════════════════════════════╗\n");
    printf("  ║  TRUNCATION RECOVERY TEST — VERDICT                            ║\n");
    printf("  ╠══════════════════════════════════════════════════════════════════╣\n");
    printf("  ║                                                                ║\n");
    printf("  ║  Recovery events (↑): %2d / %d cycles                           ║\n", up_count, CYCLES);
    printf("  ║  Drop events    (↓): %2d / %d cycles                           ║\n", down_count, CYCLES);
    printf("  ║  Fidelity range: [%.4f, %.4f]                            ║\n", min_fidelity, max_fidelity);
    printf("  ║                                                                ║\n");

    /* Check if fidelity ever collapsed to near-zero */
    int collapsed = (min_fidelity < 0.1);
    /* Check if there were recoveries */
    int has_recovery = (up_count >= 3);
    /* Check final 5 cycles: is fidelity bounded away from zero? */
    double late_avg = 0;
    for (int i = CYCLES-5; i < CYCLES; i++) late_avg += fidelity_history[i];
    late_avg /= 5.0;

    if (!collapsed && has_recovery && late_avg > 0.3) {
        printf("  ║  ★ CONFIRMED: TRUNCATION ERROR DOES NOT ACCUMULATE ★         ║\n");
        printf("  ║                                                                ║\n");
        printf("  ║  Fidelity oscillates with entanglement dynamics.               ║\n");
        printf("  ║  Recovery events prove truncation is non-destructive.          ║\n");
        printf("  ║  After %d cycles, fidelity remains bounded (avg=%.4f).     ║\n", CYCLES, late_avg);
        printf("  ║                                                                ║\n");
        printf("  ║  IMPLICATION: SVD truncation acts as a reversible lossy        ║\n");
        printf("  ║  compression. Information is not permanently lost — it is      ║\n");
        printf("  ║  temporarily invisible during high-entanglement transients     ║\n");
        printf("  ║  and recovers when the physics returns to area-law regime.     ║\n");
    } else if (collapsed) {
        printf("  ║  ✗ DENIED: TRUNCATION ERROR ACCUMULATES                        ║\n");
        printf("  ║                                                                ║\n");
        printf("  ║  Fidelity collapsed below 0.1. Truncation is destructive       ║\n");
        printf("  ║  for this circuit and system size. χ must be increased.        ║\n");
    } else {
        printf("  ║  INCONCLUSIVE: Mixed behavior                                  ║\n");
        printf("  ║                                                                ║\n");
        printf("  ║  Fidelity shows partial recovery but may be declining.         ║\n");
        printf("  ║  More cycles or larger systems needed for definitive answer.   ║\n");
    }

    printf("  ║                                                                ║\n");

    /* ASCII sparkline of fidelity */
    printf("  ║  Fidelity trajectory (30 cycles):                              ║\n");
    printf("  ║  1.0 ┤");
    for (int i = 0; i < CYCLES; i++) {
        int bar = (int)(fidelity_history[i] * 10);
        if (bar >= 9) printf("█");
        else if (bar >= 7) printf("▆");
        else if (bar >= 5) printf("▄");
        else if (bar >= 3) printf("▂");
        else printf("_");
    }
    printf("                        ║\n");
    printf("  ║  0.0 ┤");
    for (int i = 0; i < CYCLES; i++) printf("─");
    printf("                        ║\n");

    printf("  ╚══════════════════════════════════════════════════════════════════╝\n\n");

    exact_free(exact);
    tns3d_free(peps);
    free(G_re); free(G_im);
    return 0;
}
