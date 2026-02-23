/*
 * anderson_3d.c — 3D Anderson Localization (The Disorder Puddles)
 *
 * Feeds the engine maximum noise and watches it isolate surviving structures.
 *
 * Physics:
 *   Each site gets a random on-site energy h_i(k) drawn from [-W, W].
 *   The 1-site disorder gate applies: U_i|k⟩ = exp(δτ·h_i(k)) |k⟩
 *   The 2-site DFT₆ mixing + diagonal clock gates provide hopping.
 *
 *   Weak disorder (W ≪ J): wavefunctions delocalize — quantum liquid
 *   Strong disorder (W ≫ J): wavefunctions freeze — Anderson localization
 *   Critical W_c: the metal-insulator transition in 3D
 *
 *   The engine's sparse registers map the localized puddles effortlessly
 *   while ignoring the dead space between them. The macroscopic conductance
 *   drops to zero as disorder increases.
 *
 * Observable:
 *   Inverse Participation Ratio (IPR): measures localization
 *     IPR ~ 1/N  → delocalized (metal)
 *     IPR ~ 1    → localized (insulator)
 *
 * Build:
 *   gcc -O2 -std=gnu11 anderson_3d.c quhit_core.c quhit_gates.c \
 *       quhit_measure.c quhit_entangle.c quhit_register.c -lm -o anderson_3d
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "peps3d_overlay.c"

/* ═══════════════ Constants ═══════════════ */

#define LX 3
#define LY 3
#define LZ 3
#define NSITES (LX * LY * LZ)
#define TROTTER_STEPS  20
#define HOPPING_J      1.0
#define HOPPING_DTAU   1.0

/* ═══════════════ DFT₆ 1-site mixing ═══════════════ */

static double DFT_RE[36], DFT_IM[36];

static void build_dft6(void)
{
    int D = 6;
    double inv = 1.0 / sqrt((double)D);
    double omega = 2.0 * M_PI / D;
    for (int j = 0; j < D; j++)
     for (int k = 0; k < D; k++) {
         DFT_RE[j * D + k] = inv * cos(omega * j * k);
         DFT_IM[j * D + k] = inv * sin(omega * j * k);
     }
}

/* ═══════════════ Diagonal clock gate (hopping) ═══════════════ */

static void build_hop_gate(double J, double dtau,
                            double *G_re, double *G_im)
{
    int D = 6, D2 = D * D;
    memset(G_re, 0, D2 * D2 * sizeof(double));
    memset(G_im, 0, D2 * D2 * sizeof(double));
    double omega = 2.0 * M_PI / 6.0;
    for (int a = 0; a < D; a++)
     for (int b = 0; b < D; b++) {
         int diff = ((a - b) % D + D) % D;
         double w = exp(dtau * J * cos(omega * diff));
         int idx = a * D + b;
         G_re[idx * D2 + idx] = w;
     }
}

/* ═══════════════ Disorder gate (random on-site potential) ═══════════════ */

/*
 * For each site i, generate a random energy landscape h_i(k)
 * for k = 0..5, drawn from Uniform[-W, W].
 *
 * The disorder gate is diagonal:
 *   U_i|k⟩ = exp(+δτ · h_i(k)) |k⟩
 *
 * This is a Boltzmann filter: states with low energy are amplified,
 * states with high energy are suppressed. Combined with renormalization,
 * this creates a random energy landscape that traps wavefunctions.
 */
static double site_disorder[NSITES][6];  /* h_i(k) */

static void generate_disorder(double W)
{
    for (int i = 0; i < NSITES; i++)
     for (int k = 0; k < 6; k++)
         site_disorder[i][k] = W * (2.0 * ((double)rand() / RAND_MAX) - 1.0);
}

static void build_disorder_gate(int site_flat, double dtau,
                                 double *U_re, double *U_im)
{
    int D = 6;
    memset(U_re, 0, D * D * sizeof(double));
    memset(U_im, 0, D * D * sizeof(double));
    for (int k = 0; k < D; k++) {
        double w = exp(dtau * site_disorder[site_flat][k]);
        U_re[k * D + k] = w;
    }
}

/* ═══════════════ Diagnostics ═══════════════ */

static double site_entropy(Tns3dGrid *g, int x, int y, int z)
{
    double p[6]; tns3d_local_density(g, x, y, z, p);
    double S = 0;
    for (int k = 0; k < 6; k++)
        if (p[k] > 1e-15) S -= p[k] * log2(p[k]);
    return S;
}

static double avg_entropy(Tns3dGrid *g)
{
    double t = 0;
    for (int z = 0; z < g->Lz; z++)
     for (int y = 0; y < g->Ly; y++)
      for (int x = 0; x < g->Lx; x++)
          t += site_entropy(g, x, y, z);
    return t / NSITES;
}

/*
 * Inverse Participation Ratio (IPR):
 *   For each color k, measure the participation across sites.
 *   IPR_k = Σ_i p_i(k)²   where p_i(k) is prob of color k at site i
 *   For uniform distribution: IPR = 1/N
 *   For delta-localized: IPR = 1
 *   Average over all colors.
 */
static double compute_ipr(Tns3dGrid *g)
{
    double color_weight[NSITES][6];
    for (int z = 0; z < g->Lz; z++)
     for (int y = 0; y < g->Ly; y++)
      for (int x = 0; x < g->Lx; x++) {
          int flat = z * LY * LX + y * LX + x;
          tns3d_local_density(g, x, y, z, color_weight[flat]);
      }

    /* For each site, compute max probability (purity) */
    double ipr = 0;
    for (int i = 0; i < NSITES; i++) {
        double sum_sq = 0;
        for (int k = 0; k < 6; k++)
            sum_sq += color_weight[i][k] * color_weight[i][k];
        ipr += sum_sq;
    }
    return ipr / NSITES;  /* average per-site IPR */
}

/* Count "frozen" sites (entropy < 0.5 bits) */
static int count_frozen(Tns3dGrid *g)
{
    int n = 0;
    for (int z = 0; z < g->Lz; z++)
     for (int y = 0; y < g->Ly; y++)
      for (int x = 0; x < g->Lx; x++)
          if (site_entropy(g, x, y, z) < 0.5) n++;
    return n;
}

/* Total NNZ */
static long total_nnz(Tns3dGrid *g)
{
    long total = 0;
    for (int i = 0; i < NSITES; i++) {
        int reg = g->site_reg[i];
        if (reg >= 0)
            total += g->eng->registers[reg].num_nonzero;
    }
    return total;
}

static void renormalize_all(Tns3dGrid *g)
{
    for (int i = 0; i < NSITES; i++) {
        int reg = g->site_reg[i];
        if (reg < 0) continue;
        QuhitRegister *r = &g->eng->registers[reg];
        double n2 = 0;
        for (uint32_t e = 0; e < r->num_nonzero; e++)
            n2 += r->entries[e].amp_re * r->entries[e].amp_re +
                  r->entries[e].amp_im * r->entries[e].amp_im;
        if (n2 > 1e-20) {
            double inv = 1.0 / sqrt(n2);
            for (uint32_t e = 0; e < r->num_nonzero; e++) {
                r->entries[e].amp_re *= inv;
                r->entries[e].amp_im *= inv;
            }
        }
    }
}

/* ═══════════════ Single Anderson Run ═══════════════ */

typedef struct {
    double W;
    double final_entropy;
    double final_ipr;
    int    frozen_sites;
    long   final_nnz;
    double time_s;
} AndersonResult;

static AndersonResult run_anderson(double W,
                                    double *hop_re, double *hop_im)
{
    AndersonResult res;
    res.W = W;

    printf("\n  ── Disorder W = %.2f ──\n\n", W);

    /* Generate fresh random disorder */
    generate_disorder(W);

    /* Initialize lattice in |0⟩ product state */
    Tns3dGrid *g = tns3d_init(LX, LY, LZ);

    printf("  %4s  %7s  %7s  %6s  %7s  %8s\n",
           "Step", "⟨S⟩", "IPR", "Frozen", "NNZ", "Time(s)");
    printf("  ────  ───────  ───────  ──────  ───────  ────────\n");

    double total_time = 0;

    for (int step = 1; step <= TROTTER_STEPS; step++) {
        clock_t t0 = clock();

        /* 1. DFT₆ mixing on all sites (hopping/delocalization) */
        tns3d_gate_1site_all(g, DFT_RE, DFT_IM);

        /* 2. Diagonal clock gates (nearest-neighbor correlation) */
        tns3d_gate_x_all(g, hop_re, hop_im);
        tns3d_gate_y_all(g, hop_re, hop_im);
        tns3d_gate_z_all(g, hop_re, hop_im);

        /* 3. Disorder gates (random on-site potential) */
        {
            double U_re[36], U_im[36];
            for (int z = 0; z < LZ; z++)
             for (int y = 0; y < LY; y++)
              for (int x = 0; x < LX; x++) {
                  int flat = z * LY * LX + y * LX + x;
                  build_disorder_gate(flat, HOPPING_DTAU, U_re, U_im);
                  tns3d_gate_1site(g, x, y, z, U_re, U_im);
              }
        }

        renormalize_all(g);

        double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
        total_time += dt;

        double sav = avg_entropy(g);
        double ipr = compute_ipr(g);
        int frozen = count_frozen(g);
        long nnz = total_nnz(g);

        printf("  %4d  %7.4f  %7.4f  %4d/%d  %7ld  %8.3f\n",
               step, sav, ipr, frozen, NSITES, nnz, dt);
    }

    /* Final measurements */
    res.final_entropy = avg_entropy(g);
    res.final_ipr = compute_ipr(g);
    res.frozen_sites = count_frozen(g);
    res.final_nnz = total_nnz(g);
    res.time_s = total_time;

    /* Print spatial map: entropy at each site */
    printf("\n  Entropy map (z=1 slice, y↑ x→):\n");
    for (int y = LY - 1; y >= 0; y--) {
        printf("    ");
        for (int x = 0; x < LX; x++) {
            double S = site_entropy(g, x, y, 1);
            if (S < 0.3)      printf("  ·  ");  /* frozen (dead space) */
            else if (S < 1.0) printf(" ░%.1f", S);  /* partially localized */
            else if (S < 2.0) printf(" ▓%.1f", S);  /* fluid puddle */
            else              printf(" █%.1f", S);  /* fully delocalized */
        }
        printf("\n");
    }

    /* Print dominant color at each site */
    printf("\n  Dominant color map (z=1 slice):\n");
    for (int y = LY - 1; y >= 0; y--) {
        printf("    ");
        for (int x = 0; x < LX; x++) {
            double p[6]; tns3d_local_density(g, x, y, 1, p);
            int dom = 0;
            for (int k = 1; k < 6; k++)
                if (p[k] > p[dom]) dom = k;
            if (p[dom] > 0.8) printf("  [%d] ", dom);  /* strongly localized */
            else if (p[dom] > 0.5) printf("  (%d) ", dom);  /* weakly localized */
            else printf("  {%d} ", dom);  /* delocalized */
        }
        printf("\n");
    }

    printf("\n  ────────────────────────────────────────\n");
    printf("  W=%.2f: ⟨S⟩=%.4f  IPR=%.4f  Frozen=%d/%d  NNZ=%ld  Time=%.2fs\n",
           W, res.final_entropy, res.final_ipr, res.frozen_sites, NSITES,
           res.final_nnz, res.time_s);

    tns3d_free(g);
    return res;
}

/* ═══════════════ Main ═══════════════ */

int main(void)
{
    srand((unsigned)time(NULL));

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  3D ANDERSON LOCALIZATION — The Disorder Puddles           ║\n");
    printf("║  ──────────────────────────────────────────────────────────  ║\n");
    printf("║  Lattice: %d×%d×%d = %d sites (cubic)                       ║\n",
           LX, LY, LZ, NSITES);
    printf("║  Hilbert space: 6^%d ≈ 10^%.1f dimensions                  ║\n",
           NSITES, NSITES * log10(6.0));
    printf("║  χ=%d, J=%.1f, δτ=%.1f, %d Trotter steps                  ║\n",
           TNS3D_CHI, HOPPING_J, HOPPING_DTAU, TROTTER_STEPS);
    printf("║  Protocol: DFT₆ hop + clock correlate + random disorder    ║\n");
    printf("║  Sweep: W = 0 → 20 (find localization transition)         ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");

    /* Build hopping gate */
    build_dft6();
    double *hop_re = calloc(36*36, sizeof(double));
    double *hop_im = calloc(36*36, sizeof(double));
    build_hop_gate(HOPPING_J, HOPPING_DTAU, hop_re, hop_im);

    /* Sweep disorder strengths */
    double W_vals[] = { 0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0 };
    int n_W = sizeof(W_vals) / sizeof(W_vals[0]);
    AndersonResult results[7];

    for (int i = 0; i < n_W; i++)
        results[i] = run_anderson(W_vals[i], hop_re, hop_im);

    /* ═══════════════ SUMMARY ═══════════════ */

    printf("\n  ╔══════════════════════════════════════════════════════════════╗\n");
    printf("  ║  ANDERSON LOCALIZATION PHASE DIAGRAM                       ║\n");
    printf("  ╠══════════════════════════════════════════════════════════════╣\n");
    printf("  ║                                                            ║\n");
    printf("  ║   W       ⟨S⟩      IPR     Frozen   NNZ     Phase         ║\n");
    printf("  ║  ─────  ───────  ───────  ──────  ───────  ──────────     ║\n");

    for (int i = 0; i < n_W; i++) {
        const char *phase;
        if (results[i].final_ipr > 0.7)       phase = "LOCALIZED ";
        else if (results[i].final_ipr > 0.4)  phase = "CRITICAL  ";
        else if (results[i].final_ipr > 0.25) phase = "DIFFUSIVE ";
        else                                   phase = "METALLIC  ";

        printf("  ║  %5.1f  %7.4f  %7.4f  %4d/%d  %5ld  %s    ║\n",
               results[i].W, results[i].final_entropy, results[i].final_ipr,
               results[i].frozen_sites, NSITES, results[i].final_nnz, phase);
    }

    printf("  ║                                                            ║\n");
    printf("  ║  W < W_c:  Delocalized (metallic) — wavefunctions spread  ║\n");
    printf("  ║  W > W_c:  Anderson localized — frozen disorder puddles   ║\n");
    printf("  ║  W = W_c:  Metal-insulator transition (3D critical point) ║\n");
    printf("  ║                                                            ║\n");
    printf("  ║  Sparse registers map puddles, ignore dead space.          ║\n");
    printf("  ║  NNZ tracks computational cost: localized = cheap.         ║\n");
    printf("  ╚══════════════════════════════════════════════════════════════╝\n");

    free(hop_re); free(hop_im);
    return 0;
}
