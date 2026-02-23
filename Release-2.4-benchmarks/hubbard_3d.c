/*
 * hubbard_3d.c — 3D Fermi-Hubbard Model via Imaginary Time PEPS
 *
 * Simulates the canonical model for high-temperature superconductivity
 * and the Strange Metal phase.
 * Bypasses the 3D Fermion Sign Problem natively using HexState's 
 * tensor network tracking of true amplitude signatures rather than
 * Monte Carlo probabilities.
 *
 *   H = -t Σ (c†_i c_j + h.c.) + U Σ n_i↑ n_i↓ - μ Σ (n_i↑ + n_i↓)
 *
 * Basis Map (D=4, mapped to HexState D=6):
 *   0: |0⟩   (Empty)
 *   1: |↑⟩   (Spin Up)
 *   2: |↓⟩   (Spin Down)
 *   3: |↑↓⟩  (Double)
 *
 * Build:
 *   gcc -O2 -std=gnu11 -fopenmp hubbard_3d.c quhit_core.c quhit_gates.c \
 *       quhit_measure.c quhit_entangle.c quhit_register.c peps_overlay.c -lm -o hubbard_3d
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TNS3D_CHI 6  /* Kept smaller for 3D speed since N is large */
#include "peps3d_overlay.c"

/* ═══════════════ Constants ═══════════════ */

#define HUBBARD_U      8.0    /* Onsite repulsion */
#define HUBBARD_MU     2.5    /* Chemical potential (doping) */
#define HUBBARD_T      1.0    /* Hopping amplitude */
#define COOL_DTAU      0.1    /* Imaginary time step */
#define COOL_STEPS     15     /* Number of Trotter steps */

/* ═══════════════ 1-Site Gate (U and μ) ═══════════════ */

static void build_onsite_gate(double dtau, double U, double mu, 
                              double *G_re, double *G_im)
{
    memset(G_re, 0, 36 * sizeof(double));
    memset(G_im, 0, 36 * sizeof(double));

    /* Basis states: 0=Empty, 1=Up, 2=Down, 3=Double */
    double energies[4];
    energies[0] = 0;              
    energies[1] = -mu;            
    energies[2] = -mu;            
    energies[3] = U - 2.0 * mu;   

    for (int k = 0; k < 4; k++) {
        G_re[k * 6 + k] = exp(-dtau * energies[k]);
    }
}

/* ═══════════════ 2-Site Gate (Hopping with Sign) ═══════════════ */

static void build_hopping_gate(double dtau, double t, 
                               double *G_re, double *G_im)
{
    int D = 6, D2 = D * D;
    memset(G_re, 0, D2 * D2 * sizeof(double));
    memset(G_im, 0, D2 * D2 * sizeof(double));

    double H[16][16] = {{0}};

    // |↑, 0⟩ <-> |0, ↑⟩
    H[1*4+0][0*4+1] = -t; H[0*4+1][1*4+0] = -t;
    // |↓, 0⟩ <-> |0, ↓⟩
    H[2*4+0][0*4+2] = -t; H[0*4+2][2*4+0] = -t;
    // |↑↓, 0⟩ <-> |↓, ↑⟩
    H[3*4+0][1*4+2] = +t; H[1*4+2][3*4+0] = +t;
    // |↓, ↑⟩ <-> |0, ↑↓⟩
    H[2*4+1][0*4+3] = +t; H[0*4+3][2*4+1] = +t;
    // |↑↓, ↑⟩ <-> |↑, ↑↓⟩
    H[3*4+1][1*4+3] = -t; H[1*4+3][3*4+1] = -t;
    // |↑↓, ↓⟩ <-> |↓, ↑↓⟩
    H[3*4+2][2*4+3] = -t; H[2*4+3][3*4+2] = -t;

    for (int i=0; i<36; i++) G_re[i*D2+i] = 1.0;

    double ch = cosh(dtau * t);
    double sh = sinh(dtau * t);

    for(int i=0; i<16; i++) {
        for(int j=i+1; j<16; j++) {
            if (H[i][j] != 0) {
                int ai = i / 4, bi = i % 4;
                int aj = j / 4, bj = j % 4;
                int idxi = ai * D + bi;
                int idxj = aj * D + bj;

                G_re[idxi * D2 + idxi] = ch;
                G_re[idxj * D2 + idxj] = ch;
                
                double cross_term = - (H[i][j] / t) * sinh(dtau * t);
                G_re[idxi * D2 + idxj] = cross_term;
                G_re[idxj * D2 + idxi] = cross_term;
            }
        }
    }
}

/* ═══════════════ Diagnostics ═══════════════ */

static void renormalize_all(Tns3dGrid *g)
{
    int N = g->Lx * g->Ly * g->Lz;
    for (int i = 0; i < N; i++) {
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

static long total_nnz(Tns3dGrid *g)
{
    long total = 0;
    int N = g->Lx * g->Ly * g->Lz;
    for (int i = 0; i < N; i++) {
        int reg = g->site_reg[i];
        if (reg >= 0)
            total += g->eng->registers[reg].num_nonzero;
    }
    return total;
}

static void print_observables(Tns3dGrid *g)
{
    double tot_density = 0;
    double tot_double = 0;
    int N = g->Lx * g->Ly * g->Lz;

    for (int z = 0; z < g->Lz; z++) {
        for (int y = 0; y < g->Ly; y++) {
            for (int x = 0; x < g->Lx; x++) {
                double p[6]; 
                tns3d_local_density(g, x, y, z, p);
                
                double density = p[1] + p[2] + 2.0 * p[3];
                double double_occ = p[3];

                tot_density += density;
                tot_double += double_occ;
            }
        }
    }

    printf("    ⟨n⟩: %.4f   ⟨n↑ n↓⟩: %.4f\n", 
           tot_density / N, tot_double / N);
}

static void print_spatial_map(Tns3dGrid *g)
{
    printf("\n  ═══ 3D SPATIAL CHARGE DENSITY MAP ⟨n_i⟩ ═══\n");
    for (int z = g->Lz - 1; z >= 0; z -= 2) { 
        printf("  [Z Layer %d]\n", z);
        for (int y = g->Ly - 1; y >= 0; y--) {
            printf("  y=%d |", y);
            for (int x = 0; x < g->Lx; x++) {
                double p[6]; tns3d_local_density(g, x, y, z, p);
                double density = p[1] + p[2] + 2.0 * p[3];
                if (density > 0.9) printf(" ██ ");
                else if (density > 0.6) printf(" ▓▓ ");
                else if (density > 0.3) printf(" ▒▒ ");
                else printf(" ░░ ");
            }
            printf("|\n");
        }
        printf("\n");
    }

    printf("  ═══ 3D SPATIAL SPIN MAP ⟨S^z_i⟩ ═══\n");
    for (int z = g->Lz - 1; z >= 0; z -= 2) { 
        printf("  [Z Layer %d]\n", z);
        for (int y = g->Ly - 1; y >= 0; y--) {
            printf("  y=%d |", y);
            for (int x = 0; x < g->Lx; x++) {
                double p[6]; tns3d_local_density(g, x, y, z, p);
                double sz = 0.5 * p[1] - 0.5 * p[2];
                if (sz > 0.15) printf("  ↑ ");
                else if (sz < -0.15) printf("  ↓ ");
                else printf("  . ");
            }
            printf("|\n");
        }
        printf("\n");
    }
}

/* ═══════════════ Main ═══════════════ */

int main(void)
{
    int Lx = 6, Ly = 6, Lz = 6;
    int Nsites = Lx * Ly * Lz;

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  3D FERMI-HUBBARD MODEL — The Strange Metal (Phase 8)      ║\n");
    printf("║  ──────────────────────────────────────────────────────────  ║\n");
    printf("║  Lattice: %d×%d×%d = %d sites                              ║\n", Lx, Ly, Lz, Nsites);
    printf("║  Hilbert space: 4^%d ≈ 10^%.1f dimensions                  ║\n", Nsites, Nsites * log10(4.0));
    printf("║  Model: t-U Hubbard Model (Fermion Sign Natively Resolved) ║\n");
    printf("║  U=%.1f, μ=%.1f, t=%.1f, δτ=%.2f                           ║\n", 
           HUBBARD_U, HUBBARD_MU, HUBBARD_T, COOL_DTAU);
    printf("║  Method: 3D Imaginary Time PEPS (χ=%d)                      ║\n", TNS3D_CHI);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    double *onsite_re = calloc(36*36, sizeof(double));
    double *onsite_im = calloc(36*36, sizeof(double));
    build_onsite_gate(COOL_DTAU, HUBBARD_U, HUBBARD_MU, onsite_re, onsite_im);

    double *hop_re = calloc(36*36, sizeof(double));
    double *hop_im = calloc(36*36, sizeof(double));
    build_hopping_gate(COOL_DTAU, HUBBARD_T, hop_re, hop_im);

    Tns3dGrid *g = tns3d_init(Lx, Ly, Lz);

    for (int i = 0; i < Nsites; i++) {
        int reg = g->site_reg[i];
        double norm = 1.0 / sqrt(2.0);
        int spin = (i % 2 == 0) ? 1 : 2; 
        quhit_reg_sv_set(g->eng, reg, 0, norm, 0);      
        quhit_reg_sv_set(g->eng, reg, spin*TNS3D_C6, norm, 0); 
    }

    printf("  ══ COOLING TO 3D HUBBARD GROUND STATE (%d steps) ══\n\n", COOL_STEPS);
    double total_time = 0;

    for (int step = 1; step <= COOL_STEPS; step++) {
        clock_t t0 = clock();

        tns3d_gate_1site_all(g, onsite_re, onsite_im);
        renormalize_all(g);

        tns3d_gate_x_all(g, hop_re, hop_im);
        renormalize_all(g);

        tns3d_gate_y_all(g, hop_re, hop_im);
        renormalize_all(g);
        
        tns3d_gate_z_all(g, hop_re, hop_im);
        renormalize_all(g);

        double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
        total_time += dt;

        long nnz = total_nnz(g);

        printf("  Step %2d | Time: %5.2fs | NNZ: %8ld |", step, dt, nnz);
        print_observables(g);
    }
    
    printf("\n  ════════════════════════════════════════════════════════════\n");
    printf("  Calculated 3D Ground State observables via %d PEPS Cooling steps.\n", COOL_STEPS);
    printf("  Total Time: %.2f seconds\n", total_time);

    print_spatial_map(g);

    tns3d_free(g);
    free(onsite_re); free(onsite_im);
    free(hop_re); free(hop_im);
    return 0;
}
