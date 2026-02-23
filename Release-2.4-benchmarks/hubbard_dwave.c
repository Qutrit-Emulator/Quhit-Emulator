/*
 * hubbard_dwave.c — 2D Fermi-Hubbard Model
 * Probing for d-Wave Superconducting Paring (ODLRO)
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

#define TNS3D_CHI 12
#include "peps3d_overlay.c"

/* ═══════════════ Constants ═══════════════ */

#define HUBBARD_U      4.0    /* Onsite repulsion */
#define HUBBARD_MU     1.0    /* Hole-doped regime */
#define HUBBARD_T      1.0    /* Hopping amplitude */
#define DWAVE_DELTA    2.0    /* Boundary pairing field amplitude */
#define COOL_DTAU      0.1    
#define COOL_STEPS     30     

/* ═══════════════ 1-Site Gate ═══════════════ */

static void build_onsite_gate(double dtau, double U, double mu, 
                              double *G_re, double *G_im)
{
    memset(G_re, 0, 36 * sizeof(double));
    memset(G_im, 0, 36 * sizeof(double));
    double energies[4];
    energies[0] = 0;
    energies[1] = -mu;
    energies[2] = -mu;
    energies[3] = U - 2.0 * mu;

    for (int k = 0; k < 4; k++) {
        G_re[k * 6 + k] = exp(-dtau * energies[k]);
    }
}

/* ═══════════════ 2-Site Hop+Pair Gate ═══════════════ */

static void build_hoppair_gate(double dtau, double t, double delta, double *G_re, double *G_im)
{
    int D = 6, D2 = D * D;
    if (G_re) memset(G_re, 0, D2 * D2 * sizeof(double));
    if (G_im) memset(G_im, 0, D2 * D2 * sizeof(double));
    for (int i=0; i<36; i++) G_re[i*D2+i] = 1.0;

    // --- 1. Hopping Subspace ---
    double H_hop[16][16] = {{0}};
    H_hop[1*4+0][0*4+1] = -t; H_hop[0*4+1][1*4+0] = -t;
    H_hop[3*4+0][2*4+1] = +t; H_hop[2*4+1][3*4+0] = +t;
    H_hop[1*4+2][0*4+3] = +t; H_hop[0*4+3][1*4+2] = +t;
    H_hop[3*4+2][2*4+3] = -t; H_hop[2*4+3][3*4+2] = -t;

    H_hop[2*4+0][0*4+2] = -t; H_hop[0*4+2][2*4+0] = -t;
    H_hop[3*4+0][1*4+2] = +t; H_hop[1*4+2][3*4+0] = +t;
    H_hop[2*4+1][0*4+3] = +t; H_hop[0*4+3][2*4+1] = +t;
    H_hop[3*4+1][1*4+3] = -t; H_hop[1*4+3][3*4+1] = -t;

    double ch_hop = cosh(dtau * t);
    double sh_hop = sinh(dtau * t);

    for(int i=0; i<16; i++) {
        for(int j=i+1; j<16; j++) {
            if (H_hop[i][j] != 0) {
                int idxi = (i / 4) * D + (i % 4);
                int idxj = (j / 4) * D + (j % 4);
                G_re[idxi * D2 + idxi] = ch_hop;
                G_re[idxj * D2 + idxj] = ch_hop;
                double cross = - (H_hop[i][j] / t) * sh_hop;
                G_re[idxi * D2 + idxj] = cross;
                G_re[idxj * D2 + idxi] = cross;
            }
        }
    }

    // --- 2. Pairing Subspace (d-wave pairs empty/double) ---
    if (fabs(delta) > 1e-10) {
        int i0 = 0*D+0;      // |0,0>
        int i1 = 1*D+2;      // |Up,Down>
        int i2 = 2*D+1;      // |Down,Up>

        double a = sqrt(2.0) * fabs(delta);
        double term1 = - sinh(a * dtau) / a;
        double term2 = (cosh(a * dtau) - 1.0) / (a * a);
        double d2 = delta * delta;

        // Diagonal modifications
        G_re[i0*D2 + i0] = 1.0 + term2 * (2 * d2);
        G_re[i1*D2 + i1] = 1.0 + term2 * (d2);
        G_re[i2*D2 + i2] = 1.0 + term2 * (d2);
        
        G_re[i1*D2 + i2] += term2 * (-d2);
        G_re[i2*D2 + i1] += term2 * (-d2);

        // Off-diagonal mixing (injecting pairs)
        double H_01 = -delta;
        double H_02 = +delta;
        
        G_re[i0*D2 + i1] += term1 * H_01;
        G_re[i1*D2 + i0] += term1 * H_01;
        
        G_re[i0*D2 + i2] += term1 * H_02;
        G_re[i2*D2 + i0] += term1 * H_02;
    }
}

/* ═══════════════ Trotter Loop ═══════════════ */

static void apply_dwave_trotter_step(Tns3dGrid *g, double *hop_re_bulk, double *hop_im_bulk, double *hop_re_x_bdry, double *hop_im_x_bdry, double *hop_re_y_bdry, double *hop_im_y_bdry)
{
    // Apply X gates
    for (int parity = 0; parity < 2; parity++) {
        for (int y = 0; y < g->Ly; y++) {
            for (int x = parity; x < g->Lx - 1; x += 2) {
                if (y == 0 || y == g->Ly - 1 || x == 0 || x == g->Lx - 2) { // Boundary proximity
                    tns3d_gate_x(g, x, y, 0, hop_re_x_bdry, hop_im_x_bdry);
                } else {
                    tns3d_gate_x(g, x, y, 0, hop_re_bulk, hop_im_bulk);
                }
            }
        }
    }
    // Apply Y gates
    for (int parity = 0; parity < 2; parity++) {
        for (int x = 0; x < g->Lx; x++) {
            for (int y = parity; y < g->Ly - 1; y += 2) {
                if (x == 0 || x == g->Lx - 1 || y == 0 || y == g->Ly - 2) { // Boundary proximity
                    tns3d_gate_y(g, x, y, 0, hop_re_y_bdry, hop_im_y_bdry);
                } else {
                    tns3d_gate_y(g, x, y, 0, hop_re_bulk, hop_im_bulk);
                }
            }
        }
    }
}

static void renormalize_all(Tns3dGrid *g)
{
    for (int i = 0; i < g->Lx * g->Ly * g->Lz; i++) {
        int reg = g->site_reg[i];
        if (reg < 0) continue;
        QuhitRegister *r = &g->eng->registers[reg];
        double n2 = 0;
        for (uint32_t e = 0; e < r->num_nonzero; e++)
            n2 += r->entries[e].amp_re * r->entries[e].amp_re + r->entries[e].amp_im * r->entries[e].amp_im;
        if (n2 > 1e-20) {
            double inv = 1.0 / sqrt(n2);
            for (uint32_t e = 0; e < r->num_nonzero; e++) {
                r->entries[e].amp_re *= inv;
                r->entries[e].amp_im *= inv;
            }
        }
    }
}

static void print_pairing_map(Tns3dGrid *g)
{
    printf("\n  ═══ COOPER PAIR PENETRATION MAP (Empty/Double Fraction) ═══\n");
    for (int y = g->Ly - 1; y >= 0; y--) {
        printf("  y=%d |", y);
        for (int x = 0; x < g->Lx; x++) {
            double p[6]; tns3d_local_density(g, x, y, 0, p);
            // The likelihood of a site participating in a Cooper Pair is proportional 
            // to its Empty + Doubly Occupied basis fraction since pairs = (|00> + |up,dn>).
            double pair_weight = p[0] + p[3]; 
            
            if (pair_weight > 0.8) printf(" ██ ");
            else if (pair_weight > 0.6) printf(" ▓▓ ");
            else if (pair_weight > 0.45) printf(" ▒▒ ");
            else if (pair_weight > 0.35) printf(" ░░ ");
            else printf(" .. ");
        }
        printf("|\n");
    }
    printf("         (██/▓▓ = Boundary Induced Pairs | ▒▒/░░ = Bulk Superconducting Penetration | .. = Mott Insulator)\n");
}

/* ═══════════════ Main ═══════════════ */

int main(void)
{
    int Lx = 8, Ly = 8;
    int Nsites = Lx * Ly;

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  d-WAVE SUPERCONDUCTING PROBE — HexState PEPS                ║\n");
    printf("║  ──────────────────────────────────────────────────────────  ║\n");
    printf("║  Lattice: %d×%d                                               ║\n", Lx, Ly);
    printf("║  Model: Hole-Doped Hubbard with d-Wave Boundary Fields       ║\n");
    printf("║  Fields: Δ_x = +%.1f, Δ_y = -%.1f                            ║\n", DWAVE_DELTA, DWAVE_DELTA);
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    double *onsite_re = calloc(36*36, sizeof(double));
    double *onsite_im = calloc(36*36, sizeof(double));
    build_onsite_gate(COOL_DTAU, HUBBARD_U, HUBBARD_MU, onsite_re, onsite_im);

    double *hop_re_bulk = calloc(36*36, sizeof(double));
    double *hop_im_bulk = calloc(36*36, sizeof(double));
    build_hoppair_gate(COOL_DTAU, HUBBARD_T, 0.0, hop_re_bulk, hop_im_bulk);

    double *hop_re_x_bdry = calloc(36*36, sizeof(double));
    double *hop_im_x_bdry = calloc(36*36, sizeof(double));
    // d-Wave implies opposite sign for X and Y bonds
    build_hoppair_gate(COOL_DTAU, HUBBARD_T, +DWAVE_DELTA, hop_re_x_bdry, hop_im_x_bdry);

    double *hop_re_y_bdry = calloc(36*36, sizeof(double));
    double *hop_im_y_bdry = calloc(36*36, sizeof(double));
    build_hoppair_gate(COOL_DTAU, HUBBARD_T, -DWAVE_DELTA, hop_re_y_bdry, hop_im_y_bdry);

    Tns3dGrid *g = tns3d_init(Lx, Ly, 1);

    for (int i = 0; i < Nsites; i++) {
        int reg = g->site_reg[i];
        double norm = 1.0 / sqrt(2.0);
        int spin = (i % 2 == 0) ? 1 : 2; 
        quhit_reg_sv_set(g->eng, reg, 0, norm, 0);       
        quhit_reg_sv_set(g->eng, reg, spin*TNS3D_C6, norm, 0); 
    }

    printf("  ══ COOLING INTO SUPERCONDUCTING STATE (%d steps) ══\n\n", COOL_STEPS);
    double total_time = 0;

    for (int step = 1; step <= COOL_STEPS; step++) {
        clock_t t0 = clock();

        tns3d_gate_1site_all(g, onsite_re, onsite_im);
        renormalize_all(g);

        apply_dwave_trotter_step(g, hop_re_bulk, hop_im_bulk, hop_re_x_bdry, hop_im_x_bdry, hop_re_y_bdry, hop_im_y_bdry);
        renormalize_all(g);

        double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
        total_time += dt;

        printf("  Step %2d | Time: %5.2fs \n", step, dt);
    }
    
    printf("\n  ════════════════════════════════════════════════════════════\n");
    print_pairing_map(g);

    tns3d_free(g);
    free(onsite_re); free(onsite_im);
    free(hop_re_bulk); free(hop_im_bulk);
    free(hop_re_x_bdry); free(hop_im_x_bdry);
    free(hop_re_y_bdry); free(hop_im_y_bdry);
    return 0;
}
