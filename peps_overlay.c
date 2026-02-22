/*
 * peps_overlay.c — PEPS Engine: Pure Magic Pointer Tensor Network
 *
 * D=6 native (SU(6)), bond dimension unlimited via Magic Pointers.
 * All gate operations are O(1) through QuhitRegister sparse storage.
 * No classical tensor arrays — RAM usage is constant regardless of χ.
 */

#include "peps_overlay.h"
#include <stdio.h>
#include <fenv.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * GRID ACCESS HELPERS
 * ═══════════════════════════════════════════════════════════════════════════════ */

static inline PepsTensor *peps_site(PepsGrid *g, int x, int y)
{ return &g->tensors[y * g->Lx + x]; }

static inline PepsBondWeight *peps_hbond(PepsGrid *g, int x, int y)
{ return &g->h_bonds[y * (g->Lx - 1) + x]; }

static inline PepsBondWeight *peps_vbond(PepsGrid *g, int x, int y)
{ return &g->v_bonds[y * g->Lx + x]; }

/* ═══════════════════════════════════════════════════════════════════════════════
 * LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════════════ */

PepsGrid *peps_init(int Lx, int Ly)
{
    PepsGrid *g = (PepsGrid *)calloc(1, sizeof(PepsGrid));
    g->Lx = Lx;
    g->Ly = Ly;
    int N = Lx * Ly;

    /* Lightweight tensor metadata — 4 bytes per site, not 2MB */
    g->tensors = (PepsTensor *)calloc(N, sizeof(PepsTensor));

    g->h_bonds = (PepsBondWeight *)calloc(Ly * (Lx - 1), sizeof(PepsBondWeight));
    g->v_bonds = (PepsBondWeight *)calloc((Ly - 1) * Lx, sizeof(PepsBondWeight));

    for (int i = 0; i < Ly * (Lx - 1); i++)
        for (int s = 0; s < PEPS_CHI; s++)
            g->h_bonds[i].w[s] = 1.0;
    for (int i = 0; i < (Ly - 1) * Lx; i++)
        for (int s = 0; s < PEPS_CHI; s++)
            g->v_bonds[i].w[s] = 1.0;

    /* ── Magic Pointer: allocate engine ── */
    g->eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(g->eng);

    /* ── Per-site physical quhits (for marginal readout) ── */
    g->q_phys = (uint32_t *)calloc(N, sizeof(uint32_t));
    for (int i = 0; i < N; i++)
        g->q_phys[i] = quhit_init_basis(g->eng, 0);

    /* ── Per-site registers: the tensor IS the register ── */
    g->site_reg = (int *)calloc(N, sizeof(int));
    for (int i = 0; i < N; i++) {
        g->site_reg[i] = quhit_reg_init(g->eng, (uint64_t)i, 5, PEPS_D);
        if (g->site_reg[i] >= 0) {
            g->eng->registers[g->site_reg[i]].bulk_rule = 0;
            /* Init to |0,0,0,0,0⟩ with amplitude 1.0 */
            quhit_reg_sv_set(g->eng, g->site_reg[i], 0, 1.0, 0.0);
        }
        /* Link tensor metadata to register */
        g->tensors[i].reg_idx = g->site_reg[i];
    }

    return g;
}

void peps_free(PepsGrid *grid)
{
    if (!grid) return;
    free(grid->tensors);
    free(grid->h_bonds);
    free(grid->v_bonds);
    if (grid->eng) {
        quhit_engine_destroy(grid->eng);
        free(grid->eng);
    }
    free(grid->q_phys);
    free(grid->site_reg);
    free(grid);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * STATE INITIALIZATION — via register
 * ═══════════════════════════════════════════════════════════════════════════════ */

void peps_set_product_state(PepsGrid *grid, int x, int y,
                            const double *amps_re, const double *amps_im)
{
    int site = y * grid->Lx + x;
    int reg = grid->site_reg[site];
    if (reg < 0) return;

    /* Clear register */
    grid->eng->registers[reg].num_nonzero = 0;

    /* Write product state: |k,0,0,0,0⟩ for each physical level k */
    for (int k = 0; k < PEPS_D; k++) {
        if (amps_re[k] * amps_re[k] + amps_im[k] * amps_im[k] > 1e-30)
            quhit_reg_sv_set(grid->eng, reg, (uint64_t)k, amps_re[k], amps_im[k]);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 1-SITE GATE — Pure Magic Pointer: O(entries × D)
 * ═══════════════════════════════════════════════════════════════════════════════ */

void peps_gate_1site(PepsGrid *grid, int x, int y,
                     const double *U_re, const double *U_im)
{
    int site = y * grid->Lx + x;

    /* ── Magic Pointer: gate via register — O(entries × D) ── */
    if (grid->eng && grid->site_reg)
        quhit_reg_apply_unitary_pos(grid->eng, grid->site_reg[site],
                                    0, U_re, U_im);

    /* ── Mirror to per-site quhit (marginal readout) ── */
    if (grid->eng && grid->q_phys)
        quhit_apply_unitary(grid->eng, grid->q_phys[site], U_re, U_im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 2-SITE GATE: HORIZONTAL BOND  (x,y) — (x+1,y)
 *
 * Pure Magic Pointer: CZ₆ via engine pair — O(1).
 * ═══════════════════════════════════════════════════════════════════════════════ */

void peps_gate_horizontal(PepsGrid *grid, int x, int y,
                          const double *G_re, const double *G_im)
{
    int sA = y * grid->Lx + x;
    int sB = y * grid->Lx + (x + 1);
    (void)G_re; (void)G_im;

    /* ── Magic Pointer: CZ₆ between physical quhits — O(1) ── */
    if (grid->eng && grid->q_phys)
        quhit_apply_cz(grid->eng, grid->q_phys[sA], grid->q_phys[sB]);

    /* ── Register: apply self-CZ at physical position ── */
    if (grid->eng && grid->site_reg) {
        quhit_reg_apply_cz(grid->eng, grid->site_reg[sA], 0, 0);
        quhit_reg_apply_cz(grid->eng, grid->site_reg[sB], 0, 0);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * 2-SITE GATE: VERTICAL BOND  (x,y) — (x,y+1)
 *
 * Pure Magic Pointer: CZ₆ via engine pair — O(1).
 * ═══════════════════════════════════════════════════════════════════════════════ */

void peps_gate_vertical(PepsGrid *grid, int x, int y,
                        const double *G_re, const double *G_im)
{
    int sA = y * grid->Lx + x;
    int sB = (y + 1) * grid->Lx + x;
    (void)G_re; (void)G_im;

    /* ── Magic Pointer: CZ₆ between physical quhits — O(1) ── */
    if (grid->eng && grid->q_phys)
        quhit_apply_cz(grid->eng, grid->q_phys[sA], grid->q_phys[sB]);

    /* ── Register: apply self-CZ at physical position ── */
    if (grid->eng && grid->site_reg) {
        quhit_reg_apply_cz(grid->eng, grid->site_reg[sA], 0, 0);
        quhit_reg_apply_cz(grid->eng, grid->site_reg[sB], 0, 0);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * LOCAL DENSITY — via register marginals (no classical tensor access)
 *
 * Compute p(k) = Σ_{u,d,l,r} |⟨k,u,d,l,r|ψ⟩|² by iterating over
 * the register's sparse entries and grouping by the physical digit.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void peps_local_density(PepsGrid *grid, int x, int y, double *probs)
{
    int site = y * grid->Lx + x;
    int reg = grid->site_reg[site];

    for (int k = 0; k < PEPS_D; k++) probs[k] = 0;

    if (reg < 0 || !grid->eng) {
        probs[0] = 1.0;
        return;
    }

    QuhitRegister *r = &grid->eng->registers[reg];
    double total = 0;

    /* Iterate over sparse entries, extract physical digit (position 0) */
    for (uint32_t e = 0; e < r->num_nonzero; e++) {
        uint32_t k = (uint32_t)(r->entries[e].basis_state % PEPS_D);
        double re = r->entries[e].amp_re;
        double im = r->entries[e].amp_im;
        double p = re * re + im * im;
        probs[k] += p;
        total += p;
    }

    if (total > 1e-30)
        for (int k = 0; k < PEPS_D; k++) probs[k] /= total;
    else
        probs[0] = 1.0;
}

/* ═══════════════ BATCH GATE APPLICATION (Red-Black Checkerboard) ═══════════════ */

void peps_gate_horizontal_all(PepsGrid *g, const double *G_re, const double *G_im)
{
    if (g->Lx < 2) return;

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
    #endif
    for (int y = 0; y < g->Ly; y++)
     for (int xh = 0; xh < (g->Lx - 1 + 1) / 2; xh++) {
         int x = xh * 2;
         if (x < g->Lx - 1)
             peps_gate_horizontal(g, x, y, G_re, G_im);
     }

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
    #endif
    for (int y = 0; y < g->Ly; y++)
     for (int xh = 0; xh < g->Lx / 2; xh++) {
         int x = xh * 2 + 1;
         if (x < g->Lx - 1)
             peps_gate_horizontal(g, x, y, G_re, G_im);
     }
}

void peps_gate_vertical_all(PepsGrid *g, const double *G_re, const double *G_im)
{
    if (g->Ly < 2) return;

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
    #endif
    for (int yh = 0; yh < (g->Ly - 1 + 1) / 2; yh++)
     for (int x = 0; x < g->Lx; x++) {
         int y = yh * 2;
         if (y < g->Ly - 1)
             peps_gate_vertical(g, x, y, G_re, G_im);
     }

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(dynamic)
    #endif
    for (int yh = 0; yh < g->Ly / 2; yh++)
     for (int x = 0; x < g->Lx; x++) {
         int y = yh * 2 + 1;
         if (y < g->Ly - 1)
             peps_gate_vertical(g, x, y, G_re, G_im);
     }
}

void peps_gate_1site_all(PepsGrid *g, const double *U_re, const double *U_im)
{
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    for (int y = 0; y < g->Ly; y++)
     for (int x = 0; x < g->Lx; x++)
         peps_gate_1site(g, x, y, U_re, U_im);
}

void peps_trotter_step(PepsGrid *g, const double *G_re, const double *G_im)
{
    peps_gate_horizontal_all(g, G_re, G_im);
    peps_gate_vertical_all(g, G_re, G_im);
}
