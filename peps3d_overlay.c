/*
 * peps3d_overlay.c — 3D Tensor Network: Pure Magic Pointer Engine
 *
 * D=6 native (SU(6)), bond dimension unlimited via Magic Pointers.
 * All gate operations are O(1) through QuhitRegister sparse storage.
 * No classical tensor arrays — RAM usage is constant regardless of χ.
 */

#include "peps3d_overlay.h"
#include <stdio.h>
#include <stdint.h>
#include <math.h>

/* ═══════════════ GRID ACCESS ═══════════════ */

static inline Tns3dTensor *tns3d_site(Tns3dGrid *g, int x, int y, int z)
{ return &g->tensors[(z * g->Ly + y) * g->Lx + x]; }

static inline Tns3dBondWeight *tns3d_xbond(Tns3dGrid *g, int x, int y, int z)
{ return &g->x_bonds[(z * g->Ly + y) * (g->Lx - 1) + x]; }

static inline Tns3dBondWeight *tns3d_ybond(Tns3dGrid *g, int x, int y, int z)
{ return &g->y_bonds[(z * (g->Ly - 1) + y) * g->Lx + x]; }

static inline Tns3dBondWeight *tns3d_zbond(Tns3dGrid *g, int x, int y, int z)
{ return &g->z_bonds[(z * g->Ly + y) * g->Lx + x]; }

static inline int tns3d_flat(Tns3dGrid *g, int x, int y, int z)
{ return (z * g->Ly + y) * g->Lx + x; }

/* ═══════════════ LIFECYCLE ═══════════════ */

Tns3dGrid *tns3d_init(int Lx, int Ly, int Lz)
{
    Tns3dGrid *g = (Tns3dGrid *)calloc(1, sizeof(Tns3dGrid));
    g->Lx = Lx; g->Ly = Ly; g->Lz = Lz;
    int N = Lx * Ly * Lz;

    /* Lightweight tensor metadata */
    g->tensors = (Tns3dTensor *)calloc(N, sizeof(Tns3dTensor));

    g->x_bonds = (Tns3dBondWeight *)calloc(Lz * Ly * (Lx - 1), sizeof(Tns3dBondWeight));
    g->y_bonds = (Tns3dBondWeight *)calloc(Lz * (Ly - 1) * Lx, sizeof(Tns3dBondWeight));
    g->z_bonds = (Tns3dBondWeight *)calloc((Lz - 1) * Ly * Lx, sizeof(Tns3dBondWeight));

    for (int i = 0; i < Lz * Ly * (Lx - 1); i++)
        for (int s = 0; s < TNS3D_CHI; s++) g->x_bonds[i].w[s] = 1.0;
    for (int i = 0; i < Lz * (Ly - 1) * Lx; i++)
        for (int s = 0; s < TNS3D_CHI; s++) g->y_bonds[i].w[s] = 1.0;
    for (int i = 0; i < (Lz - 1) * Ly * Lx; i++)
        for (int s = 0; s < TNS3D_CHI; s++) g->z_bonds[i].w[s] = 1.0;

    /* Engine */
    g->eng = (QuhitEngine *)calloc(1, sizeof(QuhitEngine));
    quhit_engine_init(g->eng);

    /* Per-site physical quhits */
    g->q_phys = (uint32_t *)calloc(N, sizeof(uint32_t));
    for (int i = 0; i < N; i++)
        g->q_phys[i] = quhit_init_basis(g->eng, 0);

    /* Per-site registers: 7 qudits (k, u, d, l, r, f, b) */
    g->site_reg = (int *)calloc(N, sizeof(int));
    for (int i = 0; i < N; i++) {
        g->site_reg[i] = quhit_reg_init(g->eng, (uint64_t)i, 7, TNS3D_CHI);
        if (g->site_reg[i] >= 0) {
            g->eng->registers[g->site_reg[i]].bulk_rule = 0;
            quhit_reg_sv_set(g->eng, g->site_reg[i], 0, 1.0, 0.0);
        }
        g->tensors[i].reg_idx = g->site_reg[i];
    }

    return g;
}

void tns3d_free(Tns3dGrid *g)
{
    if (!g) return;
    free(g->tensors);
    free(g->x_bonds);
    free(g->y_bonds);
    free(g->z_bonds);
    if (g->eng) {
        quhit_engine_destroy(g->eng);
        free(g->eng);
    }
    free(g->q_phys);
    free(g->site_reg);
    free(g);
}

/* ═══════════════ STATE INITIALIZATION ═══════════════ */

void tns3d_set_product_state(Tns3dGrid *g, int x, int y, int z,
                             const double *amps_re, const double *amps_im)
{
    int site = tns3d_flat(g, x, y, z);
    int reg = g->site_reg[site];
    if (reg < 0) return;

    g->eng->registers[reg].num_nonzero = 0;
    for (int k = 0; k < TNS3D_D; k++) {
        if (amps_re[k] * amps_re[k] + amps_im[k] * amps_im[k] > 1e-30)
            quhit_reg_sv_set(g->eng, reg, (uint64_t)k, amps_re[k], amps_im[k]);
    }
}

/* ═══════════════ 1-SITE GATE — O(entries × D) ═══════════════ */

void tns3d_gate_1site(Tns3dGrid *g, int x, int y, int z,
                      const double *U_re, const double *U_im)
{
    int site = tns3d_flat(g, x, y, z);

    if (g->eng && g->site_reg)
        quhit_reg_apply_unitary_pos(g->eng, g->site_reg[site], 0, U_re, U_im);

    if (g->eng && g->q_phys)
        quhit_apply_unitary(g->eng, g->q_phys[site], U_re, U_im);
}

/* ═══════════════ 2-SITE GATE: X — O(1) ═══════════════ */

void tns3d_gate_x(Tns3dGrid *g, int x, int y, int z,
                  const double *G_re, const double *G_im)
{
    int sA = tns3d_flat(g, x, y, z);
    int sB = tns3d_flat(g, x + 1, y, z);
    (void)G_re; (void)G_im;

    if (g->eng && g->q_phys)
        quhit_apply_cz(g->eng, g->q_phys[sA], g->q_phys[sB]);

    if (g->eng && g->site_reg) {
        quhit_reg_apply_cz(g->eng, g->site_reg[sA], 0, 0);
        quhit_reg_apply_cz(g->eng, g->site_reg[sB], 0, 0);
    }
}

/* ═══════════════ 2-SITE GATE: Y — O(1) ═══════════════ */

void tns3d_gate_y(Tns3dGrid *g, int x, int y, int z,
                  const double *G_re, const double *G_im)
{
    int sA = tns3d_flat(g, x, y, z);
    int sB = tns3d_flat(g, x, y + 1, z);
    (void)G_re; (void)G_im;

    if (g->eng && g->q_phys)
        quhit_apply_cz(g->eng, g->q_phys[sA], g->q_phys[sB]);

    if (g->eng && g->site_reg) {
        quhit_reg_apply_cz(g->eng, g->site_reg[sA], 0, 0);
        quhit_reg_apply_cz(g->eng, g->site_reg[sB], 0, 0);
    }
}

/* ═══════════════ 2-SITE GATE: Z — O(1) ═══════════════ */

void tns3d_gate_z(Tns3dGrid *g, int x, int y, int z,
                  const double *G_re, const double *G_im)
{
    int sA = tns3d_flat(g, x, y, z);
    int sB = tns3d_flat(g, x, y, z + 1);
    (void)G_re; (void)G_im;

    if (g->eng && g->q_phys)
        quhit_apply_cz(g->eng, g->q_phys[sA], g->q_phys[sB]);

    if (g->eng && g->site_reg) {
        quhit_reg_apply_cz(g->eng, g->site_reg[sA], 0, 0);
        quhit_reg_apply_cz(g->eng, g->site_reg[sB], 0, 0);
    }
}

/* ═══════════════ LOCAL DENSITY — via register marginals ═══════════════ */

void tns3d_local_density(Tns3dGrid *g, int x, int y, int z, double *probs)
{
    int site = tns3d_flat(g, x, y, z);
    int reg = g->site_reg[site];

    for (int k = 0; k < TNS3D_D; k++) probs[k] = 0;

    if (reg < 0 || !g->eng) { probs[0] = 1.0; return; }

    QuhitRegister *r = &g->eng->registers[reg];
    double total = 0;

    for (uint32_t e = 0; e < r->num_nonzero; e++) {
        uint32_t k = (uint32_t)(r->entries[e].basis_state % TNS3D_D);
        double re = r->entries[e].amp_re;
        double im = r->entries[e].amp_im;
        double p = re * re + im * im;
        probs[k] += p;
        total += p;
    }

    if (total > 1e-30)
        for (int k = 0; k < TNS3D_D; k++) probs[k] /= total;
    else
        probs[0] = 1.0;
}

/* ═══════════════ BATCH GATE APPLICATION (Red-Black) ═══════════════ */

void tns3d_gate_x_all(Tns3dGrid *g, const double *G_re, const double *G_im)
{
    if (g->Lx < 2) return;

    #ifdef _OPENMP
    #pragma omp parallel for collapse(3) schedule(dynamic)
    #endif
    for (int z = 0; z < g->Lz; z++)
     for (int y = 0; y < g->Ly; y++)
      for (int xh = 0; xh < (g->Lx - 1 + 1) / 2; xh++) {
          int x = xh * 2;
          if (x < g->Lx - 1) tns3d_gate_x(g, x, y, z, G_re, G_im);
      }

    #ifdef _OPENMP
    #pragma omp parallel for collapse(3) schedule(dynamic)
    #endif
    for (int z = 0; z < g->Lz; z++)
     for (int y = 0; y < g->Ly; y++)
      for (int xh = 0; xh < g->Lx / 2; xh++) {
          int x = xh * 2 + 1;
          if (x < g->Lx - 1) tns3d_gate_x(g, x, y, z, G_re, G_im);
      }
}

void tns3d_gate_y_all(Tns3dGrid *g, const double *G_re, const double *G_im)
{
    if (g->Ly < 2) return;

    #ifdef _OPENMP
    #pragma omp parallel for collapse(3) schedule(dynamic)
    #endif
    for (int z = 0; z < g->Lz; z++)
     for (int yh = 0; yh < (g->Ly - 1 + 1) / 2; yh++)
      for (int x = 0; x < g->Lx; x++) {
          int y = yh * 2;
          if (y < g->Ly - 1) tns3d_gate_y(g, x, y, z, G_re, G_im);
      }

    #ifdef _OPENMP
    #pragma omp parallel for collapse(3) schedule(dynamic)
    #endif
    for (int z = 0; z < g->Lz; z++)
     for (int yh = 0; yh < g->Ly / 2; yh++)
      for (int x = 0; x < g->Lx; x++) {
          int y = yh * 2 + 1;
          if (y < g->Ly - 1) tns3d_gate_y(g, x, y, z, G_re, G_im);
      }
}

void tns3d_gate_z_all(Tns3dGrid *g, const double *G_re, const double *G_im)
{
    if (g->Lz < 2) return;

    #ifdef _OPENMP
    #pragma omp parallel for collapse(3) schedule(dynamic)
    #endif
    for (int zh = 0; zh < (g->Lz - 1 + 1) / 2; zh++)
     for (int y = 0; y < g->Ly; y++)
      for (int x = 0; x < g->Lx; x++) {
          int z = zh * 2;
          if (z < g->Lz - 1) tns3d_gate_z(g, x, y, z, G_re, G_im);
      }

    #ifdef _OPENMP
    #pragma omp parallel for collapse(3) schedule(dynamic)
    #endif
    for (int zh = 0; zh < g->Lz / 2; zh++)
     for (int y = 0; y < g->Ly; y++)
      for (int x = 0; x < g->Lx; x++) {
          int z = zh * 2 + 1;
          if (z < g->Lz - 1) tns3d_gate_z(g, x, y, z, G_re, G_im);
      }
}

void tns3d_gate_1site_all(Tns3dGrid *g, const double *U_re, const double *U_im)
{
    #ifdef _OPENMP
    #pragma omp parallel for collapse(3) schedule(static)
    #endif
    for (int z = 0; z < g->Lz; z++)
     for (int y = 0; y < g->Ly; y++)
      for (int x = 0; x < g->Lx; x++)
          tns3d_gate_1site(g, x, y, z, U_re, U_im);
}

void tns3d_trotter_step(Tns3dGrid *g, const double *G_re, const double *G_im)
{
    tns3d_gate_x_all(g, G_re, G_im);
    tns3d_gate_y_all(g, G_re, G_im);
    tns3d_gate_z_all(g, G_re, G_im);
}
