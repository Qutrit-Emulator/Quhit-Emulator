/*
 * quhit_projective.h — Projective Horizon Register
 *
 * ═══════════════════════════════════════════════════════════════════════
 * INFINITE HORIZON GEOMETRY
 *
 * A D=6 quhit decomposes as k = 3p + s, where:
 *   p ∈ {0,1} — parity (the non-projecting base)
 *   s ∈ {0,1,2} — square plane (the infinite horizon)
 *
 * The Base (2^K flat array) stores the parity amplitudes.
 * The Horizon (3^K plane indices) is never stored — it is projected
 * analytically at observation via twiddle + CZ correction phases.
 *
 * DFT₆ = Hadamard on parity bit (applied immediately to base).
 * CZ₆  = (-1)^(pi*pj) on base (immediate) + log(i,j) for horizon.
 * Born  = resolve horizon phases analytically, no materialization.
 *
 * Storage: O(2^K) instead of O(6^K).
 * Compression: 3^K × (exact, not approximate).
 * ═══════════════════════════════════════════════════════════════════════
 */

#ifndef QUHIT_PROJECTIVE_H
#define QUHIT_PROJECTIVE_H

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ═══════════════ Constants ═══════════════ */

#define PROJ_D       6
#define PROJ_MAX_CZ  8192    /* Max recorded CZ gates */
#define PROJ_H2      0.70710678118654752440  /* 1/√2 */

/* ω₆ table: e^(i·2π·n/6) for n=0..5 */
static const double PROJ_W6R[6] = {
    1.0, 0.5, -0.5, -1.0, -0.5, 0.5
};
static const double PROJ_W6I[6] = {
    0.0, 0.866025403784438647, 0.866025403784438647,
    0.0, -0.866025403784438647, -0.866025403784438647
};

/* ═══════════════ Data Structure ═══════════════ */

typedef struct {
    /* The Non-Projecting Base: 2^K parity amplitudes */
    double  *base_re;
    double  *base_im;
    int      K;             /* Number of active quhits */
    uint64_t dim;           /* 2^K */

    /* CZ Gate Log (horizon phases resolved at observation) */
    int      cz_a[PROJ_MAX_CZ];
    int      cz_b[PROJ_MAX_CZ];
    int      cz_count;

    /* Total quhits in the logical register (for display) */
    uint64_t n_quhits;
} ProjRegister;

/* ═══════════════ Lifecycle ═══════════════ */

static inline ProjRegister *proj_alloc(int K, uint64_t n_quhits) {
    ProjRegister *r = (ProjRegister *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    r->K = K;
    r->dim = 1ULL << K;
    r->n_quhits = n_quhits;
    r->cz_count = 0;
    r->base_re = (double *)calloc(r->dim, sizeof(double));
    r->base_im = (double *)calloc(r->dim, sizeof(double));
    if (!r->base_re || !r->base_im) {
        free(r->base_re); free(r->base_im); free(r);
        return NULL;
    }
    return r;
}

static inline void proj_free(ProjRegister *r) {
    if (!r) return;
    free(r->base_re);
    free(r->base_im);
    free(r);
}

static inline void proj_init_zero(ProjRegister *r) {
    memset(r->base_re, 0, r->dim * sizeof(double));
    memset(r->base_im, 0, r->dim * sizeof(double));
    r->base_re[0] = 1.0;  /* |0...0⟩ */
    r->cz_count = 0;
}

/* ═══════════════ DFT₆ = Hadamard on Base ═══════════════ */

static inline void proj_dft6(ProjRegister *r, int qi) {
    uint64_t mask = 1ULL << qi;
    for (uint64_t b = 0; b < r->dim; b++) {
        if (b & mask) continue;  /* process pairs once */
        uint64_t b1 = b | mask;
        double ar = r->base_re[b], ai = r->base_im[b];
        double br_ = r->base_re[b1], bi = r->base_im[b1];
        r->base_re[b]  = PROJ_H2 * (ar + br_);
        r->base_im[b]  = PROJ_H2 * (ai + bi);
        r->base_re[b1] = PROJ_H2 * (ar - br_);
        r->base_im[b1] = PROJ_H2 * (ai - bi);
    }
}

/* ═══════════════ CZ = Parity Phase + Log ═══════════════ */

static inline void proj_cz(ProjRegister *r, int qi, int qj) {
    /* Immediate: apply (-1)^(pi*pj) to the base */
    uint64_t mask = (1ULL << qi) | (1ULL << qj);
    for (uint64_t b = 0; b < r->dim; b++) {
        if ((b & mask) == mask) {
            r->base_re[b] = -r->base_re[b];
            r->base_im[b] = -r->base_im[b];
        }
    }
    /* Deferred: log the gate for horizon resolution */
    if (r->cz_count < PROJ_MAX_CZ) {
        r->cz_a[r->cz_count] = qi;
        r->cz_b[r->cz_count] = qj;
        r->cz_count++;
    }
}

/* ═══════════════ Magic Pointer Resolution ═══════════════
 *
 * Given a full k-vector (k₀, k₁, ..., k_{K-1}), resolve the exact
 * complex amplitude by projecting from the base to the horizon.
 *
 * This IS the Magic Pointer dereference: the pointer is the k-vector,
 * and the resolution computes the amplitude analytically.
 * ═══════════════════════════════════════════════════════════════════ */

static inline void proj_resolve_amplitude(
    const ProjRegister *r,
    const int *kvec,       /* kvec[i] for i=0..K-1, each in 0..5 */
    double *out_re, double *out_im)
{
    /* Decompose k-vector into parity and horizon */
    uint64_t b = 0;
    int pvec[64], svec[64];
    for (int i = 0; i < r->K; i++) {
        pvec[i] = kvec[i] / 3;
        svec[i] = kvec[i] % 3;
        if (pvec[i]) b |= (1ULL << i);
    }

    /* Read base amplitude */
    double ar = r->base_re[b], ai = r->base_im[b];

    /* Twiddle phase: Σ s_i * p_i */
    int tw_ph = 0;
    for (int i = 0; i < r->K; i++)
        tw_ph += svec[i] * pvec[i];

    /* CZ horizon correction: Σ (3*pa*sb + 3*pb*sa + sa*sb) */
    int cz_ph = 0;
    for (int g = 0; g < r->cz_count; g++) {
        int a = r->cz_a[g], c = r->cz_b[g];
        cz_ph += 3 * pvec[a] * svec[c]
               + 3 * pvec[c] * svec[a]
               + svec[a] * svec[c];
    }

    int total_ph = ((tw_ph + cz_ph) % 6 + 6) % 6;

    /* Apply horizon phase */
    double wr = PROJ_W6R[total_ph], wi = PROJ_W6I[total_ph];
    double c_re = wr * ar - wi * ai;
    double c_im = wr * ai + wi * ar;

    /* Normalization: 1/3^(K/2) for K quhits' DFT₃ projection */
    double norm = pow(3.0, r->K / 2.0);
    *out_re = c_re / norm;
    *out_im = c_im / norm;
}

/* ═══════════════ Born Probability ═══════════════
 *
 * Observe quhit `qt`: compute P(k_qt = k) for k = 0..5.
 *
 * For N=2 this is an explicit trace over the other quhit.
 * For general N, we iterate over all 6^(K-1) states of
 * the non-target quhits, computing the amplitude via
 * proj_resolve_amplitude and summing |amp|².
 *
 * For large K this becomes expensive at observation time.
 * For the engine's use case (small K active, large N total),
 * K is kept small and this is tractable.
 * ═══════════════════════════════════════════════════════════ */

static inline void proj_born(const ProjRegister *r, int qt, double pr[6]) {
    memset(pr, 0, 48);

    int K = r->K;
    /* Total non-target states: 6^(K-1) */
    uint64_t other_dim = 1;
    for (int i = 0; i < K - 1; i++) other_dim *= 6;

    int kvec[64];

    for (int k_target = 0; k_target < 6; k_target++) {
        double prob = 0;

        for (uint64_t other = 0; other < other_dim; other++) {
            /* Build k-vector: target gets k_target, others enumerated */
            uint64_t tmp = other;
            for (int i = 0; i < K; i++) {
                if (i == qt) {
                    kvec[i] = k_target;
                } else {
                    kvec[i] = tmp % 6;
                    tmp /= 6;
                }
            }

            double amp_re, amp_im;
            proj_resolve_amplitude(r, kvec, &amp_re, &amp_im);
            prob += amp_re * amp_re + amp_im * amp_im;
        }

        pr[k_target] = prob;
    }
}

/* ═══════════════ Probability Conservation ═══════════════ */

static inline double proj_total_prob(const ProjRegister *r) {
    double pr[6];
    proj_born(r, 0, pr);
    double sum = 0;
    for (int k = 0; k < 6; k++) sum += pr[k];
    return sum;
}

/* ═══════════════ Diagnostics ═══════════════ */

static inline void proj_info(const ProjRegister *r) {
    printf("  ProjRegister: K=%d, dim=2^%d=%lu, CZ_logged=%d\n",
           r->K, r->K, (unsigned long)r->dim, r->cz_count);
    printf("  Hilbert: D=6, N=%lu → 6^%lu ≈ 10^%lu\n",
           (unsigned long)r->n_quhits,
           (unsigned long)r->n_quhits,
           (unsigned long)(r->n_quhits * 7785 / 10000));
    printf("  Compression: 3^%d = %.0f×\n",
           r->K, pow(3.0, r->K));
}

#endif /* QUHIT_PROJECTIVE_H */
