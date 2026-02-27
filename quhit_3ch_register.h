/*
 * quhit_3ch_register.h — N-Body 3-Square Factored Register
 *
 * Scales the FQ_Quhit architecture (quhit_factored.h) to N quhits.
 *
 * Single quhit: 3 planes × 2 parities = 6 amplitudes
 * N quhits:     3 planes × 2^N basis states per plane
 *
 * k_i = 3 * p_i + s   (quhit i has parity p_i = bit i, plane s shared)
 *
 * All N quhits in one entry share the SAME plane index s.
 * The plane index is the "outer" degree of freedom.
 * The parity bits are the "inner" N-qubit register.
 *
 * DFT₆ on quhit i:
 *   Stage 1: Hadamard on bit i (per plane, independently)
 *   Stage 2: Twiddle ω₆^(s · p_i) on plane s, entries where bit i = 1
 *   Stage 3: DFT₃ across planes for each basis state
 *
 * CZ on quhits i,j:
 *   Diagonal phase ω₆^{k_i · k_j} where k = 3p + s.
 *   Since both quhits share plane s:
 *     phase = ω₆^{(3p_i + s)(3p_j + s)}
 *
 * Lazy derivation:
 *   Before DFT₃, planes 1,2 differ from plane 0 only by twiddle.
 *   Plane s, odd-parity entries: amp_s = amp_0 × ω₆^s
 *   → Store only plane 0, derive 1,2 on demand.
 *   After DFT₃, all planes are authoritative (dirty).
 */

#ifndef QUHIT_3CH_REGISTER_H
#define QUHIT_3CH_REGISTER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Constants ── */

static const double W6R[6] = {1.0, 0.5, -0.5, -1.0, -0.5, 0.5};
static const double W6I[6] = {0.0, 0.86602540378443864676,
     0.86602540378443864676, 0.0, -0.86602540378443864676,
    -0.86602540378443864676};

#define H2   0.70710678118654752440   /* 1/√2 */
#define N3   0.57735026918962576451   /* 1/√3 */
#define W3R  (-0.5)
#define W3I  0.86602540378443864676   /* √3/2 */

/* ── Sparse Plane: dynamic array of (basis, amplitude) ── */

typedef struct { uint64_t basis; double re, im; } PlaneEntry;

typedef struct {
    PlaneEntry *e;
    int         n, cap;
    uint8_t     dirty;    /* 1 = authoritative, 0 = can be derived */
} Plane;

static inline void plane_init(Plane *p, int cap) {
    p->cap = cap;
    p->e = (PlaneEntry *)calloc(cap, sizeof(PlaneEntry));
    p->n = 0;
    p->dirty = 0;
}

static inline void plane_free(Plane *p) { free(p->e); }

static inline void plane_clear(Plane *p) { p->n = 0; }

static inline void plane_grow(Plane *p) {
    p->cap = p->cap ? p->cap * 2 : 64;
    p->e = (PlaneEntry *)realloc(p->e, p->cap * sizeof(PlaneEntry));
}

static inline void plane_add(Plane *p, uint64_t b, double re, double im) {
    for (int i = 0; i < p->n; i++)
        if (p->e[i].basis == b) {
            p->e[i].re += re; p->e[i].im += im;
            return;
        }
    if (p->n >= p->cap) plane_grow(p);
    p->e[p->n++] = (PlaneEntry){b, re, im};
}

static inline void plane_set(Plane *p, uint64_t b, double re, double im) {
    for (int i = 0; i < p->n; i++)
        if (p->e[i].basis == b) {
            p->e[i].re = re; p->e[i].im = im;
            return;
        }
    if (p->n >= p->cap) plane_grow(p);
    p->e[p->n++] = (PlaneEntry){b, re, im};
}

static inline void plane_trim(Plane *p, double eps) {
    double e2 = eps * eps;
    int j = 0;
    for (int i = 0; i < p->n; i++)
        if (p->e[i].re * p->e[i].re + p->e[i].im * p->e[i].im > e2)
            p->e[j++] = p->e[i];
    p->n = j;
}

static inline double plane_prob(const Plane *p) {
    double s = 0;
    for (int i = 0; i < p->n; i++)
        s += p->e[i].re * p->e[i].re + p->e[i].im * p->e[i].im;
    return s;
}

/* ── 3-Plane Register ── */

typedef struct {
    Plane    planes[3];
    int      N;           /* number of quhits */
    uint8_t  primary;     /* which plane is authoritative (0,1,2) */
} TCR;

static inline TCR *tcr_alloc(int N) {
    TCR *r = (TCR *)calloc(1, sizeof *r);
    r->N = N;
    r->primary = 0;
    for (int s = 0; s < 3; s++) plane_init(&r->planes[s], 64);
    return r;
}

static inline void tcr_free(TCR *r) {
    for (int s = 0; s < 3; s++) plane_free(&r->planes[s]);
    free(r);
}

/* All quhits |0⟩: k=0 → (s=0, p=0)
 * Only plane 0 has amplitude. Planes 1,2 are genuinely empty.
 * All planes authoritative — no lazy derivation at init. */
static inline void tcr_init_zero(TCR *r) {
    for (int s = 0; s < 3; s++) {
        plane_clear(&r->planes[s]);
        r->planes[s].dirty = 1;  /* all authoritative */
    }
    plane_set(&r->planes[0], 0ULL, 1.0, 0.0);
    r->primary = 0;
}

/* ── Lazy derivation: derive plane s from primary ──
 *
 * Before DFT₃, plane s differs from primary by twiddle ω₆^(s·p_i)
 * for each quhit i. This is a per-entry factor depending on the
 * popcount of set bits.
 *
 * For the initial state (only plane 0 populated, planes 1,2 zero):
 *   plane s has same amplitudes as plane 0 at even-parity entries,
 *   and ω₆^s × plane 0 at odd-parity entries.
 *
 * For N quhits, the twiddle factor for plane s at basis b is:
 *   Π_i ω₆^(s · b_i) = ω₆^(s · popcount(b))
 *
 * since each quhit i contributes ω₆^(s · b_i).
 */
static inline void tcr_derive_plane(TCR *r, int target) {
    if (r->planes[target].dirty) return;

    int src = r->primary;
    if (!r->planes[src].dirty) return;

    plane_clear(&r->planes[target]);
    int ds = target;  /* relative twiddle index */

    for (int i = 0; i < r->planes[src].n; i++) {
        uint64_t b = r->planes[src].e[i].basis;
        double re = r->planes[src].e[i].re;
        double im = r->planes[src].e[i].im;

        /* Twiddle factor: ω₆^(ds · popcount(b)) */
        int pop = __builtin_popcountll(b);
        int tw = (ds * pop) % 6;

        double ore = W6R[tw]*re - W6I[tw]*im;
        double oim = W6R[tw]*im + W6I[tw]*re;

        plane_add(&r->planes[target], b, ore, oim);
    }
    r->planes[target].dirty = 1;
}

/* Ensure all planes are authoritative */
static inline void tcr_materialize(TCR *r) {
    for (int s = 0; s < 3; s++)
        tcr_derive_plane(r, s);
}

/* ── DFT₆ on quhit qi ── */

static inline void tcr_dft6(TCR *r, int qi) {
    uint64_t mask = 1ULL << qi;

    /* Materialize all planes (DFT₃ needs all 3) */
    tcr_materialize(r);

    /* Stage 1: Hadamard on bit qi, per plane independently */
    for (int s = 0; s < 3; s++) {
        int on = r->planes[s].n;
        PlaneEntry *old = (PlaneEntry *)malloc(on * sizeof(PlaneEntry));
        memcpy(old, r->planes[s].e, on * sizeof(PlaneEntry));
        plane_clear(&r->planes[s]);

        for (int i = 0; i < on; i++) {
            uint64_t b0 = old[i].basis & ~mask;
            uint64_t b1 = old[i].basis | mask;
            double re = old[i].re, im = old[i].im;
            double sg = ((old[i].basis >> qi) & 1) ? -1.0 : 1.0;
            plane_add(&r->planes[s], b0, H2 * re, H2 * im);
            plane_add(&r->planes[s], b1, H2 * sg * re, H2 * sg * im);
        }
        free(old);
    }

    /* Stage 2: Twiddle ω₆^s on entries where bit qi = 1 */
    for (int s = 1; s < 3; s++)
        for (int i = 0; i < r->planes[s].n; i++) {
            if (!((r->planes[s].e[i].basis >> qi) & 1)) continue;
            double re = r->planes[s].e[i].re, im = r->planes[s].e[i].im;
            r->planes[s].e[i].re = W6R[s]*re - W6I[s]*im;
            r->planes[s].e[i].im = W6R[s]*im + W6I[s]*re;
        }

    /* Stage 3: DFT₃ across planes for each basis state */
    /* Collect all basis states present across 3 planes */
    int nb = 0, nbcap = r->planes[0].n + r->planes[1].n + r->planes[2].n + 1;
    uint64_t *bases = (uint64_t *)malloc(nbcap * sizeof(uint64_t));

    for (int s = 0; s < 3; s++)
        for (int i = 0; i < r->planes[s].n; i++) {
            uint64_t b = r->planes[s].e[i].basis;
            int found = 0;
            for (int j = 0; j < nb; j++)
                if (bases[j] == b) { found = 1; break; }
            if (!found) {
                if (nb >= nbcap) {
                    nbcap *= 2;
                    bases = (uint64_t *)realloc(bases, nbcap * sizeof(uint64_t));
                }
                bases[nb++] = b;
            }
        }

    /* For each basis state, gather 3 plane amps, apply DFT₃, scatter */
    for (int bi = 0; bi < nb; bi++) {
        uint64_t b = bases[bi];
        double ar[3] = {0}, ai[3] = {0};

        for (int s = 0; s < 3; s++)
            for (int i = 0; i < r->planes[s].n; i++)
                if (r->planes[s].e[i].basis == b) {
                    ar[s] = r->planes[s].e[i].re;
                    ai[s] = r->planes[s].e[i].im;
                }

        /* DFT₃: out[s'] = (1/√3) Σ_s ω₃^{s'·s} in[s] */
        double or_[3], oi[3];
        for (int sp = 0; sp < 3; sp++) {
            or_[sp] = 0; oi[sp] = 0;
            for (int s = 0; s < 3; s++) {
                int ph = (sp * s * 2) % 6;
                or_[sp] += W6R[ph]*ar[s] - W6I[ph]*ai[s];
                oi[sp]  += W6R[ph]*ai[s] + W6I[ph]*ar[s];
            }
            or_[sp] *= N3;  oi[sp] *= N3;
        }

        for (int s = 0; s < 3; s++)
            plane_set(&r->planes[s], b, or_[s], oi[s]);
    }
    free(bases);

    /* After DFT₃, all planes are authoritative */
    for (int s = 0; s < 3; s++) {
        plane_trim(&r->planes[s], 1e-15);
        r->planes[s].dirty = 1;
    }
}

/* ── CZ on quhits qi, qj ── */

static inline void tcr_cz(TCR *r, int qi, int qj) {
    tcr_materialize(r);
    uint64_t mi = 1ULL << qi, mj = 1ULL << qj;

    for (int s = 0; s < 3; s++)
        for (int i = 0; i < r->planes[s].n; i++) {
            int pi = (r->planes[s].e[i].basis & mi) ? 1 : 0;
            int pj = (r->planes[s].e[i].basis & mj) ? 1 : 0;
            int ki = 3*pi + s, kj = 3*pj + s;
            int ph = (ki * kj) % 6;
            if (!ph) continue;
            double re = r->planes[s].e[i].re, im = r->planes[s].e[i].im;
            r->planes[s].e[i].re = W6R[ph]*re - W6I[ph]*im;
            r->planes[s].e[i].im = W6R[ph]*im + W6I[ph]*re;
        }
}

/* ── Statistics ── */

static inline double tcr_prob(const TCR *r) {
    double s = 0;
    for (int i = 0; i < 3; i++) s += plane_prob(&r->planes[i]);
    return s;
}

static inline int tcr_nnz(const TCR *r) {
    return r->planes[0].n + r->planes[1].n + r->planes[2].n;
}

static inline void tcr_print(const TCR *r, const char *lb) {
    printf("  %s: N=%d quhits\n", lb, r->N);
    for (int s = 0; s < 3; s++)
        printf("    plane[%d]: %d entries, prob=%.6f %s\n",
               s, r->planes[s].n, plane_prob(&r->planes[s]),
               r->planes[s].dirty ? "(auth)" : "(lazy)");
    printf("    total: %d entries, prob=%.6f\n", tcr_nnz(r), tcr_prob(r));
}

/* ── Born probabilities ── */

static inline void tcr_born(const TCR *r, int qi, double pr[6]) {
    memset(pr, 0, 6 * sizeof(double));
    uint64_t mask = 1ULL << qi;
    for (int s = 0; s < 3; s++)
        for (int i = 0; i < r->planes[s].n; i++) {
            int p = (r->planes[s].e[i].basis & mask) ? 1 : 0;
            double m = r->planes[s].e[i].re * r->planes[s].e[i].re +
                       r->planes[s].e[i].im * r->planes[s].e[i].im;
            pr[3*p + s] += m;
        }
}

/* ── Two-quhit correlation ⟨k_i · k_j⟩ ── */

static inline double tcr_corr(const TCR *r, int qi, int qj) {
    double c = 0;
    uint64_t mi = 1ULL << qi, mj = 1ULL << qj;
    for (int s = 0; s < 3; s++)
        for (int i = 0; i < r->planes[s].n; i++) {
            int ki = 3*((r->planes[s].e[i].basis & mi)?1:0) + s;
            int kj = 3*((r->planes[s].e[i].basis & mj)?1:0) + s;
            double m = r->planes[s].e[i].re * r->planes[s].e[i].re +
                       r->planes[s].e[i].im * r->planes[s].e[i].im;
            c += (double)(ki*kj) * m;
        }
    return c;
}

#endif /* QUHIT_3CH_REGISTER_H */
