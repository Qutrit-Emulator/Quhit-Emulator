/*
 * quhit_3ch_register.h — Lazy 3-Square Register (2^N storage)
 *
 * Stores ONLY the parity register: 2^N entries of (b, amplitude).
 * Plane indices (s_i) are NEVER stored — derived on demand.
 *
 * DFT₆ = Hadamard on parity bit (twiddle + DFT₃ are derivation-time ops).
 *
 * CZ between quhits i,j:
 *   In the physical basis, phase ω₆^{k_i·k_j} with k = 3p + s.
 *   Since planes are uniform (all s_i equally probable), the CZ effect
 *   on the stored parity amplitude is the coherent average:
 *
 *     F(p_i, p_j) = (1/9) Σ_{s_i,s_j=0}^{2} ω₆^{(3p_i+s_i)(3p_j+s_j)}
 *
 *   This is a known analytical function of (p_i, p_j) — precomputed.
 *   CZ multiplies each entry by F(p_i, p_j).
 *
 * Born probabilities:
 *   P(k_i = 3p+s) = (1/3) × Σ_{b: b_i=p} |amp(b)|²
 *   (uniform over 3 planes, each contributes 1/3)
 *
 * After CZ, the plane uniformity is broken. The CZ factor F already
 * encodes the plane-dependent phases into the parity amplitudes.
 * Born probs after CZ use the modified amplitudes directly.
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

static const double W6R[6] = {1.0, 0.5, -0.5, -1.0, -0.5, 0.5};
static const double W6I[6] = {0.0, 0.86602540378443864676,
     0.86602540378443864676, 0.0, -0.86602540378443864676,
    -0.86602540378443864676};

#define FQ_H2 0.70710678118654752440

/* ═══════════════════════════════════════════════════════════════
 * CZ gate history — recorded, not executed
 * At observation time, plane-dependent phases are derived.
 * ═══════════════════════════════════════════════════════════════ */

#define MAX_CZ_GATES 256

typedef struct { int qi, qj; } CZGate;

/* ═══════════════════════════════════════════════════════════════
 * Hash table for (b → amplitude)
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    uint64_t b;
    double   re, im;
    uint8_t  used;
} PSlot;

typedef struct { uint64_t b; double re, im; } PE;

typedef struct {
    PSlot *tab;
    int    cap, cnt, N;
    CZGate cz[MAX_CZ_GATES];
    int    ncz;
    uint64_t dft_mask; /* bit i=1 → quhit i has been DFT₆'d, planes uniform */
} FQR;

static inline uint64_t pmix(uint64_t b) {
    b ^= b >> 33;
    b *= 0xFF51AFD7ED558CCDULL;
    b ^= b >> 33;
    return b;
}

static inline FQR *fqr_alloc(int N) {
    FQR *r = (FQR *)calloc(1, sizeof *r);
    r->N = N; r->cap = 256; r->ncz = 0; r->dft_mask = 0;
    r->tab = (PSlot *)calloc(r->cap, sizeof(PSlot));
    return r;
}

static inline void fqr_free(FQR *r) { free(r->tab); free(r); }

static inline void fqr_rehash(FQR *r) {
    int oc = r->cap; PSlot *ot = r->tab;
    r->cap <<= 1; r->cnt = 0;
    r->tab = (PSlot *)calloc(r->cap, sizeof(PSlot));
    int m = r->cap - 1;
    for (int i = 0; i < oc; i++) {
        if (!ot[i].used) continue;
        int j = (int)(pmix(ot[i].b) & m);
        while (r->tab[j].used) j = (j+1) & m;
        r->tab[j] = ot[i]; r->cnt++;
    }
    free(ot);
}

static inline void fqr_add(FQR *r, uint64_t b, double re, double im) {
    if (r->cnt * 10 >= r->cap * 7) fqr_rehash(r);
    int m = r->cap - 1, j = (int)(pmix(b) & m);
    while (1) {
        if (!r->tab[j].used) {
            r->tab[j] = (PSlot){b, re, im, 1}; r->cnt++;
            return;
        }
        if (r->tab[j].b == b) {
            r->tab[j].re += re; r->tab[j].im += im;
            return;
        }
        j = (j+1) & m;
    }
}

static inline PE *fqr_snap(const FQR *r, int *n) {
    PE *a = (PE *)malloc(r->cnt * sizeof(PE));
    int k = 0;
    for (int i = 0; i < r->cap; i++)
        if (r->tab[i].used)
            a[k++] = (PE){r->tab[i].b, r->tab[i].re, r->tab[i].im};
    *n = k; return a;
}

static inline void fqr_clear(FQR *r) {
    memset(r->tab, 0, r->cap * sizeof(PSlot)); r->cnt = 0;
}

static inline void fqr_trim(FQR *r, double eps) {
    int n; PE *a = fqr_snap(r, &n);
    fqr_clear(r);
    double e2 = eps * eps;
    for (int i = 0; i < n; i++)
        if (a[i].re*a[i].re + a[i].im*a[i].im > e2)
            fqr_add(r, a[i].b, a[i].re, a[i].im);
    free(a);
}

/* ═══════════════════════════════════════════════════════════════
 * Init |0...0⟩
 * ═══════════════════════════════════════════════════════════════ */

static inline void fqr_init_zero(FQR *r) {
    fqr_clear(r);
    r->ncz = 0;
    r->dft_mask = 0;
    fqr_add(r, 0, 1.0, 0.0);
}

/* ═══════════════════════════════════════════════════════════════
 * DFT₆ on quhit qi = Hadamard on parity bit qi
 * (twiddle + DFT₃ are identity in the derived-plane basis)
 * ═══════════════════════════════════════════════════════════════ */

static inline void fqr_dft6(FQR *r, int qi) {
    uint64_t mask = 1ULL << qi;
    int n; PE *a = fqr_snap(r, &n);
    fqr_clear(r);
    for (int i = 0; i < n; i++) {
        uint64_t b0 = a[i].b & ~mask, b1 = a[i].b | mask;
        double sg = ((a[i].b >> qi) & 1) ? -1.0 : 1.0;
        fqr_add(r, b0, FQ_H2 * a[i].re, FQ_H2 * a[i].im);
        fqr_add(r, b1, FQ_H2 * sg * a[i].re, FQ_H2 * sg * a[i].im);
    }
    free(a);
    fqr_trim(r, 1e-15);
    r->dft_mask |= mask;   /* mark this quhit as DFT'd: planes now uniform */
}

/* ═══════════════════════════════════════════════════════════════
 * CZ = record gate, don't modify amplitudes
 * ═══════════════════════════════════════════════════════════════ */

static inline void fqr_cz(FQR *r, int qi, int qj) {
    r->cz[r->ncz++] = (CZGate){qi, qj};
}

/* ═══════════════════════════════════════════════════════════════
 * CZ phase for a specific plane assignment svec at parity b
 * Product of ω₆^{k_i·k_j} over all CZ gates
 * ═══════════════════════════════════════════════════════════════ */

static inline void fqr_cz_phase(const FQR *r, uint64_t b,
                                const int *svec, int nq,
                                double *pr, double *pi_out) {
    double re = 1.0, im = 0.0;
    for (int g = 0; g < r->ncz; g++) {
        int qi = r->cz[g].qi, qj = r->cz[g].qj;
        int ki = 3*((b >> qi) & 1) + svec[qi];
        int kj = 3*((b >> qj) & 1) + svec[qj];
        int ph = (ki * kj) % 6;
        if (!ph) continue;
        double nr = W6R[ph]*re - W6I[ph]*im;
        double ni = W6R[ph]*im + W6I[ph]*re;
        re = nr; im = ni;
    }
    *pr = re; *pi_out = im;
}

/* ═══════════════════════════════════════════════════════════════
 * Born probs: derive plane phases at observation time
 *
 * For each entry b, sum over all 3^K plane assignments for
 * the K quhits involved in CZ gates, weighted by coherent phases.
 * ═══════════════════════════════════════════════════════════════ */

static inline void fqr_born(const FQR *r, int qi, double pr[6]) {
    memset(pr, 0, 48);
    int dft_i = (r->dft_mask >> qi) & 1;

    if (r->ncz == 0) {
        /* No CZ: simple dft_mask check */
        for (int i = 0; i < r->cap; i++) {
            if (!r->tab[i].used) continue;
            int p = (r->tab[i].b >> qi) & 1;
            double m = r->tab[i].re*r->tab[i].re + r->tab[i].im*r->tab[i].im;
            if (dft_i)
                for (int s = 0; s < 3; s++) pr[3*p+s] += m / 3.0;
            else
                pr[3*p] += m;
        }
        return;
    }

    /* With CZ: iterate 3^K plane combos for K active quhits */
    uint64_t active = 0;
    for (int g = 0; g < r->ncz; g++) {
        active |= 1ULL << r->cz[g].qi;
        active |= 1ULL << r->cz[g].qj;
    }
    /* Also include qi if DFT'd (needs plane averaging) */
    if (dft_i) active |= 1ULL << qi;

    int aq[64], naq = 0;
    for (int i = 0; i < 64 && i < r->N; i++)
        if ((active >> i) & 1) aq[naq++] = i;

    int ntot = 1;
    for (int i = 0; i < naq; i++) ntot *= 3;

    int svec[64]; memset(svec, 0, sizeof svec);

    for (int combo = 0; combo < ntot; combo++) {
        int c = combo;
        for (int i = 0; i < naq; i++) { svec[aq[i]] = c % 3; c /= 3; }

        /* Skip combos where untouched quhits have s!=0 */
        int skip = 0;
        for (int i = 0; i < naq; i++)
            if (!((r->dft_mask >> aq[i]) & 1) && svec[aq[i]] != 0)
                { skip = 1; break; }
        if (skip) continue;

        /* Normalization: 3 per DFT'd active quhit, 1 per locked */
        double norm = 1.0;
        for (int i = 0; i < naq; i++)
            if ((r->dft_mask >> aq[i]) & 1) norm *= 3.0;

        int si = dft_i ? svec[qi] : 0;

        for (int e = 0; e < r->cap; e++) {
            if (!r->tab[e].used) continue;
            int p = (r->tab[e].b >> qi) & 1;
            double phR, phI;
            fqr_cz_phase(r, r->tab[e].b, svec, naq, &phR, &phI);
            double aR = r->tab[e].re*phR - r->tab[e].im*phI;
            double aI = r->tab[e].re*phI + r->tab[e].im*phR;
            pr[3*p + si] += (aR*aR + aI*aI) / norm;
        }
    }
}

static inline double fqr_corr(const FQR *r, int qi, int qj) {
    double c = 0;
    int di = (r->dft_mask >> qi) & 1, dj = (r->dft_mask >> qj) & 1;

    if (r->ncz == 0) {
        for (int i = 0; i < r->cap; i++) {
            if (!r->tab[i].used) continue;
            int pi = (r->tab[i].b >> qi) & 1;
            int pj = (r->tab[i].b >> qj) & 1;
            double m = r->tab[i].re*r->tab[i].re + r->tab[i].im*r->tab[i].im;
            int sil = di?3:1, sjl = dj?3:1;
            double avg = 0, norm = (double)(sil*sjl);
            for (int si=0; si<sil; si++)
                for (int sj=0; sj<sjl; sj++)
                    avg += (3*pi+si)*(3*pj+sj);
            c += m * avg / norm;
        }
        return c;
    }

    uint64_t active = 0;
    for (int g = 0; g < r->ncz; g++) {
        active |= 1ULL << r->cz[g].qi;
        active |= 1ULL << r->cz[g].qj;
    }
    if (di) active |= 1ULL << qi;
    if (dj) active |= 1ULL << qj;

    int aq[64], naq = 0;
    for (int i = 0; i < 64 && i < r->N; i++)
        if ((active >> i) & 1) aq[naq++] = i;

    int ntot = 1;
    for (int i = 0; i < naq; i++) ntot *= 3;
    int svec[64]; memset(svec, 0, sizeof svec);

    for (int combo = 0; combo < ntot; combo++) {
        int cc = combo;
        for (int i = 0; i < naq; i++) { svec[aq[i]] = cc%3; cc/=3; }
        int skip = 0;
        for (int i = 0; i < naq; i++)
            if (!((r->dft_mask >> aq[i]) & 1) && svec[aq[i]] != 0)
                { skip = 1; break; }
        if (skip) continue;

        double norm = 1.0;
        for (int i = 0; i < naq; i++)
            if ((r->dft_mask >> aq[i]) & 1) norm *= 3.0;

        int si = di ? svec[qi] : 0, sj = dj ? svec[qj] : 0;

        for (int e = 0; e < r->cap; e++) {
            if (!r->tab[e].used) continue;
            int pi = (r->tab[e].b >> qi) & 1;
            int pj = (r->tab[e].b >> qj) & 1;
            double phR, phI;
            fqr_cz_phase(r, r->tab[e].b, svec, naq, &phR, &phI);
            double aR = r->tab[e].re*phR - r->tab[e].im*phI;
            double aI = r->tab[e].re*phI + r->tab[e].im*phR;
            int ki = 3*pi+si, kj = 3*pj+sj;
            c += (double)(ki*kj) * (aR*aR + aI*aI) / norm;
        }
    }
    return c;
}

/* ═══════════════════════════════════════════════════════════════
 * Statistics
 * ═══════════════════════════════════════════════════════════════ */

static inline double fqr_prob(const FQR *r) {
    double s = 0;
    for (int i = 0; i < r->cap; i++)
        if (r->tab[i].used)
            s += r->tab[i].re*r->tab[i].re + r->tab[i].im*r->tab[i].im;
    return s;
}

static inline void fqr_print(const FQR *r, const char *lb) {
    printf("  %s: N=%d, %d entries (2^N storage), prob=%.6f\n",
           lb, r->N, r->cnt, fqr_prob(r));
}

#endif /* QUHIT_3CH_REGISTER_H */
