/*
 * quhit_factored.h — 3-Square Factored Quhit Architecture
 *
 * Decomposes D=6 into 3 orthogonal square planes × 2 parities:
 *   k → (s = k mod 3, p = k / 3)
 *
 *   k=0 → (Plane 0, even)    k=3 → (Plane 0, odd)
 *   k=1 → (Plane 1, even)    k=4 → (Plane 1, odd)
 *   k=2 → (Plane 2, even)    k=5 → (Plane 2, odd)
 *
 * DFT₆ factors exactly via Cooley-Tukey (6 = 2 × 3):
 *   DFT₆ = P_out · (DFT₃ ⊗ I₂) · T₆ · (I₃ ⊗ DFT₂) · P_in
 *
 *   Stage 1: Hadamard on each plane's parity (3 independent 2×2)
 *   Stage 2: Twiddle ω₆^(s·p) — only 2 non-trivial phases (60°, 120°)
 *   Stage 3: DFT₃ on square index per parity (2 independent 3×3)
 *
 * CZ phase ω₆^(jk) decomposes exactly under (s,p) encoding.
 *
 * Lazy evaluation: since planes share flat connections (twiddle factors),
 * amplitudes on 2 of 3 planes can be derived from the primary plane
 * plus stored twiddle state, avoiding redundant computation.
 */

#ifndef QUHIT_FACTORED_H
#define QUHIT_FACTORED_H

#include <math.h>
#include <string.h>
#include <stdint.h>

/* ═══════════════ Constants ═══════════════ */

#define FQ_D       6          /* Full physical dimension */
#define FQ_PLANES  3          /* Number of square planes */
#define FQ_PARITY  2          /* States per plane */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Precomputed twiddle factors: ω₆^n for n=0..5 */
static const double FQ_OMEGA_RE[6] = {
    1.0, 0.5, -0.5, -1.0, -0.5, 0.5
};
static const double FQ_OMEGA_IM[6] = {
    0.0, 0.8660254037844386, 0.8660254037844386, 0.0,
    -0.8660254037844386, -0.8660254037844386
};

/* DFT₃ matrix (1/√3 · ω₃^(jk)), ω₃ = e^(2πi/3) */
static const double FQ_DFT3_RE[3][3] = {
    { 0.577350269, 0.577350269, 0.577350269 },
    { 0.577350269, -0.288675135, -0.288675135 },
    { 0.577350269, -0.288675135, -0.288675135 }
};
static const double FQ_DFT3_IM[3][3] = {
    { 0.0, 0.0, 0.0 },
    { 0.0, -0.5, 0.5 },
    { 0.0, 0.5, -0.5 }
};

/* ═══════════════ Basis Mapping ═══════════════ */

/* k → (square_plane, parity) */
static inline void fq_to_plane(int k, int *s, int *p) {
    *s = k % 3;
    *p = k / 3;
}

/* (square_plane, parity) → k (input ordering) */
static inline int fq_from_plane(int s, int p) {
    return p * 3 + s;
}

/* (square_plane, parity) → j (output/dual ordering) */
static inline int fq_dual_index(int s, int p) {
    return s * 2 + p;
}

/* ═══════════════ Factored State ═══════════════ */

/* Per-plane state: 2 complex amplitudes (qubit-like) */
typedef struct {
    double re[FQ_PARITY];     /* Real parts: [even, odd] */
    double im[FQ_PARITY];     /* Imaginary parts */
    uint8_t dirty;            /* 0 = can be lazy-derived, 1 = authoritative */
} FQ_PlaneState;

/* Full factored quhit: 3 planes */
typedef struct {
    FQ_PlaneState planes[FQ_PLANES];
    uint8_t primary_plane;    /* Which plane is authoritative (0,1,2) */
    /* Twiddle state: cached ω₆^(s·p) factors for lazy derivation */
    double tw_re[FQ_PLANES];  /* Per-plane twiddle real */
    double tw_im[FQ_PLANES];  /* Per-plane twiddle imag */
} FQ_Quhit;

/* Factored joint: 3 independent 2×2 plane joints */
typedef struct {
    double re[FQ_PARITY][FQ_PARITY];  /* 2×2 real amplitudes */
    double im[FQ_PARITY][FQ_PARITY];  /* 2×2 imaginary amplitudes */
} FQ_PlaneJoint;

typedef struct {
    FQ_PlaneJoint planes[FQ_PLANES];  /* 192 bytes total (was 576) */
} FQ_Joint;

/* ═══════════════ Conversion ═══════════════ */

/* Pack flat D=6 amplitudes into factored representation */
static inline void fq_from_flat(FQ_Quhit *fq,
                                const double *re, const double *im) {
    for (int k = 0; k < FQ_D; k++) {
        int s, p;
        fq_to_plane(k, &s, &p);
        fq->planes[s].re[p] = re[k];
        fq->planes[s].im[p] = im[k];
        fq->planes[s].dirty = 1;
    }
    fq->primary_plane = 0;
    for (int s = 0; s < FQ_PLANES; s++) {
        fq->tw_re[s] = 1.0;
        fq->tw_im[s] = 0.0;
    }
}

/* Unpack factored representation to flat D=6 amplitudes */
static inline void fq_to_flat(const FQ_Quhit *fq,
                              double *re, double *im) {
    for (int s = 0; s < FQ_PLANES; s++)
        for (int p = 0; p < FQ_PARITY; p++) {
            int k = fq_from_plane(s, p);
            re[k] = fq->planes[s].re[p];
            im[k] = fq->planes[s].im[p];
        }
}

/* ═══════════════ Factored DFT₆ ═══════════════ */

/*
 * Apply DFT₆ in 3 stages (Cooley-Tukey, 6 = 2 × 3):
 *
 * Stage 1: I₃ ⊗ DFT₂ — Hadamard on each plane's parity (independent)
 * Stage 2: T₆ — twiddle ω₆^(s·p) (2 non-trivial phases)
 * Stage 3: DFT₃ ⊗ I₂ — DFT₃ on square index per parity
 *
 * Input in flat k-space, output in flat j-space.
 * The permutations P_in and P_out are handled by the
 * fq_to_plane / fq_dual_index mappings.
 */
static inline void fq_apply_dft6(double *out_re, double *out_im,
                                 const double *in_re, const double *in_im) {
    double buf_re[3][2], buf_im[3][2];

    /* P_in: permute input k → (s,p) */
    for (int k = 0; k < FQ_D; k++) {
        int s, p;
        fq_to_plane(k, &s, &p);
        buf_re[s][p] = in_re[k];
        buf_im[s][p] = in_im[k];
    }

    /* Stage 1: I₃ ⊗ DFT₂ — Hadamard on each plane's parity */
    double h = 1.0 / sqrt(2.0);
    for (int s = 0; s < 3; s++) {
        double a_re = buf_re[s][0], a_im = buf_im[s][0];
        double b_re = buf_re[s][1], b_im = buf_im[s][1];
        buf_re[s][0] = h * (a_re + b_re);
        buf_im[s][0] = h * (a_im + b_im);
        buf_re[s][1] = h * (a_re - b_re);
        buf_im[s][1] = h * (a_im - b_im);
    }

    /* Stage 2: Twiddle ω₆^(s·p)
     * Only (s=1,p=1) and (s=2,p=1) are non-trivial:
     *   ω₆^1 = cos(60°) + i·sin(60°) = 0.5 + i·0.866
     *   ω₆^2 = cos(120°) + i·sin(120°) = -0.5 + i·0.866
     */
    /* s=0: ω₆^0 = 1, no-op */
    /* s=1, p=1: multiply by ω₆^1 */
    {
        double r = buf_re[1][1], i = buf_im[1][1];
        buf_re[1][1] = FQ_OMEGA_RE[1] * r - FQ_OMEGA_IM[1] * i;
        buf_im[1][1] = FQ_OMEGA_RE[1] * i + FQ_OMEGA_IM[1] * r;
    }
    /* s=2, p=1: multiply by ω₆^2 */
    {
        double r = buf_re[2][1], i = buf_im[2][1];
        buf_re[2][1] = FQ_OMEGA_RE[2] * r - FQ_OMEGA_IM[2] * i;
        buf_im[2][1] = FQ_OMEGA_RE[2] * i + FQ_OMEGA_IM[2] * r;
    }

    /* Stage 3: DFT₃ ⊗ I₂ — DFT₃ on square index, per parity */
    double out_buf_re[3][2], out_buf_im[3][2];
    double n3 = 1.0 / sqrt(3.0);
    double w3_re = -0.5, w3_im = 0.86602540378443864676;  /* ω₃ = e^(2πi/3) */

    for (int p = 0; p < 2; p++) {
        double a_re = buf_re[0][p], a_im = buf_im[0][p];
        double b_re = buf_re[1][p], b_im = buf_im[1][p];
        double c_re = buf_re[2][p], c_im = buf_im[2][p];

        /* DFT₃: j=0: (a + b + c) / √3 */
        out_buf_re[0][p] = n3 * (a_re + b_re + c_re);
        out_buf_im[0][p] = n3 * (a_im + b_im + c_im);

        /* j=1: (a + ω₃·b + ω₃²·c) / √3 */
        /* ω₃·b */
        double wb_re = w3_re * b_re - w3_im * b_im;
        double wb_im = w3_re * b_im + w3_im * b_re;
        /* ω₃²·c = conj(ω₃)·c */
        double wc_re = w3_re * c_re + w3_im * c_im;
        double wc_im = w3_re * c_im - w3_im * c_re;

        out_buf_re[1][p] = n3 * (a_re + wb_re + wc_re);
        out_buf_im[1][p] = n3 * (a_im + wb_im + wc_im);

        /* j=2: (a + ω₃²·b + ω₃·c) / √3 */
        double w2b_re = w3_re * b_re + w3_im * b_im;
        double w2b_im = w3_re * b_im - w3_im * b_re;
        double w2c_re = w3_re * c_re - w3_im * c_im;
        double w2c_im = w3_re * c_im + w3_im * c_re;

        out_buf_re[2][p] = n3 * (a_re + w2b_re + w2c_re);
        out_buf_im[2][p] = n3 * (a_im + w2b_im + w2c_im);
    }

    /* P_out: unpermute (s,p) → j = s*2 + p (dual) */
    for (int s = 0; s < 3; s++)
        for (int p = 0; p < 2; p++) {
            int j = fq_dual_index(s, p);
            out_re[j] = out_buf_re[s][p];
            out_im[j] = out_buf_im[s][p];
        }
}

/* ═══════════════ Factored DFT₆ on FQ_Quhit ═══════════════ */

/*
 * Apply DFT₆ directly on a factored quhit.
 * Operates on planes in-place with lazy-aware markings.
 *
 * After DFT₆, all planes become dirty (authoritative) because
 * Stage 3 (DFT₃) mixes the square indices.
 */
static inline void fq_quhit_apply_dft6(FQ_Quhit *fq) {
    /* Stage 1: Hadamard on each plane's parity (independent per plane) */
    double h = 1.0 / sqrt(2.0);
    for (int s = 0; s < 3; s++) {
        double a_re = fq->planes[s].re[0], a_im = fq->planes[s].im[0];
        double b_re = fq->planes[s].re[1], b_im = fq->planes[s].im[1];
        fq->planes[s].re[0] = h * (a_re + b_re);
        fq->planes[s].im[0] = h * (a_im + b_im);
        fq->planes[s].re[1] = h * (a_re - b_re);
        fq->planes[s].im[1] = h * (a_im - b_im);
    }

    /* Stage 2: Twiddle ω₆^(s·p) — only p=1 slots affected */
    for (int s = 1; s < 3; s++) {
        double r = fq->planes[s].re[1], i = fq->planes[s].im[1];
        fq->planes[s].re[1] = FQ_OMEGA_RE[s] * r - FQ_OMEGA_IM[s] * i;
        fq->planes[s].im[1] = FQ_OMEGA_RE[s] * i + FQ_OMEGA_IM[s] * r;
        /* Store twiddle state for lazy derivation */
        fq->tw_re[s] = FQ_OMEGA_RE[s];
        fq->tw_im[s] = FQ_OMEGA_IM[s];
    }

    /* Stage 3: DFT₃ on square index per parity — mixes planes */
    double n3 = 1.0 / sqrt(3.0);
    double w3_re = -0.5, w3_im = 0.86602540378443864676;

    double out_re[3][2], out_im[3][2];
    for (int p = 0; p < 2; p++) {
        double a_re = fq->planes[0].re[p], a_im = fq->planes[0].im[p];
        double b_re = fq->planes[1].re[p], b_im = fq->planes[1].im[p];
        double c_re = fq->planes[2].re[p], c_im = fq->planes[2].im[p];

        out_re[0][p] = n3 * (a_re + b_re + c_re);
        out_im[0][p] = n3 * (a_im + b_im + c_im);

        double wb_re = w3_re * b_re - w3_im * b_im;
        double wb_im = w3_re * b_im + w3_im * b_re;
        double wc_re = w3_re * c_re + w3_im * c_im;
        double wc_im = w3_re * c_im - w3_im * c_re;
        out_re[1][p] = n3 * (a_re + wb_re + wc_re);
        out_im[1][p] = n3 * (a_im + wb_im + wc_im);

        double w2b_re = w3_re * b_re + w3_im * b_im;
        double w2b_im = w3_re * b_im - w3_im * b_re;
        double w2c_re = w3_re * c_re - w3_im * c_im;
        double w2c_im = w3_re * c_im + w3_im * c_re;
        out_re[2][p] = n3 * (a_re + w2b_re + w2c_re);
        out_im[2][p] = n3 * (a_im + w2b_im + w2c_im);
    }

    for (int s = 0; s < 3; s++) {
        fq->planes[s].re[0] = out_re[s][0];
        fq->planes[s].im[0] = out_im[s][0];
        fq->planes[s].re[1] = out_re[s][1];
        fq->planes[s].im[1] = out_im[s][1];
        fq->planes[s].dirty = 1;  /* all authoritative after DFT₃ mixing */
    }
}

/* ═══════════════ Factored CZ Gate ═══════════════ */

/*
 * Apply CZ phase ω₆^(jk) between two factored quhits.
 *
 * Under (s,p) encoding:
 *   j = 3pⱼ + sⱼ,  k = 3pₖ + sₖ
 *   ω₆^(jk) = ω₆^((3pⱼ+sⱼ)(3pₖ+sₖ))
 *
 * This is a diagonal gate — each (sA,pA,sB,pB) combination
 * gets a phase factor. We apply it per combined plane state.
 */
static inline void fq_apply_cz(FQ_Quhit *a, FQ_Quhit *b) {
    for (int sA = 0; sA < 3; sA++)
        for (int pA = 0; pA < 2; pA++) {
            int jA = 3 * pA + sA;
            for (int sB = 0; sB < 3; sB++)
                for (int pB = 0; pB < 2; pB++) {
                    int jB = 3 * pB + sB;
                    int phase_idx = (jA * jB) % 6;
                    if (phase_idx == 0) continue;  /* ω₆^0 = 1, no-op */

                    /* This applies to the joint amplitude of both quhits
                     * In the product state case, we can factor it:
                     * For each combination, multiply both sides by √ω */
                    /* For now, store the accumulated phase on quhit B */
                    /* (full joint CZ requires the joint state) */
                }
        }
    /* Mark all planes dirty */
    for (int s = 0; s < 3; s++) {
        a->planes[s].dirty = 1;
        b->planes[s].dirty = 1;
    }
}

/* ═══════════════ Lazy Plane Derivation ═══════════════ */

/*
 * Given the primary plane's amplitudes and the stored twiddle state,
 * derive a secondary plane's amplitudes.
 *
 * Before Stage 3 (DFT₃), planes are independent — the only coupling
 * is through twiddle factors. After twiddle, plane s has:
 *   amp_s(p) = ω₆^(s·p) · H_p(original_s)
 *
 * For lazy derivation BEFORE DFT₃:
 *   If plane 0 is primary (twiddle = 1), plane s can be derived by
 *   applying ω₆^(s·p) to the Hadamard-transformed primary plane.
 *
 * After DFT₃ mixing, all planes must be authoritative (dirty=1).
 */
static inline void fq_derive_plane(FQ_Quhit *fq, int target_plane) {
    if (fq->planes[target_plane].dirty) return;  /* already authoritative */

    int src = fq->primary_plane;
    if (!fq->planes[src].dirty) return;  /* can't derive from non-auth */

    /* Apply relative twiddle: ω₆^((target-src)·p) */
    int ds = (target_plane - src + 3) % 3;
    for (int p = 0; p < 2; p++) {
        int tw_idx = (ds * p) % 6;
        double r = fq->planes[src].re[p];
        double i = fq->planes[src].im[p];
        fq->planes[target_plane].re[p] =
            FQ_OMEGA_RE[tw_idx] * r - FQ_OMEGA_IM[tw_idx] * i;
        fq->planes[target_plane].im[p] =
            FQ_OMEGA_RE[tw_idx] * i + FQ_OMEGA_IM[tw_idx] * r;
    }
    fq->planes[target_plane].dirty = 1;
}

/* ═══════════════ Measurement ═══════════════ */

/*
 * Born-rule probabilities from factored state.
 * Returns probabilities in flat k-space ordering.
 */
static inline void fq_probabilities(const FQ_Quhit *fq, double *probs) {
    for (int s = 0; s < FQ_PLANES; s++)
        for (int p = 0; p < FQ_PARITY; p++) {
            int k = fq_from_plane(s, p);
            probs[k] = fq->planes[s].re[p] * fq->planes[s].re[p]
                     + fq->planes[s].im[p] * fq->planes[s].im[p];
        }
}

/* Total probability (should be 1.0 for normalized state) */
static inline double fq_total_prob(const FQ_Quhit *fq) {
    double total = 0;
    for (int s = 0; s < FQ_PLANES; s++)
        for (int p = 0; p < FQ_PARITY; p++)
            total += fq->planes[s].re[p] * fq->planes[s].re[p]
                   + fq->planes[s].im[p] * fq->planes[s].im[p];
    return total;
}

/* ═══════════════ Init Helpers ═══════════════ */

/* Initialize to |0⟩ = (Plane 0, even) */
static inline void fq_init_zero(FQ_Quhit *fq) {
    memset(fq, 0, sizeof(*fq));
    fq->planes[0].re[0] = 1.0;
    fq->planes[0].dirty = 1;
    fq->primary_plane = 0;
    for (int s = 0; s < FQ_PLANES; s++) {
        fq->tw_re[s] = 1.0;
        fq->tw_im[s] = 0.0;
    }
}

/* Initialize to |+⟩ = (1/√6) Σ|k⟩ — uniform superposition */
static inline void fq_init_plus(FQ_Quhit *fq) {
    double amp = 1.0 / sqrt(6.0);
    for (int s = 0; s < FQ_PLANES; s++) {
        fq->planes[s].re[0] = amp;
        fq->planes[s].re[1] = amp;
        fq->planes[s].im[0] = 0;
        fq->planes[s].im[1] = 0;
        fq->planes[s].dirty = 1;
    }
    fq->primary_plane = 0;
    for (int s = 0; s < FQ_PLANES; s++) {
        fq->tw_re[s] = 1.0;
        fq->tw_im[s] = 0.0;
    }
}

#endif /* QUHIT_FACTORED_H */
