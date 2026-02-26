/*
 * tensor_svd.h — Shared Jacobi SVD for Tensor Network Overlays
 *
 * Provides truncated SVD of complex matrices using Jacobi
 * eigendecomposition of the Hermitian product M†M.
 *
 * Used by MPS, PEPS 2D, PEPS 3D–6D for 2-site gate application.
 * All inputs/outputs are flat row-major arrays.
 *
 * ── Side-Channel Optimizations (from tns_contraction_probe.c) ──
 *   • Zero attractor: 60% of Jacobi angles < 0.01 → aggressive skip
 *   • Early sweep termination via relative off-diagonal check
 *   • 1/6 spectrum awareness: contraction σ → 1/√D
 */

#ifndef TENSOR_SVD_H
#define TENSOR_SVD_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * JACOBI HERMITIAN EIGENDECOMPOSITION
 *
 * Diagonalizes n×n Hermitian H via Jacobi rotations.
 * Returns eigenvalues in `diag` (descending) and eigenvectors in W.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void tsvd_jacobi_hermitian(double *H_re, double *H_im, int n,
                                  double *diag, double *W_re, double *W_im)
{
    /* Init W = I */
    memset(W_re, 0, (size_t)n * n * sizeof(double));
    memset(W_im, 0, (size_t)n * n * sizeof(double));
    for (int i = 0; i < n; i++) W_re[i * n + i] = 1.0;

    /* Diagonal norm for relative threshold (side-channel: zero attractor) */
    double diag_norm = 0;
    for (int i = 0; i < n; i++)
        diag_norm += H_re[i*n+i] * H_re[i*n+i];
    double sc_thresh = 1e-20 * (diag_norm > 1e-30 ? diag_norm : 1.0);

    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                off += H_re[i*n+j]*H_re[i*n+j] + H_im[i*n+j]*H_im[i*n+j];
        if (off < sc_thresh) break;  /* side-channel: relative convergence */

        for (int p = 0; p < n; p++)
         for (int q = p + 1; q < n; q++) {
             double apr = H_re[p*n+q], api = H_im[p*n+q];
             double mag2 = apr*apr + api*api;
             /* Side-channel zero attractor: 60% of angles are < 0.01.
              * Skip rotation entirely when off-diagonal magnitude² is
              * negligible relative to diagonal gap. This is the single
              * biggest speedup — eliminates ~60% of rotation work. */
             if (mag2 < sc_thresh) continue;
             double mag = sqrt(mag2);

             double hpp = H_re[p*n+p], hqq = H_re[q*n+q];
             double tau = (hqq - hpp) / (2.0 * mag);
             double t = (tau >= 0 ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau*tau));
             double c = 1.0 / sqrt(1.0 + t*t);
             double s = t * c;

             /* Phase to make H[p][q] real: e^{-iθ} */
             double er = apr / mag, ei = -api / mag;

             /* Rotate H */
             H_re[p*n+p] -= t * mag;
             H_re[q*n+q] += t * mag;
             H_re[p*n+q] = 0; H_im[p*n+q] = 0;
             H_re[q*n+p] = 0; H_im[q*n+p] = 0;

             for (int k = 0; k < n; k++) {
                 if (k == p || k == q) continue;
                 /* gp = H[k][p], gq = H[k][q] */
                 double gpr = H_re[k*n+p], gpi = H_im[k*n+p];
                 double gqr = H_re[k*n+q], gqi = H_im[k*n+q];

                 /* Apply phase: gq' = e^{iθ} gq */
                 double gqr2 =  er * gqr + ei * gqi;
                 double gqi2 = -ei * gqr + er * gqi;

                 H_re[k*n+p] =  c * gpr + s * gqr2;
                 H_im[k*n+p] =  c * gpi + s * gqi2;
                 H_re[k*n+q] = -s * gpr + c * gqr2;
                 H_im[k*n+q] = -s * gpi + c * gqi2;

                 /* Hermitian: H[p][k] = conj(H[k][p]) */
                 H_re[p*n+k] =  H_re[k*n+p]; H_im[p*n+k] = -H_im[k*n+p];
                 H_re[q*n+k] =  H_re[k*n+q]; H_im[q*n+k] = -H_im[k*n+q];
             }

             /* Rotate W: W[:,p], W[:,q] */
             for (int k = 0; k < n; k++) {
                 double wpr = W_re[k*n+p], wpi = W_im[k*n+p];
                 double wqr = W_re[k*n+q], wqi = W_im[k*n+q];

                 double wqr2 =  er * wqr + ei * wqi;
                 double wqi2 = -ei * wqr + er * wqi;

                 W_re[k*n+p] =  c * wpr + s * wqr2;
                 W_im[k*n+p] =  c * wpi + s * wqi2;
                 W_re[k*n+q] = -s * wpr + c * wqr2;
                 W_im[k*n+q] = -s * wpi + c * wqi2;
             }
         }
    }

    for (int i = 0; i < n; i++) diag[i] = H_re[i*n+i];

    /* Sort descending by eigenvalue */
    for (int i = 0; i < n - 1; i++) {
        int mx = i;
        for (int j = i + 1; j < n; j++)
            if (diag[j] > diag[mx]) mx = j;
        if (mx != i) {
            double tmp = diag[i]; diag[i] = diag[mx]; diag[mx] = tmp;
            for (int k = 0; k < n; k++) {
                double tr, ti;
                tr = W_re[k*n+i]; W_re[k*n+i] = W_re[k*n+mx]; W_re[k*n+mx] = tr;
                ti = W_im[k*n+i]; W_im[k*n+i] = W_im[k*n+mx]; W_im[k*n+mx] = ti;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * TRUNCATED SVD
 *
 * M (m×n complex) → U (m×chi) × σ (chi) × V† (chi×n)
 * Uses Jacobi eigendecomposition of M†M to find V, σ.
 * U = M V σ⁻¹.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void tsvd_truncated(const double *M_re, const double *M_im,
                           int m, int n, int chi,
                           double *U_re, double *U_im,
                           double *sigma,
                           double *Vc_re, double *Vc_im)
{
    /* Form H = M† M  (n×n Hermitian) */
    size_t hsz = (size_t)n * n;
    double *H_re = (double *)calloc(hsz, sizeof(double));
    double *H_im = (double *)calloc(hsz, sizeof(double));

    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k < m; k++) {
                double ar = M_re[k*n+i], ai = -M_im[k*n+i]; /* conj */
                double br = M_re[k*n+j], bi =  M_im[k*n+j];
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            H_re[i*n+j] = sr; H_im[i*n+j] = si;
            H_re[j*n+i] = sr; H_im[j*n+i] = -si; /* Hermitian */
        }

    /* Jacobi eigendecomposition: H = V D V† */
    double *eig = (double *)calloc(n, sizeof(double));
    double *V_re = (double *)calloc(hsz, sizeof(double));
    double *V_im = (double *)calloc(hsz, sizeof(double));

    tsvd_jacobi_hermitian(H_re, H_im, n, eig, V_re, V_im);

    /* σ = sqrt(eigenvalues), clamped at chi */
    int rank = chi < n ? chi : n;
    if (rank > m) rank = m;
    for (int i = 0; i < rank; i++)
        sigma[i] = eig[i] > 0 ? sqrt(eig[i]) : 0;

    /* U = M V σ⁻¹  (m × rank) */
    memset(U_re, 0, (size_t)m * rank * sizeof(double));
    memset(U_im, 0, (size_t)m * rank * sizeof(double));

    for (int j = 0; j < rank; j++) {
        if (sigma[j] < 1e-100) continue;
        double inv = 1.0 / sigma[j];
        for (int i = 0; i < m; i++) {
            double sr = 0, si = 0;
            for (int k = 0; k < n; k++) {
                double mr = M_re[i*n+k], mi = M_im[i*n+k];
                double vr = V_re[k*n+j], vi = V_im[k*n+j];
                sr += mr*vr - mi*vi;
                si += mr*vi + mi*vr;
            }
            U_re[i*rank+j] = sr * inv;
            U_im[i*rank+j] = si * inv;
        }
    }

    /* V† = conj(V)^T  (rank × n) */
    for (int i = 0; i < rank; i++)
        for (int j = 0; j < n; j++) {
            Vc_re[i*n+j] =  V_re[j*n+i];
            Vc_im[i*n+j] = -V_im[j*n+i];
        }

    free(H_re); free(H_im);
    free(eig);
    free(V_re); free(V_im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * MAGIC POINTER SVD — Halko-Martinsson-Tropp Randomized SVD
 *
 * Sparse-native SVD for PEPS tensor contractions.
 * Theta is stored in COO format — never materialized as a dense matrix.
 *
 * Algorithm (Algorithm 5.1 of Halko, Martinsson, Tropp 2011):
 *   1. Draw random Gaussian Ω (n × ℓ), ℓ = rank + oversample
 *   2. Range sketch: Y = (A A†)^q A Ω   (power iteration for gap amp.)
 *   3. Q = QR(Y) — orthonormal basis for range of A  (m × ℓ)
 *   4. B = Q† A  via sparse ops  (ℓ × n — small!)
 *   5. SVD(B) = Ũ Σ V†  via Jacobi on ℓ×ℓ Hermitian B B†
 *   6. U = Q Ũ, truncate to top-chi
 *
 * Complexity: O(nnz × ℓ × (2q+2)) total
 *   vs Jacobi: O(n² × sweeps × n)
 *
 * With nnz ≤ 4096, ℓ ≈ 20, q = 2:  ~500K flops
 * With n = 72:  Jacobi ≈ 50M flops  →  ~100× speedup
 *
 * Side-channel synergy:
 *   • Register 4096-entry cap → nnz naturally bounded
 *   • Zero attractor → nnz typically 40% of capacity
 *   • 1/6 spectrum → q=2 power iters sufficient (gap ~ 1/D)
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* ── Sparse COO entry ── */
typedef struct {
    int    row, col;
    double re, im;
} TsvdSparseEntry;

/* ── LCG PRNG (deterministic, seeded from sparse data) ── */
static inline double tsvd_lcg(uint64_t *s) {
    *s = (*s) * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((int64_t)(*s >> 1)) * 1.0842e-19;  /* ~uniform [-1, 1] */
}

/* ── Sparse A × dense X:  Y[m×k] = A × X[n×k]  ── */
static void tsvd_sp_ax(const TsvdSparseEntry *sp, int nnz,
                        int m, int k,
                        const double *X_re, const double *X_im,
                        double *Y_re, double *Y_im)
{
    memset(Y_re, 0, (size_t)m * k * sizeof(double));
    memset(Y_im, 0, (size_t)m * k * sizeof(double));
    for (int e = 0; e < nnz; e++) {
        int r = sp[e].row, c = sp[e].col;
        double ar = sp[e].re, ai = sp[e].im;
        for (int j = 0; j < k; j++) {
            double xr = X_re[c*k+j], xi = X_im[c*k+j];
            Y_re[r*k+j] += ar*xr - ai*xi;
            Y_im[r*k+j] += ar*xi + ai*xr;
        }
    }
}

/* ── Sparse A† × dense X:  Y[n×k] = A† × X[m×k]  ── */
static void tsvd_sp_ahx(const TsvdSparseEntry *sp, int nnz,
                          int n, int k,
                          const double *X_re, const double *X_im,
                          double *Y_re, double *Y_im)
{
    memset(Y_re, 0, (size_t)n * k * sizeof(double));
    memset(Y_im, 0, (size_t)n * k * sizeof(double));
    for (int e = 0; e < nnz; e++) {
        int r = sp[e].row, c = sp[e].col;
        double ar = sp[e].re, ai = -sp[e].im;  /* conjugate transpose */
        for (int j = 0; j < k; j++) {
            double xr = X_re[r*k+j], xi = X_im[r*k+j];
            Y_re[c*k+j] += ar*xr - ai*xi;
            Y_im[c*k+j] += ar*xi + ai*xr;
        }
    }
}

/* ── Modified Gram-Schmidt QR on rows×cols complex block ── */
static void tsvd_mgs(double *Q_re, double *Q_im, int rows, int cols)
{
    for (int j = 0; j < cols; j++) {
        /* Orthogonalize column j against 0..j-1 */
        for (int k = 0; k < j; k++) {
            /* inner = <col_k, col_j> = sum conj(col_k[i]) * col_j[i] */
            double dr = 0, di = 0;
            for (int i = 0; i < rows; i++) {
                dr += Q_re[i*cols+k]*Q_re[i*cols+j] + Q_im[i*cols+k]*Q_im[i*cols+j];
                di += Q_re[i*cols+k]*Q_im[i*cols+j] - Q_im[i*cols+k]*Q_re[i*cols+j];
            }
            /* col_j -= inner * col_k */
            for (int i = 0; i < rows; i++) {
                Q_re[i*cols+j] -= dr*Q_re[i*cols+k] + di*Q_im[i*cols+k];
                Q_im[i*cols+j] -= dr*Q_im[i*cols+k] - di*Q_re[i*cols+k];
            }
        }
        /* Normalize */
        double norm = 0;
        for (int i = 0; i < rows; i++)
            norm += Q_re[i*cols+j]*Q_re[i*cols+j] + Q_im[i*cols+j]*Q_im[i*cols+j];
        if (norm > 1e-30) {
            double inv = 1.0 / sqrt(norm);
            for (int i = 0; i < rows; i++) {
                Q_re[i*cols+j] *= inv;
                Q_im[i*cols+j] *= inv;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * tsvd_sparse_power — Randomized SVD on sparse COO matrix
 *
 * Input:  sp[nnz] = COO Theta (m × n), chi = target rank
 * Output: U (m×chi), sigma(chi), Vc (chi×n) — same as tsvd_truncated
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void tsvd_sparse_power(const TsvdSparseEntry *sp, int nnz,
                               int m, int n, int chi,
                               double *U_re, double *U_im,
                               double *sigma,
                               double *Vc_re, double *Vc_im)
{
    int rank = chi < n ? chi : n;
    if (rank > m) rank = m;

    memset(sigma, 0, rank * sizeof(double));
    memset(U_re,  0, (size_t)m * rank * sizeof(double));
    memset(U_im,  0, (size_t)m * rank * sizeof(double));
    memset(Vc_re, 0, (size_t)rank * n * sizeof(double));
    memset(Vc_im, 0, (size_t)rank * n * sizeof(double));

    if (nnz == 0 || rank == 0) return;

    /* Oversampling parameter — stabilizes flat spectra */
    int p = rank < 10 ? rank : 10;
    int ell = rank + p;          /* sketch width */
    if (ell > n) ell = n;
    if (ell > m) ell = m;

    /* ── Step 1: Draw random Ω (n × ℓ) ── */
    uint64_t rng = 0xCAFEBABE13579BDFULL;
    for (int e = 0; e < nnz && e < 8; e++)
        rng ^= (uint64_t)(sp[e].re * 1e9) ^ ((uint64_t)(sp[e].im * 1e9) << 32);

    size_t osz = (size_t)n * ell;
    double *Om_re = (double *)malloc(osz * sizeof(double));
    double *Om_im = (double *)malloc(osz * sizeof(double));
    for (int i = 0; i < (int)osz; i++) {
        Om_re[i] = tsvd_lcg(&rng);
        Om_im[i] = tsvd_lcg(&rng);
    }

    /* ── Step 2: Range sketch with power iteration ──
     * Y = (A A†)^q A Ω
     * q = 2 is sufficient for PEPS (1/6 spectrum gap).
     * Each step: Y = A Ω, then q times: Y = A(A†Y). */

    size_t ysz = (size_t)m * ell;
    size_t tsz = (size_t)n * ell;
    double *Y_re  = (double *)malloc(ysz * sizeof(double));
    double *Y_im  = (double *)malloc(ysz * sizeof(double));
    double *T_re  = (double *)malloc(tsz * sizeof(double));
    double *T_im  = (double *)malloc(tsz * sizeof(double));

    /* Y = A Ω */
    tsvd_sp_ax(sp, nnz, m, ell, Om_re, Om_im, Y_re, Y_im);
    free(Om_re); free(Om_im);

    int q = 3;  /* power iterations — 3 sufficient for 1/6 gap + dense */
    for (int qi = 0; qi < q; qi++) {
        tsvd_mgs(Y_re, Y_im, m, ell);        /* stabilize */
        /* T = A† Y  (n × ℓ) */
        tsvd_sp_ahx(sp, nnz, n, ell, Y_re, Y_im, T_re, T_im);
        tsvd_mgs(T_re, T_im, n, ell);        /* stabilize */
        /* Y = A T   (m × ℓ) */
        tsvd_sp_ax(sp, nnz, m, ell, T_re, T_im, Y_re, Y_im);
    }

    /* ── Step 3: Q = QR(Y)  — orthonormal basis for range, m × ℓ ── */
    tsvd_mgs(Y_re, Y_im, m, ell);
    /* Q is now Y_re, Y_im (in-place) */
    double *Q_re = Y_re, *Q_im = Y_im;

    /* ── Step 4: B = Q† A  (ℓ × n) via sparse transpose multiply ──
     * B[i][j] = sum_k conj(Q[k][i]) * A[k][j]
     * = sum over sparse entries (r,c,a): conj(Q[r][i]) * a * δ(col=j)
     */
    size_t bsz = (size_t)ell * n;
    double *B_re = (double *)calloc(bsz, sizeof(double));
    double *B_im = (double *)calloc(bsz, sizeof(double));

    for (int e = 0; e < nnz; e++) {
        int r = sp[e].row, c = sp[e].col;
        double ar = sp[e].re, ai = sp[e].im;
        for (int i = 0; i < ell; i++) {
            /* conj(Q[r][i]) * A[r][c] */
            double qr = Q_re[r*ell+i], qi = -Q_im[r*ell+i]; /* conjugate */
            B_re[i*n+c] += qr*ar - qi*ai;
            B_im[i*n+c] += qr*ai + qi*ar;
        }
    }

    /* ── Step 5: SVD of small B (ℓ × n) via Jacobi on B†B (n × n) ──
     * For PEPS: ℓ ≈ 22, n ≈ 72 → B†B is 72×72.
     * But we can do better: form B B† (ℓ × ℓ) instead!
     * ℓ ≈ 22 → Jacobi on 22×22 = trivially fast. */
    size_t lsz = (size_t)ell * ell;
    double *BBh_re = (double *)calloc(lsz, sizeof(double));
    double *BBh_im = (double *)calloc(lsz, sizeof(double));

    /* BB† [ℓ×ℓ] = B × B†  where B is ℓ×n */
    for (int i = 0; i < ell; i++)
        for (int j = i; j < ell; j++) {
            double sr = 0, si = 0;
            for (int k = 0; k < n; k++) {
                double ar = B_re[i*n+k], ai = B_im[i*n+k];
                double br = B_re[j*n+k], bi = -B_im[j*n+k]; /* conj(B[j][k]) */
                sr += ar*br - ai*bi;
                si += ar*bi + ai*br;
            }
            BBh_re[i*ell+j] = sr; BBh_im[i*ell+j] = si;
            BBh_re[j*ell+i] = sr; BBh_im[j*ell+i] = -si;
        }

    double *eig   = (double *)calloc(ell, sizeof(double));
    double *Ub_re = (double *)calloc(lsz, sizeof(double));
    double *Ub_im = (double *)calloc(lsz, sizeof(double));

    tsvd_jacobi_hermitian(BBh_re, BBh_im, ell, eig, Ub_re, Ub_im);

    /* σ = sqrt(eigenvalues), keep top-rank */
    for (int i = 0; i < rank; i++)
        sigma[i] = (i < ell && eig[i] > 0) ? sqrt(eig[i]) : 0;

    /* ── Step 6: Reconstruct U and V† ──
     * U_full = Q @ Ub  (m × ℓ → m × rank, truncated)
     * V = B† Ub σ⁻¹    (n × rank)
     * Vc = conj(V)^T   (rank × n) */

    /* U = Q × Ub[:, 0:rank]  (m × rank) */
    for (int j = 0; j < rank; j++) {
        for (int i = 0; i < m; i++) {
            double sr = 0, si = 0;
            for (int k = 0; k < ell; k++) {
                double qr = Q_re[i*ell+k], qi = Q_im[i*ell+k];
                double ur = Ub_re[k*ell+j], ui = Ub_im[k*ell+j];
                sr += qr*ur - qi*ui;
                si += qr*ui + qi*ur;
            }
            U_re[i*rank+j] = sr;
            U_im[i*rank+j] = si;
        }
    }

    /* V† computed as: V[i][j] = (B† Ub σ⁻¹)[i][j]
     * V[i][j] = sum_k conj(B[k][i]) * Ub[k][j] / σ[j]
     * Vc[j][i] = conj(V[i][j]) */
    for (int j = 0; j < rank; j++) {
        if (sigma[j] < 1e-100) continue;
        double inv = 1.0 / sigma[j];
        for (int i = 0; i < n; i++) {
            double sr = 0, si = 0;
            for (int k = 0; k < ell; k++) {
                /* conj(B[k][i]) */
                double br = B_re[k*n+i], bi = -B_im[k*n+i];
                double ur = Ub_re[k*ell+j], ui = Ub_im[k*ell+j];
                sr += br*ur - bi*ui;
                si += br*ui + bi*ur;
            }
            /* V[i][j] = (sr + i*si) * inv */
            /* Vc[j][i] = conj(V[i][j]) */
            Vc_re[j*n+i] =  sr * inv;
            Vc_im[j*n+i] = -si * inv;
        }
    }

    free(Q_re);   free(Q_im);
    free(T_re);   free(T_im);
    free(B_re);   free(B_im);
    free(BBh_re); free(BBh_im);
    free(eig);
    free(Ub_re);  free(Ub_im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * Drop-in replacement for tsvd_truncated — same signature.
 * Converts dense M to COO, calls tsvd_sparse_power.
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void tsvd_truncated_sparse(const double *M_re, const double *M_im,
                                   int m, int n, int chi,
                                   double *U_re, double *U_im,
                                   double *sigma,
                                   double *Vc_re, double *Vc_im)
{
    /* Count nonzeros */
    int nnz = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double mag2 = M_re[i*n+j]*M_re[i*n+j] + M_im[i*n+j]*M_im[i*n+j];
            if (mag2 > 1e-30) nnz++;
        }

    /* Build COO */
    TsvdSparseEntry *sp = (TsvdSparseEntry *)malloc(
        (nnz > 0 ? (size_t)nnz : 1) * sizeof(TsvdSparseEntry));
    int k = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double mag2 = M_re[i*n+j]*M_re[i*n+j] + M_im[i*n+j]*M_im[i*n+j];
            if (mag2 > 1e-30) {
                sp[k].row = i; sp[k].col = j;
                sp[k].re = M_re[i*n+j]; sp[k].im = M_im[i*n+j];
                k++;
            }
        }

    tsvd_sparse_power(sp, nnz, m, n, chi, U_re, U_im, sigma, Vc_re, Vc_im);
    free(sp);
}

#endif /* TENSOR_SVD_H */

