/*
 * gpu_accel.h — GPU Acceleration Layer for HexState Tensor SVD
 *
 * Zero compile-time dependency: OpenCL loaded via dlopen at runtime.
 * Falls back to CPU if no GPU available.
 *
 * Usage:
 *   gpu_accel_init();               // once at startup
 *   gpu_zgemm(...);                 // in hot loops
 *   gpu_accel_shutdown();           // at exit
 */

#ifndef GPU_ACCEL_H
#define GPU_ACCEL_H

#include <stddef.h>

/* ═══════════════ LIFECYCLE ═══════════════ */

/* Probe for GPU via dlopen("libOpenCL.so.1").
 * Returns 1 if GPU backend loaded, 0 if CPU fallback. */
int  gpu_accel_init(void);
void gpu_accel_shutdown(void);
int  gpu_accel_available(void);

/* ═══════════════ COMPLEX GEMM ═══════════════ */

/*
 * Complex matrix multiply: C = alpha * op(A) * op(B) + beta * C
 *
 * transA/transB: 'N' = no transpose, 'T' = transpose, 'C' = conjugate transpose
 * A is (M × K), B is (K × N), C is (M × N)
 * All matrices stored as separate real/imag arrays, row-major.
 *
 * GPU-accelerated if available, otherwise CPU loop.
 */
void gpu_zgemm(char transA, char transB,
               int M, int N, int K,
               double alpha_re, double alpha_im,
               const double *A_re, const double *A_im, int lda,
               const double *B_re, const double *B_im, int ldb,
               double beta_re, double beta_im,
               double *C_re, double *C_im, int ldc);

#endif /* GPU_ACCEL_H */
