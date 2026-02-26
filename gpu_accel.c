/*
 * gpu_accel.c — GPU Acceleration via Runtime OpenCL (dlopen)
 *
 * Zero compile-time dependency on OpenCL SDK.
 * Loads libOpenCL.so.1 at runtime, builds a ZGEMM compute kernel,
 * falls back to CPU loops if GPU unavailable.
 *
 * Env: HEXSTATE_NO_GPU=1 to force CPU fallback.
 */

#include "gpu_accel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <math.h>

/* ═══════════════════════════════════════════════════════════════════════════════
 * OpenCL types and function pointer typedefs (avoids needing CL/cl.h)
 * ═══════════════════════════════════════════════════════════════════════════════ */

typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_device_type;
typedef uint64_t cl_mem_flags;
typedef uint64_t cl_command_queue_properties;
typedef uint64_t cl_program_build_info;
typedef uint64_t cl_platform_info;
typedef uint64_t cl_device_info;

typedef void *cl_platform_id;
typedef void *cl_device_id;
typedef void *cl_context;
typedef void *cl_command_queue;
typedef void *cl_program;
typedef void *cl_kernel;
typedef void *cl_mem;
typedef void *cl_event;

/* Constants */
#define CL_SUCCESS                  0
#define CL_DEVICE_TYPE_GPU          (1 << 2)
#define CL_DEVICE_TYPE_CPU          (1 << 1)
#define CL_DEVICE_TYPE_ALL          0xFFFFFFFF
#define CL_MEM_READ_ONLY            (1 << 2)
#define CL_MEM_WRITE_ONLY           (1 << 1)
#define CL_MEM_READ_WRITE           (1 << 0)
#define CL_MEM_COPY_HOST_PTR        (1 << 5)
#define CL_PROGRAM_BUILD_LOG        0x1183
#define CL_DEVICE_NAME              0x102B
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_DEVICE_MAX_COMPUTE_UNITS   0x1002

/* Function pointer types */
typedef cl_int (*fn_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*fn_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_context (*fn_clCreateContext)(void*, cl_uint, const cl_device_id*, void*, void*, cl_int*);
typedef cl_command_queue (*fn_clCreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*);
typedef cl_program (*fn_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*fn_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*);
typedef cl_kernel (*fn_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_mem (*fn_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int (*fn_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*fn_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*fn_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_int, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*fn_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_int, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*fn_clFinish)(cl_command_queue);
typedef cl_int (*fn_clReleaseMemObject)(cl_mem);
typedef cl_int (*fn_clReleaseKernel)(cl_kernel);
typedef cl_int (*fn_clReleaseProgram)(cl_program);
typedef cl_int (*fn_clReleaseCommandQueue)(cl_command_queue);
typedef cl_int (*fn_clReleaseContext)(cl_context);
typedef cl_int (*fn_clGetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*);
typedef cl_int (*fn_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);

/* ═══════════════════════════════════════════════════════════════════════════════
 * Global GPU state
 * ═══════════════════════════════════════════════════════════════════════════════ */

static struct {
    void *lib;
    int   available;
    int   initialized;

    cl_context       ctx;
    cl_command_queue  queue;
    cl_device_id     device;
    cl_program       prog;
    cl_kernel        kern_zgemm_nn;
    cl_kernel        kern_zgemm_cn;

    /* Function pointers */
    fn_clGetPlatformIDs       GetPlatformIDs;
    fn_clGetDeviceIDs         GetDeviceIDs;
    fn_clCreateContext        CreateContext;
    fn_clCreateCommandQueue   CreateCommandQueue;
    fn_clCreateProgramWithSource CreateProgramWithSource;
    fn_clBuildProgram         BuildProgram;
    fn_clCreateKernel         CreateKernel;
    fn_clCreateBuffer         CreateBuffer;
    fn_clSetKernelArg         SetKernelArg;
    fn_clEnqueueNDRangeKernel EnqueueNDRangeKernel;
    fn_clEnqueueReadBuffer    EnqueueReadBuffer;
    fn_clEnqueueWriteBuffer   EnqueueWriteBuffer;
    fn_clFinish               Finish;
    fn_clReleaseMemObject     ReleaseMemObject;
    fn_clReleaseKernel        ReleaseKernel;
    fn_clReleaseProgram       ReleaseProgram;
    fn_clReleaseCommandQueue  ReleaseCommandQueue;
    fn_clReleaseContext       ReleaseContext;
    fn_clGetDeviceInfo        GetDeviceInfo;
    fn_clGetProgramBuildInfo  GetProgramBuildInfo;
} g_gpu = {0};

/* ═══════════════════════════════════════════════════════════════════════════════
 * OpenCL kernel source — complex GEMM
 *
 * Two variants: NN (no transpose) and CN (conjugate-transpose on A).
 * Each work-item computes one element C[row, col].
 * ═══════════════════════════════════════════════════════════════════════════════ */

static const char *ZGEMM_KERNEL_SRC =
"__kernel void zgemm_nn(\n"
"    int M, int N, int K,\n"
"    double alpha_re, double alpha_im,\n"
"    __global const double *A_re, __global const double *A_im, int lda,\n"
"    __global const double *B_re, __global const double *B_im, int ldb,\n"
"    double beta_re, double beta_im,\n"
"    __global double *C_re, __global double *C_im, int ldc)\n"
"{\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"
"    if (row >= M || col >= N) return;\n"
"    double sr = 0.0, si = 0.0;\n"
"    for (int k = 0; k < K; k++) {\n"
"        double ar = A_re[row * lda + k];\n"
"        double ai = A_im[row * lda + k];\n"
"        double br = B_re[k * ldb + col];\n"
"        double bi = B_im[k * ldb + col];\n"
"        sr += ar*br - ai*bi;\n"
"        si += ar*bi + ai*br;\n"
"    }\n"
"    int idx = row * ldc + col;\n"
"    double cr = alpha_re*sr - alpha_im*si;\n"
"    double ci = alpha_re*si + alpha_im*sr;\n"
"    C_re[idx] = beta_re*C_re[idx] - beta_im*C_im[idx] + cr;\n"
"    C_im[idx] = beta_re*C_im[idx] + beta_im*C_re[idx] + ci;\n"
"}\n"
"\n"
"__kernel void zgemm_cn(\n"
"    int M, int N, int K,\n"
"    double alpha_re, double alpha_im,\n"
"    __global const double *A_re, __global const double *A_im, int lda,\n"
"    __global const double *B_re, __global const double *B_im, int ldb,\n"
"    double beta_re, double beta_im,\n"
"    __global double *C_re, __global double *C_im, int ldc)\n"
"{\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"
"    if (row >= M || col >= N) return;\n"
"    double sr = 0.0, si = 0.0;\n"
"    for (int k = 0; k < K; k++) {\n"
"        double ar =  A_re[k * lda + row];\n"
"        double ai = -A_im[k * lda + row];\n"
"        double br = B_re[k * ldb + col];\n"
"        double bi = B_im[k * ldb + col];\n"
"        sr += ar*br - ai*bi;\n"
"        si += ar*bi + ai*br;\n"
"    }\n"
"    int idx = row * ldc + col;\n"
"    double cr = alpha_re*sr - alpha_im*si;\n"
"    double ci = alpha_re*si + alpha_im*sr;\n"
"    C_re[idx] = beta_re*C_re[idx] - beta_im*C_im[idx] + cr;\n"
"    C_im[idx] = beta_re*C_im[idx] + beta_im*C_re[idx] + ci;\n"
"}\n";

/* ═══════════════════════════════════════════════════════════════════════════════
 * INIT / SHUTDOWN
 * ═══════════════════════════════════════════════════════════════════════════════ */

#define LOAD_FN(name) do { \
    g_gpu.name = (fn_cl##name)dlsym(g_gpu.lib, "cl" #name); \
    if (!g_gpu.name) { dlclose(g_gpu.lib); g_gpu.lib = NULL; return 0; } \
} while(0)

int gpu_accel_init(void)
{
    if (g_gpu.initialized) return g_gpu.available;
    g_gpu.initialized = 1;

    /* Env override */
    if (getenv("HEXSTATE_NO_GPU")) {
        fprintf(stderr, "[gpu_accel] HEXSTATE_NO_GPU set — CPU fallback\n");
        return 0;
    }

    /* Try to load OpenCL */
    g_gpu.lib = dlopen("libOpenCL.so.1", RTLD_LAZY);
    if (!g_gpu.lib) g_gpu.lib = dlopen("libOpenCL.so", RTLD_LAZY);
    if (!g_gpu.lib) {
        fprintf(stderr, "[gpu_accel] OpenCL not found — CPU fallback\n");
        return 0;
    }

    /* Load function pointers */
    LOAD_FN(GetPlatformIDs);
    LOAD_FN(GetDeviceIDs);
    LOAD_FN(CreateContext);
    LOAD_FN(CreateCommandQueue);
    LOAD_FN(CreateProgramWithSource);
    LOAD_FN(BuildProgram);
    LOAD_FN(CreateKernel);
    LOAD_FN(CreateBuffer);
    LOAD_FN(SetKernelArg);
    LOAD_FN(EnqueueNDRangeKernel);
    LOAD_FN(EnqueueReadBuffer);
    LOAD_FN(EnqueueWriteBuffer);
    LOAD_FN(Finish);
    LOAD_FN(ReleaseMemObject);
    LOAD_FN(ReleaseKernel);
    LOAD_FN(ReleaseProgram);
    LOAD_FN(ReleaseCommandQueue);
    LOAD_FN(ReleaseContext);
    LOAD_FN(GetDeviceInfo);
    LOAD_FN(GetProgramBuildInfo);

    /* Get platform + device (prefer GPU, fall back to CPU compute) */
    cl_platform_id plat;
    cl_uint n_plat = 0;
    if (g_gpu.GetPlatformIDs(1, &plat, &n_plat) != CL_SUCCESS || n_plat == 0) {
        fprintf(stderr, "[gpu_accel] No OpenCL platforms — CPU fallback\n");
        dlclose(g_gpu.lib); g_gpu.lib = NULL;
        return 0;
    }

    cl_int err;
    cl_uint n_dev = 0;
    err = g_gpu.GetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &g_gpu.device, &n_dev);
    if (err != CL_SUCCESS || n_dev == 0) {
        /* Try CPU compute device as fallback */
        err = g_gpu.GetDeviceIDs(plat, CL_DEVICE_TYPE_CPU, 1, &g_gpu.device, &n_dev);
        if (err != CL_SUCCESS || n_dev == 0) {
            fprintf(stderr, "[gpu_accel] No compute devices — CPU fallback\n");
            dlclose(g_gpu.lib); g_gpu.lib = NULL;
            return 0;
        }
    }

    /* Print device name */
    char dev_name[256] = {0};
    g_gpu.GetDeviceInfo(g_gpu.device, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    fprintf(stderr, "[gpu_accel] Using device: %s\n", dev_name);

    /* Create context + command queue */
    g_gpu.ctx = g_gpu.CreateContext(NULL, 1, &g_gpu.device, NULL, NULL, &err);
    if (err != CL_SUCCESS) goto fail;

    g_gpu.queue = g_gpu.CreateCommandQueue(g_gpu.ctx, g_gpu.device, 0, &err);
    if (err != CL_SUCCESS) goto fail;

    /* Compile kernel */
    size_t src_len = strlen(ZGEMM_KERNEL_SRC);
    g_gpu.prog = g_gpu.CreateProgramWithSource(g_gpu.ctx, 1, &ZGEMM_KERNEL_SRC, &src_len, &err);
    if (err != CL_SUCCESS) goto fail;

    err = g_gpu.BuildProgram(g_gpu.prog, 1, &g_gpu.device, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS) {
        /* Print build log */
        char log[4096] = {0};
        g_gpu.GetProgramBuildInfo(g_gpu.prog, g_gpu.device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "[gpu_accel] Build failed:\n%s\n", log);
        goto fail;
    }

    g_gpu.kern_zgemm_nn = g_gpu.CreateKernel(g_gpu.prog, "zgemm_nn", &err);
    if (err != CL_SUCCESS) goto fail;
    g_gpu.kern_zgemm_cn = g_gpu.CreateKernel(g_gpu.prog, "zgemm_cn", &err);
    if (err != CL_SUCCESS) goto fail;

    g_gpu.available = 1;
    fprintf(stderr, "[gpu_accel] GPU ZGEMM ready\n");
    return 1;

fail:
    fprintf(stderr, "[gpu_accel] Init failed (err=%d) — CPU fallback\n", err);
    if (g_gpu.queue)  g_gpu.ReleaseCommandQueue(g_gpu.queue);
    if (g_gpu.ctx)    g_gpu.ReleaseContext(g_gpu.ctx);
    dlclose(g_gpu.lib); g_gpu.lib = NULL;
    g_gpu.available = 0;
    return 0;
}

void gpu_accel_shutdown(void)
{
    if (!g_gpu.lib) return;
    if (g_gpu.kern_zgemm_nn) g_gpu.ReleaseKernel(g_gpu.kern_zgemm_nn);
    if (g_gpu.kern_zgemm_cn) g_gpu.ReleaseKernel(g_gpu.kern_zgemm_cn);
    if (g_gpu.prog)  g_gpu.ReleaseProgram(g_gpu.prog);
    if (g_gpu.queue) g_gpu.ReleaseCommandQueue(g_gpu.queue);
    if (g_gpu.ctx)   g_gpu.ReleaseContext(g_gpu.ctx);
    dlclose(g_gpu.lib);
    memset(&g_gpu, 0, sizeof(g_gpu));
}

int gpu_accel_available(void)
{
    if (!g_gpu.initialized) gpu_accel_init();
    return g_gpu.available;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CPU FALLBACK GEMM
 * ═══════════════════════════════════════════════════════════════════════════════ */

static void cpu_zgemm(char transA, char transB,
                      int M, int N, int K,
                      double alpha_re, double alpha_im,
                      const double *A_re, const double *A_im, int lda,
                      const double *B_re, const double *B_im, int ldb,
                      double beta_re, double beta_im,
                      double *C_re, double *C_im, int ldc)
{
    for (int i = 0; i < M; i++)
     for (int j = 0; j < N; j++) {
         double sr = 0, si = 0;
         for (int k = 0; k < K; k++) {
             double ar, ai, br, bi;
             if (transA == 'C' || transA == 'c') {
                 ar =  A_re[k * lda + i];
                 ai = -A_im[k * lda + i];
             } else if (transA == 'T' || transA == 't') {
                 ar = A_re[k * lda + i];
                 ai = A_im[k * lda + i];
             } else {
                 ar = A_re[i * lda + k];
                 ai = A_im[i * lda + k];
             }
             if (transB == 'C' || transB == 'c') {
                 br =  B_re[j * ldb + k];
                 bi = -B_im[j * ldb + k];
             } else if (transB == 'T' || transB == 't') {
                 br = B_re[j * ldb + k];
                 bi = B_im[j * ldb + k];
             } else {
                 br = B_re[k * ldb + j];
                 bi = B_im[k * ldb + j];
             }
             sr += ar*br - ai*bi;
             si += ar*bi + ai*br;
         }
         int idx = i * ldc + j;
         double cr = alpha_re*sr - alpha_im*si;
         double ci = alpha_re*si + alpha_im*sr;
         double old_re = C_re[idx], old_im = C_im[idx];
         C_re[idx] = beta_re*old_re - beta_im*old_im + cr;
         C_im[idx] = beta_re*old_im + beta_im*old_re + ci;
     }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * GPU GEMM (OpenCL)
 *
 * Transfer matrices to GPU, run kernel, read back result.
 * For small matrices (< 32×32), CPU is faster — skip GPU overhead.
 * ═══════════════════════════════════════════════════════════════════════════════ */

/* Minimum matrix dimension to offload to GPU (below this CPU is faster) */
#define GPU_MIN_DIM 32

void gpu_zgemm(char transA, char transB,
               int M, int N, int K,
               double alpha_re, double alpha_im,
               const double *A_re, const double *A_im, int lda,
               const double *B_re, const double *B_im, int ldb,
               double beta_re, double beta_im,
               double *C_re, double *C_im, int ldc)
{
    /* Auto-init on first call */
    if (!g_gpu.initialized) gpu_accel_init();

    /* Use CPU for small matrices or if GPU unavailable */
    if (!g_gpu.available || (M < GPU_MIN_DIM && N < GPU_MIN_DIM && K < GPU_MIN_DIM)) {
        cpu_zgemm(transA, transB, M, N, K,
                  alpha_re, alpha_im, A_re, A_im, lda,
                  B_re, B_im, ldb, beta_re, beta_im,
                  C_re, C_im, ldc);
        return;
    }

    /* Select kernel based on transA */
    cl_kernel kern;
    int A_rows, A_cols;
    if (transA == 'C' || transA == 'c') {
        kern = g_gpu.kern_zgemm_cn;
        A_rows = K; A_cols = M;  /* A is K×M, using as M×K via conj-transpose */
    } else {
        kern = g_gpu.kern_zgemm_nn;
        A_rows = M; A_cols = K;
    }
    /* For now, transB must be 'N' (all our use cases) */
    int B_rows = K, B_cols = N;

    cl_int err;
    size_t sz_A = (size_t)A_rows * lda * sizeof(double);
    size_t sz_B = (size_t)B_rows * ldb * sizeof(double);
    size_t sz_C = (size_t)M * ldc * sizeof(double);

    /* Create GPU buffers */
    cl_mem d_Are = g_gpu.CreateBuffer(g_gpu.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_A, (void*)A_re, &err);
    cl_mem d_Aim = g_gpu.CreateBuffer(g_gpu.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_A, (void*)A_im, &err);
    cl_mem d_Bre = g_gpu.CreateBuffer(g_gpu.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_B, (void*)B_re, &err);
    cl_mem d_Bim = g_gpu.CreateBuffer(g_gpu.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sz_B, (void*)B_im, &err);
    cl_mem d_Cre = g_gpu.CreateBuffer(g_gpu.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz_C, (void*)C_re, &err);
    cl_mem d_Cim = g_gpu.CreateBuffer(g_gpu.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sz_C, (void*)C_im, &err);

    if (err != CL_SUCCESS) {
        /* GPU buffer alloc failed, fall back to CPU */
        if (d_Are) g_gpu.ReleaseMemObject(d_Are);
        if (d_Aim) g_gpu.ReleaseMemObject(d_Aim);
        if (d_Bre) g_gpu.ReleaseMemObject(d_Bre);
        if (d_Bim) g_gpu.ReleaseMemObject(d_Bim);
        if (d_Cre) g_gpu.ReleaseMemObject(d_Cre);
        if (d_Cim) g_gpu.ReleaseMemObject(d_Cim);
        cpu_zgemm(transA, transB, M, N, K,
                  alpha_re, alpha_im, A_re, A_im, lda,
                  B_re, B_im, ldb, beta_re, beta_im,
                  C_re, C_im, ldc);
        return;
    }

    /* Set kernel args */
    g_gpu.SetKernelArg(kern, 0,  sizeof(int),    &M);
    g_gpu.SetKernelArg(kern, 1,  sizeof(int),    &N);
    g_gpu.SetKernelArg(kern, 2,  sizeof(int),    &K);
    g_gpu.SetKernelArg(kern, 3,  sizeof(double), &alpha_re);
    g_gpu.SetKernelArg(kern, 4,  sizeof(double), &alpha_im);
    g_gpu.SetKernelArg(kern, 5,  sizeof(cl_mem), &d_Are);
    g_gpu.SetKernelArg(kern, 6,  sizeof(cl_mem), &d_Aim);
    g_gpu.SetKernelArg(kern, 7,  sizeof(int),    &lda);
    g_gpu.SetKernelArg(kern, 8,  sizeof(cl_mem), &d_Bre);
    g_gpu.SetKernelArg(kern, 9,  sizeof(cl_mem), &d_Bim);
    g_gpu.SetKernelArg(kern, 10, sizeof(int),    &ldb);
    g_gpu.SetKernelArg(kern, 11, sizeof(double), &beta_re);
    g_gpu.SetKernelArg(kern, 12, sizeof(double), &beta_im);
    g_gpu.SetKernelArg(kern, 13, sizeof(cl_mem), &d_Cre);
    g_gpu.SetKernelArg(kern, 14, sizeof(cl_mem), &d_Cim);
    g_gpu.SetKernelArg(kern, 15, sizeof(int),    &ldc);

    /* Launch: one work-item per output element */
    size_t global[2] = { (size_t)M, (size_t)N };
    err = g_gpu.EnqueueNDRangeKernel(g_gpu.queue, kern, 2, NULL, global, NULL, 0, NULL, NULL);
    g_gpu.Finish(g_gpu.queue);

    /* Read back result */
    g_gpu.EnqueueReadBuffer(g_gpu.queue, d_Cre, 1, 0, sz_C, C_re, 0, NULL, NULL);
    g_gpu.EnqueueReadBuffer(g_gpu.queue, d_Cim, 1, 0, sz_C, C_im, 0, NULL, NULL);

    /* Cleanup */
    g_gpu.ReleaseMemObject(d_Are);
    g_gpu.ReleaseMemObject(d_Aim);
    g_gpu.ReleaseMemObject(d_Bre);
    g_gpu.ReleaseMemObject(d_Bim);
    g_gpu.ReleaseMemObject(d_Cre);
    g_gpu.ReleaseMemObject(d_Cim);
}
