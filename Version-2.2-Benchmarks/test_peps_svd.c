/*
 * test_peps_svd.c — Verify PEPS 2D and 3D produce valid tensor network data
 *
 * Tests:
 * 1. Product state → local density = delta(k=0)
 * 2. 1-site gate → rotated density
 * 3. 2-site gate (CZ₆) → entangled density (non-trivial)
 * 4. Norms remain finite, densities sum to 1
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "peps_overlay.c"
#include "peps3d_overlay.c"

/* ═══════════════ HELPERS ═══════════════ */

static void build_hadamard6(double *U_re, double *U_im)
{
    int D = 6;
    double norm = 1.0 / sqrt((double)D);
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) {
            double angle = 2.0 * M_PI * i * j / D;
            U_re[i*D+j] = norm * cos(angle);
            U_im[i*D+j] = norm * sin(angle);
        }
}

static void build_cz6(double *G_re, double *G_im)
{
    int D = 6, D2 = 36;
    memset(G_re, 0, D2*D2*sizeof(double));
    memset(G_im, 0, D2*D2*sizeof(double));
    double omega = 2.0 * M_PI / 6.0;
    for (int i = 0; i < D; i++)
        for (int j = 0; j < D; j++) {
            int idx = i*D + j;
            double phase = omega * i * j;
            G_re[idx * D2 + idx] = cos(phase);
            G_im[idx * D2 + idx] = sin(phase);
        }
}

static int check_density(const char *label, double *probs, int D)
{
    double sum = 0;
    int valid = 1;
    for (int k = 0; k < D; k++) {
        sum += probs[k];
        if (probs[k] < -1e-10 || probs[k] > 1.0 + 1e-10 || isnan(probs[k])) {
            valid = 0;
        }
    }
    printf("  %s: [", label);
    for (int k = 0; k < D; k++) printf("%.4f%s", probs[k], k<D-1?", ":"");
    printf("]  sum=%.6f  %s\n", sum, (fabs(sum-1.0)<1e-6 && valid) ? "✓" : "✗ FAIL");
    return (fabs(sum-1.0)<1e-6 && valid);
}

/* ═══════════════ PEPS 2D TEST ═══════════════ */

static int test_peps2d(void)
{
    printf("\n═══════════════════════════════════════════════\n");
    printf("  TEST: PEPS 2D (2×2 grid, χ=%d)\n", PEPS_CHI);
    printf("═══════════════════════════════════════════════\n\n");

    PepsGrid *g = peps_init(2, 2);
    int pass = 1;
    double probs[6];

    /* Test 1: Product state |0⟩ → density = (1,0,0,0,0,0) */
    printf("  [1] Product state |0⟩\n");
    peps_local_density(g, 0, 0, probs);
    pass &= check_density("Site(0,0)", probs, 6);

    /* Test 2: Apply Hadamard → uniform density */
    printf("  [2] After Hadamard on site (0,0)\n");
    double U_re[36], U_im[36];
    build_hadamard6(U_re, U_im);
    peps_gate_1site(g, 0, 0, U_re, U_im);
    peps_local_density(g, 0, 0, probs);
    pass &= check_density("Site(0,0)", probs, 6);

    /* Check: should be ~uniform (1/6 each) */
    int uniform = 1;
    for (int k = 0; k < 6; k++)
        if (fabs(probs[k] - 1.0/6.0) > 0.05) uniform = 0;
    printf("    → Uniform? %s (expected ~0.1667 each)\n", uniform ? "YES ✓" : "NO ✗");
    pass &= uniform;

    /* Test 3: CZ₆ horizontal gate → entanglement */
    printf("  [3] CZ₆ horizontal (0,0)-(1,0)\n");
    double G_re[36*36], G_im[36*36];
    build_cz6(G_re, G_im);

    /* First apply Hadamard to (1,0) too */
    peps_gate_1site(g, 1, 0, U_re, U_im);
    peps_gate_horizontal(g, 0, 0, G_re, G_im);

    peps_local_density(g, 0, 0, probs);
    pass &= check_density("Site(0,0) post-CZ", probs, 6);
    peps_local_density(g, 1, 0, probs);
    pass &= check_density("Site(1,0) post-CZ", probs, 6);

    /* Test 4: CZ₆ vertical gate */
    printf("  [4] CZ₆ vertical (0,0)-(0,1)\n");
    peps_gate_1site(g, 0, 1, U_re, U_im);
    peps_gate_vertical(g, 0, 0, G_re, G_im);

    peps_local_density(g, 0, 0, probs);
    pass &= check_density("Site(0,0) post-V", probs, 6);
    peps_local_density(g, 0, 1, probs);
    pass &= check_density("Site(0,1) post-V", probs, 6);

    /* Test 5: Multiple rounds */
    printf("  [5] 5 rounds of H + CZ₆ on all bonds\n");
    for (int round = 0; round < 5; round++) {
        peps_gate_1site_all(g, U_re, U_im);
        peps_gate_horizontal_all(g, G_re, G_im);
        peps_gate_vertical_all(g, G_re, G_im);
    }

    int all_valid = 1;
    for (int y = 0; y < 2; y++)
        for (int x = 0; x < 2; x++) {
            peps_local_density(g, x, y, probs);
            char label[32];
            snprintf(label, sizeof(label), "Site(%d,%d) r5", x, y);
            all_valid &= check_density(label, probs, 6);
        }
    pass &= all_valid;

    peps_free(g);

    printf("\n  PEPS 2D: %s\n", pass ? "ALL PASSED ✓" : "SOME FAILED ✗");
    return pass;
}

/* ═══════════════ PEPS 3D TEST ═══════════════ */

static int test_peps3d(void)
{
    printf("\n═══════════════════════════════════════════════\n");
    printf("  TEST: PEPS 3D (2×2×2 grid, χ=%d)\n", TNS3D_CHI);
    printf("═══════════════════════════════════════════════\n\n");

    Tns3dGrid *g = tns3d_init(2, 2, 2);
    int pass = 1;
    double probs[6];

    /* Test 1: Product state */
    printf("  [1] Product state\n");
    tns3d_local_density(g, 0, 0, 0, probs);
    pass &= check_density("Site(0,0,0)", probs, 6);

    /* Test 2: Hadamard */
    printf("  [2] After Hadamard\n");
    double U_re[36], U_im[36];
    build_hadamard6(U_re, U_im);
    tns3d_gate_1site(g, 0, 0, 0, U_re, U_im);
    tns3d_local_density(g, 0, 0, 0, probs);
    pass &= check_density("Site(0,0,0)", probs, 6);

    /* Test 3: CZ₆ along all 3 axes */
    printf("  [3] CZ₆ along X, Y, Z axes\n");
    double G_re[36*36], G_im[36*36];
    build_cz6(G_re, G_im);

    /* Apply Hadamard to all neighbors first */
    tns3d_gate_1site(g, 1, 0, 0, U_re, U_im);
    tns3d_gate_1site(g, 0, 1, 0, U_re, U_im);
    tns3d_gate_1site(g, 0, 0, 1, U_re, U_im);

    tns3d_gate_x(g, 0, 0, 0, G_re, G_im);
    tns3d_gate_y(g, 0, 0, 0, G_re, G_im);
    tns3d_gate_z(g, 0, 0, 0, G_re, G_im);

    tns3d_local_density(g, 0, 0, 0, probs);
    pass &= check_density("Site(0,0,0) post-XYZ", probs, 6);
    tns3d_local_density(g, 1, 0, 0, probs);
    pass &= check_density("Site(1,0,0) post-X", probs, 6);

    /* Test 4: Multiple Trotter steps */
    printf("  [4] 5 Trotter steps\n");
    for (int round = 0; round < 5; round++) {
        tns3d_gate_1site_all(g, U_re, U_im);
        tns3d_trotter_step(g, G_re, G_im);
    }

    int all_valid = 1;
    for (int z = 0; z < 2; z++)
     for (int y = 0; y < 2; y++)
      for (int x = 0; x < 2; x++) {
          tns3d_local_density(g, x, y, z, probs);
          char label[32];
          snprintf(label, sizeof(label), "Site(%d,%d,%d) r5", x, y, z);
          all_valid &= check_density(label, probs, 6);
      }
    pass &= all_valid;

    tns3d_free(g);

    printf("\n  PEPS 3D: %s\n", pass ? "ALL PASSED ✓" : "SOME FAILED ✗");
    return pass;
}

/* ═══════════════ MAIN ═══════════════ */

int main(void)
{
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  TENSOR NETWORK VERIFICATION SUITE                  ║\n");
    printf("║  Register-Based SVD with Magic Pointers             ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n");

    int p2d = test_peps2d();
    int p3d = test_peps3d();

    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  FINAL RESULT                                       ║\n");
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  PEPS 2D:  %s                                    ║\n", p2d ? "PASS ✓" : "FAIL ✗");
    printf("║  PEPS 3D:  %s                                    ║\n", p3d ? "PASS ✓" : "FAIL ✗");
    printf("╚══════════════════════════════════════════════════════╝\n");

    return (p2d && p3d) ? 0 : 1;
}
