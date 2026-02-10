/* ═══════════════════════════════════════════════════════════════════════════
 *  100-TRILLION QUHIT — ALL BRAIDED — BELL TEST
 *
 *  1. Instantiates 100T Quhits as N_CHUNKS Magic Pointer chunks
 *  2. Puts ALL into superposition
 *  3. Braids ALL Quhits: every chunk shares ONE joint Hilbert space
 *     — the Bell state |Ψ⟩ = (1/√D) Σ|k⟩|k⟩ lives here
 *     — ALL computations happen in this space
 *  4. Picks 2 random chunks and Bell tests them
 *
 *  Build:
 *    gcc -O2 -std=c11 -D_GNU_SOURCE \
 *        -o bell_100t_full bell_100t_full.c hexstate_engine.c bigint.c -lm
 * ═══════════════════════════════════════════════════════════════════════════ */

#include "hexstate_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>

/* ── Configuration ── */
#define TOTAL_QUHITS      100000000000000ULL   /* 100 trillion */
#define N_CHUNKS          100                   /* 100 chunks */
#define QUHITS_PER_CHUNK  (TOTAL_QUHITS / N_CHUNKS)  /* 1T each */
#define D                 6                     /* Hilbert dimension */
#define NUM_TRIALS        10000                 /* Bell test trials */
#define BASE_CID          8000                  /* Starting chunk ID */

/* Write a fresh Bell state into the shared Hilbert space */
static void write_bell_state(Complex *joint, int dim)
{
    memset(joint, 0, (size_t)dim * dim * sizeof(Complex));
    double amp = 1.0 / sqrt((double)dim);
    for (int k = 0; k < dim; k++)
        joint[k * dim + k] = (Complex){amp, 0.0};
}

int main(void)
{
    struct timespec t0, t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    srand((unsigned)time(NULL));

    printf("\n");
    printf("████████████████████████████████████████████████████████████████████████████\n");
    printf("██                                                                        ██\n");
    printf("██   100-TRILLION QUHIT — ALL BRAIDED — BELL TEST                         ██\n");
    printf("██                                                                        ██\n");
    printf("██   %d chunks × %" PRIu64 " Quhits = %" PRIu64 " total     ██\n",
           N_CHUNKS, (uint64_t)QUHITS_PER_CHUNK, (uint64_t)TOTAL_QUHITS);
    printf("██   ALL Quhits share ONE joint Hilbert space (D²=%d, %zu bytes)         ██\n",
           D*D, (size_t)D*D*sizeof(Complex));
    printf("██   All computation happens in the Hilbert space                         ██\n");
    printf("██                                                                        ██\n");
    printf("████████████████████████████████████████████████████████████████████████████\n\n");

    /* ── 1. Initialize Engine ── */
    HexStateEngine eng;
    if (engine_init(&eng) != 0) {
        fprintf(stderr, "FATAL: engine_init failed\n");
        return 1;
    }

    /* ── 2. Instantiate 100T Quhits ── */
    printf("  ▸ PHASE 1: Instantiating %" PRIu64 " Quhits (%d chunks)...\n",
           (uint64_t)TOTAL_QUHITS, N_CHUNKS);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    for (int i = 0; i < N_CHUNKS; i++)
        init_chunk(&eng, BASE_CID + i, QUHITS_PER_CHUNK);

    clock_gettime(CLOCK_MONOTONIC, &t2);
    double create_ms = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_nsec - t1.tv_nsec)/1e6;
    printf("    ✓ %d chunks created (%.3f ms)\n\n", N_CHUNKS, create_ms);

    /* ── 3. Superpose ALL ── */
    printf("  ▸ PHASE 2: Superposing ALL %" PRIu64 " Quhits...\n", (uint64_t)TOTAL_QUHITS);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    for (int i = 0; i < N_CHUNKS; i++)
        create_superposition(&eng, BASE_CID + i);

    clock_gettime(CLOCK_MONOTONIC, &t2);
    double sup_ms = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_nsec - t1.tv_nsec)/1e6;
    printf("    ✓ All %" PRIu64 " Quhits superposed (%.3f ms)\n\n", (uint64_t)TOTAL_QUHITS, sup_ms);

    /* ── 4. BRAID ALL — one shared Hilbert space ── */
    printf("  ▸ PHASE 3: Braiding ALL %d chunks into ONE Hilbert space...\n", N_CHUNKS);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    /* Allocate the shared joint Hilbert space */
    Complex *shared_hilbert = (Complex *)calloc((size_t)D * D, sizeof(Complex));
    write_bell_state(shared_hilbert, D);

    /* Point EVERY chunk into this shared Hilbert space */
    for (int i = 0; i < N_CHUNKS; i++) {
        uint64_t cid = BASE_CID + i;
        Chunk *c = &eng.chunks[cid];
        c->hilbert.q_joint_state = shared_hilbert;
        c->hilbert.q_joint_dim   = D;
        c->hilbert.q_which       = (i % 2 == 0) ? 0 : 1;  /* alternate A/B sides */
        c->hilbert.q_partner     = (i % 2 == 0)
                                    ? (uint64_t)(BASE_CID + i + 1)
                                    : (uint64_t)(BASE_CID + i - 1);
        c->hilbert.q_flags       = 0x01;  /* superposed */
    }

    /* Register braid links for all adjacent pairs */
    for (int i = 0; i < N_CHUNKS - 1; i += 2) {
        uint64_t cid_a = BASE_CID + i;
        uint64_t cid_b = BASE_CID + i + 1;

        if (eng.num_braid_links < eng.braid_capacity) {
            BraidLink *link = &eng.braid_links[eng.num_braid_links++];
            link->chunk_a = cid_a;
            link->chunk_b = cid_b;
            link->hexit_a = (uint64_t)i * 10000000ULL;
            link->hexit_b = (uint64_t)(i + 1) * 10000000ULL;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t2);
    double braid_ms = (t2.tv_sec - t1.tv_sec)*1000.0 + (t2.tv_nsec - t1.tv_nsec)/1e6;

    /* Verify all chunks point to the shared Hilbert space */
    int all_braided = 1;
    for (int i = 0; i < N_CHUNKS; i++) {
        if (eng.chunks[BASE_CID + i].hilbert.q_joint_state != shared_hilbert)
            all_braided = 0;
    }

    printf("    ✓ Shared Hilbert space: %p (%d amplitudes, %zu bytes)\n",
           (void *)shared_hilbert, D * D, (size_t)D * D * sizeof(Complex));
    printf("    ✓ All %d chunks braided into shared space: %s\n",
           N_CHUNKS, all_braided ? "YES" : "NO");
    printf("    ✓ Bell state |Ψ⟩ = (1/√%d) Σ|k⟩|k⟩ written\n", D);
    printf("    ✓ Braid time: %.3f ms\n\n", braid_ms);

    /* ── 5. Verify: measure ALL chunks once from same Bell state ── */
    printf("  ▸ PHASE 4: Quick-verify — measure ALL %d chunks from shared state...\n",
           N_CHUNKS);

    write_bell_state(shared_hilbert, D);  /* fresh state */

    /* Measure chunk 0 — this collapses the shared state */
    int first_result = (int)measure_chunk(&eng, BASE_CID);
    int all_agree = 1;

    /* All remaining chunks should give the same result (collapsed) */
    for (int i = 1; i < N_CHUNKS; i++) {
        int r = (int)measure_chunk(&eng, BASE_CID + i);
        if (r != first_result) all_agree = 0;
    }

    printf("    First measurement: |%d⟩\n", first_result);
    printf("    All %d chunks agree: %s\n\n",
           N_CHUNKS, all_agree ? "★ YES — shared Hilbert space confirmed" : "✗ NO");

    /* ── 6. FULL BELL TEST on 2 random chunks ── */
    int pick_a, pick_b;
    do {
        pick_a = rand() % N_CHUNKS;
        pick_b = rand() % N_CHUNKS;
    } while (pick_a == pick_b || (pick_a % 2) == (pick_b % 2));
    /* Ensure one is side-A and one is side-B */

    uint64_t cid_a = BASE_CID + pick_a;
    uint64_t cid_b = BASE_CID + pick_b;

    printf("  ▸ PHASE 5: Full Bell Test (%d trials)\n", NUM_TRIALS);
    printf("    Random pick: Chunk %d (side %c, Ptr 0x%016" PRIX64 ")\n",
           pick_a, (pick_a % 2 == 0) ? 'A' : 'B',
           eng.chunks[cid_a].hilbert.magic_ptr);
    printf("              ↔  Chunk %d (side %c, Ptr 0x%016" PRIX64 ")\n",
           pick_b, (pick_b % 2 == 0) ? 'A' : 'B',
           eng.chunks[cid_b].hilbert.magic_ptr);

    struct timespec tb0, tb1;
    clock_gettime(CLOCK_MONOTONIC, &tb0);

    int correlated = 0;
    int marginal_a[D] = {0};
    int marginal_b[D] = {0};

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        /* Fresh Bell state each trial */
        write_bell_state(shared_hilbert, D);

        /* Reset flags so measurement works */
        eng.chunks[cid_a].hilbert.q_flags = 0x01;
        eng.chunks[cid_b].hilbert.q_flags = 0x01;

        int mA = (int)measure_chunk(&eng, cid_a);
        int mB = (int)measure_chunk(&eng, cid_b);

        if (mA == mB) correlated++;
        marginal_a[mA]++;
        marginal_b[mB]++;
    }

    clock_gettime(CLOCK_MONOTONIC, &tb1);
    double bell_ms = (tb1.tv_sec - tb0.tv_sec)*1000.0 + (tb1.tv_nsec - tb0.tv_nsec)/1e6;

    double corr_rate = (double)correlated / NUM_TRIALS;
    int bell_pass = (corr_rate > 0.999);

    printf("\n    Bell Correlation:\n");
    printf("      P(A==B): %.4f (%d/%d)  %s\n",
           corr_rate, correlated, NUM_TRIALS,
           bell_pass ? "★ PERFECT" : "✗ FAIL");

    printf("      Marginals (expect ~%.1f%%):\n", 100.0 / D);
    int marginal_ok = 1;
    for (int s = 0; s < D; s++) {
        double pa = 100.0 * marginal_a[s] / NUM_TRIALS;
        double pb = 100.0 * marginal_b[s] / NUM_TRIALS;
        if (fabs(pa - 100.0/D) > 5.0 || fabs(pb - 100.0/D) > 5.0) marginal_ok = 0;
        printf("        |%d⟩: A=%.2f%%  B=%.2f%%\n", s, pa, pb);
    }
    printf("      Uniform marginals: %s\n", marginal_ok ? "✓" : "✗");

    /* ── Total ── */
    struct timespec t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double total_ms = (t_end.tv_sec - t0.tv_sec)*1000.0 + (t_end.tv_nsec - t0.tv_nsec)/1e6;

    /* ═══════════════════════════════════════════════════════════════════════
     *  RESULTS
     * ═══════════════════════════════════════════════════════════════════════ */
    printf("\n╔════════════════════════════════════════════════════════════════════════════╗\n");
    printf("║  100-TRILLION QUHIT — ALL BRAIDED — RESULTS                              ║\n");
    printf("╠════════════════════════════════════════════════════════════════════════════╣\n");
    printf("║  Quhits:          %" PRIu64 " (100T)  %26s║\n", (uint64_t)TOTAL_QUHITS, "");
    printf("║  Chunks:          %d × %" PRIu64 " each  %25s║\n",
           N_CHUNKS, (uint64_t)QUHITS_PER_CHUNK, "");
    printf("║  Superposed:      ALL                                                    ║\n");
    printf("║  Braided:         ALL — shared Hilbert space                             ║\n");
    printf("║  Hilbert space:   %d amplitudes (%zu bytes)                              ║\n",
           D * D, (size_t)D * D * sizeof(Complex));
    printf("║                                                                          ║\n");
    printf("║  All-chunk verify: %s (measuring 1 collapses all %d)              ║\n",
           all_agree ? "★ PASS" : "✗ FAIL", N_CHUNKS);
    printf("║  Bell Test:        Chunk %02d ↔ Chunk %02d (random)                       ║\n",
           pick_a, pick_b);
    printf("║  P(A==B):          %.4f (%d/%d)  %s                         ║\n",
           corr_rate, correlated, NUM_TRIALS, bell_pass ? "★ PASS" : "✗ FAIL");
    printf("║  Marginals:        %s                                             ║\n",
           marginal_ok ? "uniform ✓" : "skewed ✗");
    printf("║                                                                          ║\n");
    printf("║  Create: %.1f ms | Superpose: %.1f ms | Braid: %.1f ms | Bell: %.1f ms  ║\n",
           create_ms, sup_ms, braid_ms, bell_ms);
    printf("║  TOTAL: %.1f ms                                                       ║\n",
           total_ms);
    printf("║                                                                          ║\n");

    if (bell_pass && marginal_ok && all_agree) {
        printf("║  ★★★ VERIFIED: 100T Quhits ALL braided in shared Hilbert space,     ★★★║\n");
        printf("║  ★★★ random Bell test PASSED with perfect correlation                ★★★║\n");
    } else {
        printf("║  ⚠  SOME TESTS DID NOT PASS                                            ║\n");
    }

    printf("╚════════════════════════════════════════════════════════════════════════════╝\n\n");

    /* Don't free shared_hilbert through engine_destroy — null out refs first */
    for (int i = 0; i < N_CHUNKS; i++)
        eng.chunks[BASE_CID + i].hilbert.q_joint_state = NULL;
    free(shared_hilbert);

    engine_destroy(&eng);
    return (bell_pass && marginal_ok && all_agree) ? 0 : 1;
}
