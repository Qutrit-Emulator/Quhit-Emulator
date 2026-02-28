/*
 * quhit_gates.c — Quantum Gate Operations
 *
 * DFT₆, CZ, arbitrary unitaries, phase gates, X, Z.
 * CZ on entangled pairs applies ω^(a·b) directly to the 36 joint amplitudes.
 * CZ on unentangled quhits auto-creates a product pair first.
 */

#include "quhit_engine.h"

/* ═══════════════════════════════════════════════════════════════════════════════
 * DFT₆ (Generalized Hadamard)
 *
 * Uses the precomputed hex-exact twiddle table from superposition.h.
 * No cos/sin at runtime.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_apply_dft(QuhitEngine *eng, uint32_t id)
{
    if (id >= eng->num_quhits) return;
    Quhit *q = &eng->quhits[id];

    /* If entangled, apply DFT to our side of the joint state */
    if (q->pair_id >= 0) {
        QuhitPair *p = &eng->pairs[q->pair_id];
        uint8_t side = q->pair_side;

        /* DFT on one subsystem of a joint state:
         * ψ'(j, b) = Σ_k DFT[j][k] × ψ(k, b)   (side 0)
         * ψ'(a, j) = Σ_k DFT[j][k] × ψ(a, k)   (side 1) */
        double new_re[QUHIT_D2], new_im[QUHIT_D2];
        memset(new_re, 0, sizeof(new_re));
        memset(new_im, 0, sizeof(new_im));

        for (int j = 0; j < QUHIT_D; j++) {
            for (int partner = 0; partner < QUHIT_D; partner++) {
                double acc_re = 0, acc_im = 0;
                for (int k = 0; k < QUHIT_D; k++) {
                    int src_idx;
                    if (side == 0) {
                        src_idx = k * QUHIT_D + partner;
                    } else {
                        src_idx = partner * QUHIT_D + k;
                    }
                    /* DFT6[j][k] × ψ[src] */
                    double tw_re = DFT6[j][k].re;
                    double tw_im = DFT6[j][k].im;
                    double s_re  = p->joint.re[src_idx];
                    double s_im  = p->joint.im[src_idx];
                    acc_re += tw_re * s_re - tw_im * s_im;
                    acc_im += tw_re * s_im + tw_im * s_re;
                }
                int dst_idx = (side == 0) ? j * QUHIT_D + partner
                                          : partner * QUHIT_D + j;
                new_re[dst_idx] = acc_re;
                new_im[dst_idx] = acc_im;
            }
        }

        memcpy(p->joint.re, new_re, sizeof(new_re));
        memcpy(p->joint.im, new_im, sizeof(new_im));
        return;
    }

    /* Local DFT: uses superposition.h precomputed table */
    sup_apply_dft6(q->state.re, q->state.im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * INVERSE DFT₆
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_apply_idft(QuhitEngine *eng, uint32_t id)
{
    if (id >= eng->num_quhits) return;
    Quhit *q = &eng->quhits[id];

    if (q->pair_id >= 0) {
        QuhitPair *p = &eng->pairs[q->pair_id];
        uint8_t side = q->pair_side;

        double new_re[QUHIT_D2], new_im[QUHIT_D2];
        memset(new_re, 0, sizeof(new_re));
        memset(new_im, 0, sizeof(new_im));

        for (int j = 0; j < QUHIT_D; j++) {
            for (int partner = 0; partner < QUHIT_D; partner++) {
                double acc_re = 0, acc_im = 0;
                for (int k = 0; k < QUHIT_D; k++) {
                    int src_idx = (side == 0) ? k * QUHIT_D + partner
                                              : partner * QUHIT_D + k;
                    /* IDFT: conjugate twiddles, swap indices */
                    double tw_re =  DFT6[k][j].re;
                    double tw_im = -DFT6[k][j].im;  /* conjugate */
                    double s_re  = p->joint.re[src_idx];
                    double s_im  = p->joint.im[src_idx];
                    acc_re += tw_re * s_re - tw_im * s_im;
                    acc_im += tw_re * s_im + tw_im * s_re;
                }
                int dst_idx = (side == 0) ? j * QUHIT_D + partner
                                          : partner * QUHIT_D + j;
                new_re[dst_idx] = acc_re;
                new_im[dst_idx] = acc_im;
            }
        }

        memcpy(p->joint.re, new_re, sizeof(new_re));
        memcpy(p->joint.im, new_im, sizeof(new_im));
        return;
    }

    sup_apply_idft6(q->state.re, q->state.im);
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * CZ GATE — Controlled-Z: |a,b⟩ → ω^(a·b) |a,b⟩
 *
 * If quhits are already entangled (share a pair), apply ω^(a·b) phases
 * directly to the 36 joint amplitudes. O(D²) = O(36).
 *
 * If quhits are NOT entangled, auto-create a product pair from their
 * current local states, then apply CZ. This is the key operation that
 * creates entanglement from product states.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_apply_cz(QuhitEngine *eng, uint32_t id_a, uint32_t id_b)
{
    if (id_a >= eng->num_quhits || id_b >= eng->num_quhits) return;
    Quhit *qa = &eng->quhits[id_a];
    Quhit *qb = &eng->quhits[id_b];

    /* If not already paired, create product pair */
    if (qa->pair_id < 0 || qa->pair_id != qb->pair_id) {
        /* Disentangle each from any existing pairs first */
        if (qa->pair_id >= 0) {
            QuhitPair *old = &eng->pairs[qa->pair_id];
            uint32_t partner = (qa->pair_side == 0) ? old->id_b : old->id_a;
            quhit_disentangle(eng, id_a, partner);
        }
        if (qb->pair_id >= 0) {
            QuhitPair *old = &eng->pairs[qb->pair_id];
            uint32_t partner = (qb->pair_side == 0) ? old->id_b : old->id_a;
            quhit_disentangle(eng, id_b, partner);
        }
        quhit_entangle_product(eng, id_a, id_b);
    }

    /* Now both share a pair — apply ω^(a·b) phases using precomputed table */
    QuhitPair *p = &eng->pairs[qa->pair_id];

    for (int a = 0; a < QUHIT_D; a++) {
        for (int b = 0; b < QUHIT_D; b++) {
            int idx = a * QUHIT_D + b;
            double cos_p = CZ_PHASE[a][b].re;
            double sin_p = CZ_PHASE[a][b].im;
            double re = p->joint.re[idx];
            double im = p->joint.im[idx];
            p->joint.re[idx] = re * cos_p - im * sin_p;
            p->joint.im[idx] = re * sin_p + im * cos_p;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * ARBITRARY D×D UNITARY — U|ψ⟩
 *
 * U is given as two D×D arrays (re, im), row-major.
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_apply_unitary(QuhitEngine *eng, uint32_t id,
                         const double *U_re, const double *U_im)
{
    if (id >= eng->num_quhits) return;
    Quhit *q = &eng->quhits[id];

    if (q->pair_id >= 0) {
        /* Apply unitary to our subsystem within the joint state */
        QuhitPair *p = &eng->pairs[q->pair_id];
        uint8_t side = q->pair_side;

        double new_re[QUHIT_D2], new_im[QUHIT_D2];
        memset(new_re, 0, sizeof(new_re));
        memset(new_im, 0, sizeof(new_im));

        for (int j = 0; j < QUHIT_D; j++) {
            for (int partner = 0; partner < QUHIT_D; partner++) {
                double acc_re = 0, acc_im = 0;
                for (int k = 0; k < QUHIT_D; k++) {
                    int src_idx = (side == 0) ? k * QUHIT_D + partner
                                              : partner * QUHIT_D + k;
                    double u_re = U_re[j * QUHIT_D + k];
                    double u_im = U_im[j * QUHIT_D + k];
                    double s_re = p->joint.re[src_idx];
                    double s_im = p->joint.im[src_idx];
                    acc_re += u_re * s_re - u_im * s_im;
                    acc_im += u_re * s_im + u_im * s_re;
                }
                int dst_idx = (side == 0) ? j * QUHIT_D + partner
                                          : partner * QUHIT_D + j;
                new_re[dst_idx] = acc_re;
                new_im[dst_idx] = acc_im;
            }
        }

        memcpy(p->joint.re, new_re, sizeof(new_re));
        memcpy(p->joint.im, new_im, sizeof(new_im));
        return;
    }

    /* Local unitary */
    double new_re[QUHIT_D] = {0}, new_im[QUHIT_D] = {0};
    for (int j = 0; j < QUHIT_D; j++) {
        for (int k = 0; k < QUHIT_D; k++) {
            double u_re = U_re[j * QUHIT_D + k];
            double u_im = U_im[j * QUHIT_D + k];
            new_re[j] += u_re * q->state.re[k] - u_im * q->state.im[k];
            new_im[j] += u_re * q->state.im[k] + u_im * q->state.re[k];
        }
    }
    memcpy(q->state.re, new_re, sizeof(new_re));
    memcpy(q->state.im, new_im, sizeof(new_im));
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * PHASE GATE — Diagonal: |k⟩ → e^(i·phases[k]) |k⟩
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_apply_phase(QuhitEngine *eng, uint32_t id, const double *phases)
{
    if (id >= eng->num_quhits) return;
    Quhit *q = &eng->quhits[id];

    /* Phase gates always apply to local state (diagonal = no entanglement change) */
    for (int k = 0; k < QUHIT_D; k++) {
        double cos_p = cos(phases[k]), sin_p = sin(phases[k]);
        double re = q->state.re[k], im = q->state.im[k];
        q->state.re[k] = re * cos_p - im * sin_p;
        q->state.im[k] = re * sin_p + im * cos_p;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * X GATE — Cyclic shift: |k⟩ → |k+1 mod D⟩
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_apply_x(QuhitEngine *eng, uint32_t id)
{
    if (id >= eng->num_quhits) return;
    QuhitState *s = &eng->quhits[id].state;

    double last_re = s->re[QUHIT_D - 1];
    double last_im = s->im[QUHIT_D - 1];
    for (int k = QUHIT_D - 1; k > 0; k--) {
        s->re[k] = s->re[k - 1];
        s->im[k] = s->im[k - 1];
    }
    s->re[0] = last_re;
    s->im[0] = last_im;
}

/* ═══════════════════════════════════════════════════════════════════════════════
 * Z GATE — Phase gate: |k⟩ → ω^k |k⟩
 * ═══════════════════════════════════════════════════════════════════════════════ */

void quhit_apply_z(QuhitEngine *eng, uint32_t id)
{
    if (id >= eng->num_quhits) return;
    QuhitState *s = &eng->quhits[id].state;

    for (int k = 0; k < QUHIT_D; k++) {
        double re = s->re[k], im = s->im[k];
        /* ω^k where ω = OMEGA6[1] */
        double w_re = OMEGA6[k % QUHIT_D].re;
        double w_im = OMEGA6[k % QUHIT_D].im;
        s->re[k] = re * w_re - im * w_im;
        s->im[k] = re * w_im + im * w_re;
    }
}
