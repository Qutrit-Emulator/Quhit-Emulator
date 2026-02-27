/*
 * hexagram_sparse.c — Sparse Hexagram Register
 *
 * DERIVED BY GAUSS SUM ANALYSIS:
 * ════════════════════════════════════════════════════════════════════
 * Circuit: DFT₆^⊗N → CZ(0,1) → CZ(1,2) → ... → CZ(N-2,N-1) → DFT₆^⊗N
 * Starting from |0...0⟩.
 *
 * Amplitude at output state |j₀,...,j_{N-1}⟩:
 *
 *   A(j) = (1/6)^N Σ_{k ∈ Z₆^N} ω₆^{Σ_i k_i·k_{i+1} + Σ_i k_i·j_i}
 *
 * Evaluated by integrating the quadratic Gauss sum from right inward:
 *
 *   Σ_{k_{N-1}} ω₆^{k_{N-1}(k_{N-2}+j_{N-1})} = 6·δ(k_{N-2}=-j_{N-1})
 *
 * This creates a recursion that determines all forced k values:
 *   k_{N-2} = -j_{N-1}
 *   k_{N-4} = -k_{N-2} - j_{N-3} = j_{N-1} - j_{N-3}
 *   k_{N-6} = -k_{N-4} - j_{N-5} = -j_{N-1} + j_{N-3} - j_{N-5}
 *   ...  (alternating signs from right)
 *
 * General formula for the forced values at even positions:
 *   k_{N-2-2m} = Σ_{l=0}^{m} (-1)^l j_{N-1-2l}   (alternating sum from right)
 *
 * EVEN N=2M: All N values k_0,...,k_{N-1} are determined (no free sum).
 *   The final "sum" over k_0 just evaluates ω₆^{k_0·j_0} at the forced value.
 *   Result: ALL entries nonzero, magnitude = 1/6^M.
 *   Phase = k_0·j_0 + k_2·j_2 + ... + accumulated correction.
 *   (dense, no compression benefit from this formula alone)
 *
 * ODD N=2M+1: After fixing k_1,k_3,...,k_{2M-1} from right-side sums,
 *   the final sum Σ_{k_0} ω₆^{k_0(k_1+j_0)} = 6·δ(j_0 + k_1 = 0).
 *   This gives a HARD CONSTRAINT on the j variables:
 *   j_0 = -k_1 = j_N-1 - j_{N-3} + j_{N-5} - ... (alternating from right for M terms)
 *
 * EXACT CLOSED-FORM (verified for N=3,5):
 *   N=3: A(j) = (1/6)·δ(j_0=j_2)·ω₆^{-j_1·j_2}
 *   N=5: A(j) = (1/36)·δ(j_2=(j_0+j_4)%6)·ω₆^{-j_0j_1-j_3j_4}
 *
 * GENERAL ODD N=2M+1:
 *   The constraint is: j_M = Σ_{m=0}^{M-1} (-1)^m j_{N-1-2m}    [i.e. alternating sum from right]
 *   The phase is:      φ = -(j_0·j_1 + j_2·j_3 + ... + j_{M-1}·j_M)  [adjacent pair products]
 *   Wait -- this needs verification for N=7. Let me derive:
 *
 *   N=7: CZ(0,1)+...+CZ(5,6)
 *   Sum k_6: k_5 = -j_6
 *   Sum k_4: k_3 = -k_5 - j_4 = j_6 - j_4, phase += k_5·j_5 = -j_6·j_5 ← NO WAIT
 *
 *   Actually the phase accumulates from the j-values of the FIXED k's:
 *   When k_{N-2} = -j_{N-1} is forced, the term k_{N-2}·j_{N-2} = -j_{N-1}·j_{N-2} adds to phase.
 *   When k_{N-4} = j_{N-1}-j_{N-3} is forced, adds (j_{N-1}-j_{N-3})·j_{N-4}.
 *   ...
 *   Final constraint forces k_0 via δ, adding k_0·j_0 = (-k_1)·j_0 = -k_1·j_0 to phase.
 *
 * For N=3: only k_1 = -j_2 is forced, adds -j_2·j_1. Constraint k_0=-j_2→j_0=j_2. DONE.
 * For N=5:
 *   k_3 = -j_4, phase += -j_4·j_3
 *   k_1 = j_4-j_2, phase += (j_4-j_2)·j_1 = j_4j_1-j_2j_1
 *   Constraint: k_0 = -(k_1+j_0) → j_0 = -k_1 = j_2-j_4 (mod 6)... wait, gives j_2=j_0+j_4 ✓
 *   phase contribution from k_0: k_0·j_0 = (j_2-j_4)·j_0... Hmm, but j_0 is just the index,
 *   the k_0 sum gives δ not a phase contribution.
 *   Total phase: -j_4·j_3 + (j_4-j_2)·j_1. With j_2=j_0+j_4:
 *   = -j_4j_3 + j_4j_1 - (j_0+j_4)j_1 = -j_4j_3 + j_4j_1 - j_0j_1 - j_4j_1 = -j_0j_1 - j_4j_3 ✓
 *
 * GENERAL PATTERN (for odd N=2M+1, CZ chain with M constraints):
 *   Free parameters: j_1, j_3, ..., j_{2M-1} (the M "odd-indexed" values)
 *                    j_0 (the leftmost) — but constrained!
 *                    j_2, j_4, ..., j_{2M-2} (the inner "even" values), all free
 *                    j_{2M} = j_{N-1} (the rightmost) — constrained by j_0 and middle values
 *   ACTUALLY: the free parameters for storage are the independent components:
 *   For odd N: { j_odd = j_1,j_3,...,j_{2M-1} } (M values) × { j_even_left to j_{N-3} } × constraint
 *
 *   Simpler: just store the FREE variables, compute constrained ones from formula.
 *
 * ════════════════════════════════════════════════════════════════════
 * SPARSE REGISTER IMPLEMENTATION
 *
 * For odd N=2M+1, store the FREE j-variables as a sparse array.
 * The free variables are: j_1, j_3, ..., j_{2M-1} (M odd-indexed)
 * plus j_0, j_2, ..., j_{2M-2} (M even-indexed, left half) — total 2M free.
 * The constrained variable: j_{2M} = j_{N-1} is determined by the constraint.
 *
 * Constraint (from N=3: j_0=j_2, N=5: j_2=(j_0+j_4)%6):
 * For general odd N=2M+1:
 *   j_{N-1} = j_{N-3} - j_{N-5} + ... (alternating from left)
 *
 * Storage: 6^{2M} entries in Z₆ × Z₆ × ... (2M dimensions) — NOT sparse itself
 * BUT we're working in the EDGE SLICE j_i ∈ {0,1,2}, so 3^{2M} entries.
 *
 * For N=3: 2M=2, store 3²=9 entries. Number nonzero: 9/27 that satisfy j_2 ≤ 2.
 *           But all 9 stored entries ARE valid (j_{N-1}=j_0 ∈ {0,1,2} always).
 * For N=5: 2M=4, store 3⁴=81 entries. Constraint j_4=(j_0+j_2... wait no.
 *
 * Hmm, N=5 constraint is j_2=j_0+j_4 — j_2 is the MIDDLE one. So the free vars
 * are { j_0, j_1, j_3, j_4 } and j_2 is constrained. Storage = 3⁴=81 (but 54 valid).
 * Actually nonzero = those with (j_0+j_4)%6 ∈ {0,1,2} → 54 out of 81 (j_0,j_1,j_3,j_4) combos.
 *
 * ════════════════════════════════════════════════════════════════════
 * BOTTOM LINE: The FREE variables for storage are:
 *   j_0, j_1, j_3, j_5, ..., j_{2M-1}, j_{2M}  (all except the middle j_M)
 * Total count: N-1 free variables × range 3 each = 3^{N-1} entries.
 * Valid (nonzero): those where the middle j_M = constraint ∈ {0,1,2}.
 *
 * This gives EXACTLY 3^{N-1} × P(constraint ∈ {0,1,2}) entries.
 * For N=3: 3² × (1/1) = 9. For N=5: 3⁴ × (54/81) = 54. Correct!
 *
 * So the "sparse" register stores 3^{N-1} entries (not 3^N), which is
 * a 3× compression vs the dense 3^N register.
 * Larger compression comes from Gauss sums over larger ranges.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static const double W6R[6]={1,0.5,-0.5,-1,-0.5,0.5};
static const double W6I[6]={0,0.866025403784438647,0.866025403784438647,
                             0,-0.866025403784438647,-0.866025403784438647};

static int d6(int x,int i){for(int j=0;j<i;j++)x/=6;return x%6;}
static int sd6(int x,int i,int v){int p=1;for(int j=0;j<i;j++)p*=6;return x+(v-(x/p)%6)*p;}
static void dft6_ref(double *re,double *im,int dim,int qi){
    double n6=1.0/sqrt(6.0);double *nr=calloc(dim,8),*ni=calloc(dim,8);
    for(int i=0;i<dim;i++){int ki=d6(i,qi);
        for(int j=0;j<6;j++){int t=sd6(i,qi,j);int ph=(ki*j)%6;
            nr[t]+=n6*(W6R[ph]*re[i]-W6I[ph]*im[i]);
            ni[t]+=n6*(W6R[ph]*im[i]+W6I[ph]*re[i]);}}
    memcpy(re,nr,dim*8);memcpy(im,ni,dim*8);free(nr);free(ni);}
static void cz_ref(double *re,double *im,int dim,int qi,int qj){
    for(int i=0;i<dim;i++){int ki=d6(i,qi),kj=d6(i,qj);
        int ph=(ki*kj)%6;if(!ph)continue;double r=re[i],m=im[i];
        re[i]=W6R[ph]*r-W6I[ph]*m;im[i]=W6R[ph]*m+W6I[ph]*r;}}

static int pow6n(int n){int r=1;for(int i=0;i<n;i++)r*=6;return r;}
static int pow3n(int n){int r=1;for(int i=0;i<n;i++)r*=3;return r;}
static void decode6(int x,int N,int *v){for(int i=0;i<N;i++){v[i]=x%6;x/=6;}}
static void decode3(int x,int N,int *v){for(int i=0;i<N;i++){v[i]=x%3;x/=3;}}

/* ═══════════════════════════════════════════════════════════
 * ANALYTIC AMPLITUDE: general formula for the DFT+CZ_chain+DFT circuit.
 *
 * Works for ANY N. Evaluates by simulating the Gauss sum recursion.
 * j[] has N entries in Z₆. Returns amplitude via re/im.
 *
 * Algorithm:
 *   1. Integrate from right: fix k[N-2], k[N-4], ... and accumulate phase.
 *   2. The final integration sum over k[0] (or k[1]) gives a δ constraint.
 *   3. Check constraint; if violated return 0. Else return (1/6)^M × ω₆^{phase}.
 * ═══════════════════════════════════════════════════════════ */
static void analytic_amp_general(const int *j, int N, double *re_out, double *im_out) {
    *re_out = 0; *im_out = 0;
    if (N == 0) { *re_out = 1; return; }
    if (N == 1) { if (j[0]==0) *re_out=1; return; }  /* DFT+DFT = identity on |0⟩ */

    /* Integrate from right: sum over k[N-1], k[N-3], ... 
     * Each sum forces k[N-2], k[N-4], ...
     * Accumulate phase contributions from the forced k values times their j-neighbors. */

    int phase = 0;          /* accumulated phase in Z₆ */
    int  k_forced[64];      /* k_forced[i] = k_i value forced by Gauss sum, -1 if not yet forced */
    for (int i=0;i<N;i++) k_forced[i]=-1;

    /* Right-to-left integration: sum k[N-1] fixes k[N-2], then k[N-3] fixes k[N-4], etc. */
    int num_sums = (N-1)/2;  /* number of Gauss sums from the right */
    int num_constraints;

    /* k[N-2] = -j[N-1] from Σ_{k[N-1]} ω₆^{k[N-1](k[N-2]+j[N-1])} */
    k_forced[N-2] = ((6 - j[N-1]) % 6 + 6) % 6;
    /* Phase contribution: k[N-2] × j[N-2] (the "fixed k × its j-neighbor on the left") */
    phase = (phase + k_forced[N-2] * j[N-2]) % 6;

    /* Continue: k[N-3] forced by k[N-2], fixes k[N-4]:
     * Σ_{k[N-3]} ω₆^{k[N-3](k[N-2]+j[N-3]+k[N-4])}... wait, that's not right.
     * The sum structure: the argument of k[N-3] is k[N-2]*k[N-3] + k[N-3]*k[N-4] + k[N-3]*j[N-3]
     * = k[N-3] × (k[N-2] + k[N-4] + j[N-3])
     * Summing over k[N-3]: δ(k[N-4] = -k[N-2] - j[N-3]) */
    for (int step = 2; step <= N-2; step += 2) {
        int right_forced = k_forced[N-2-(step-2)];  /* k already forced to left of this */
        int new_idx = N-2-step;                      /* k index to fix now */
        k_forced[new_idx] = ((6 - right_forced - j[N-1-step]) % 6 + 6) % 6;
        if (new_idx > 0)
            phase = (phase + k_forced[new_idx] * j[new_idx]) % 6;
        else {
            /* new_idx == 0: this is the final constraint check */
            /* Σ_{k[0]} ω₆^{k[0]×(k[1]+j[0])} = 6·δ(k[1]=-j[0]) → j[0] = -k[1] */
            /* But we forced k[0] = k_forced[0]. The sum over the LAST free variable is k[1]. */
            /* Wait — for even step when new_idx=0: k[0] is forced, but there's no sum left. */
            /* This means k[0] forced = some value, and it MUST equal the value derived from j[0]. */
            /* Specifically: after all sums, the remaining expression is:
             * ω₆^{k_forced[0] × j[0]}  (the last phase factor from the "fixed" k)
             * With NO delta constraint — k[0] is already determined, so we just compute phase. */
            phase = (phase + k_forced[0] * j[0]) % 6;
        }
    }

    /* For ODD N: after (N-1)/2 Gauss sums from right, the last free sum is over k[0].
     * Only applies if N is odd. We integrate from the LEFT in this case:
     * Σ_{k[0]} ω₆^{k[0]×(k[1]+j[0])} = 6·δ(k[1]=-j[0])
     * But k[1] was already forced above! This gives the CONSTRAINT: k_forced[1] = -j[0] mod 6. */
    if (N % 2 == 1) {
        /* Constraint: k_forced[1] must equal -j[0] mod 6 */
        int required_k1 = ((6 - j[0]) % 6 + 6) % 6;
        if (k_forced[1] != required_k1) return;  /* amplitude = 0 */
        /* No additional phase: the δ-sum contributes the 1/6 factor, no phase */
    }

    /* Compute the magnitude: 1/6^{floor(N/2)} */
    double mag = 1.0;
    for (int m=0; m<N/2; m++) mag /= 6.0;

    phase = ((phase % 6) + 6) % 6;
    *re_out = mag * W6R[phase];
    *im_out = mag * W6I[phase];
}

/* ═══════════════════════════════════════════════════════════
 * VERIFICATION: compare analytic formula vs reference
 * ═══════════════════════════════════════════════════════════ */
static void verify(int N) {
    int dim6 = pow6n(N);
    double *re=calloc(dim6,8),*im=calloc(dim6,8); re[0]=1;
    for(int i=0;i<N;i++) dft6_ref(re,im,dim6,i);
    for(int i=0;i<N-1;i++) cz_ref(re,im,dim6,i,i+1);
    for(int i=0;i<N;i++) dft6_ref(re,im,dim6,i);

    int jv[16]; double max_err=0;
    int nonzero_ref=0, nonzero_ana=0;
    for(int j=0;j<dim6;j++){
        decode6(j,N,jv);
        double ref_r=re[j], ref_i=im[j];
        double ana_r, ana_i;
        analytic_amp_general(jv,N,&ana_r,&ana_i);
        double err=sqrt((ref_r-ana_r)*(ref_r-ana_r)+(ref_i-ana_i)*(ref_i-ana_i));
        if(err>max_err) max_err=err;
        if(sqrt(ref_r*ref_r+ref_i*ref_i)>1e-10) nonzero_ref++;
        if(sqrt(ana_r*ana_r+ana_i*ana_i)>1e-10) nonzero_ana++;
    }
    printf("  N=%d: %6d nonzero ref, %6d analytic, max_err=%.2e %s\n",
           N, nonzero_ref, nonzero_ana, max_err,
           max_err<1e-12?"✓ EXACT!":"✗");
    free(re);free(im);
}

/* ═══════════════════════════════════════════════════════════
 * BORN probabilities from analytic formula — no dense storage!
 * For q_target ∈ {0,...,N-1}, compute P(k=0..5) analytically.
 * ═══════════════════════════════════════════════════════════ */
static void born_analytic(int N, int qt, double pr[6]) {
    memset(pr,0,48);
    int jv[64];
    /* Iterate over all j-vectors in Z₆^N; compute amplitude analytically */
    int dim6 = pow6n(N);
    for(int j=0;j<dim6;j++){
        decode6(j,N,jv);
        double ar,ai;
        analytic_amp_general(jv,N,&ar,&ai);
        double m2=ar*ar+ai*ai;
        if(m2<1e-30) continue;
        pr[jv[qt]] += m2;
    }
}

/* Edge-slice Born: the correct Born rule sums |A(j)|² over ALL j ∈ Z₆^N.
 * The analytic formula is zero for most entries (the constraint skips them).
 * We exploit this: iterate j_target ∈ Z₆ and j_others ∈ Z₆^{N-1},
 * calling analytic_amp_general which returns 0 for invalid entries.
 * For K=20, 6^20 is too large. But the constraint means only ~3^N survive.
 * So iterate in Z₃^N for the "others" and also check j_target ∈ Z₆:
 * Actually the simplest correct fix: the full sum gives P(k)=1/6 for uniform circuits.
 * The edge-slice is not a proper Born rule — it's a partial sum.
 * Correct Born = Σ_{j∈Z₆^N, j_qt=k} |A(j)|² = P(k).
 * For uniform |A|²=1/6^N × (# nonzero):
 * For N=2M: all 6^N nonzero, P(k)=6^N/6 × (1/6)^N^2... = (1/6)^M × (1/6)/(1/6) = 1/6.
 * So Born is trivially uniform for this circuit. The Born probs are exactly 1/6 each.
 *
 * The real question is Born probs for GENERAL circuits (not just DFT+CZ+DFT).
 * For a general circuit, we need the full amplitude, not just the edge slice.
 * The analytic Gauss sum formula gives the FULL amplitude at any j ∈ Z₆^N in O(N) time.
 * So Born at large K: iterate j_target ∈ {0..5}, others in constrained Z₆^{N-1} subset.
 */
static void born_analytic_full(int N, int qt, double pr[6]) {
    memset(pr,0,48);
    int jv[64];
    int dim6 = pow6n(N);
    for(int j=0;j<dim6;j++){
        decode6(j,N,jv);
        double ar,ai;
        analytic_amp_general(jv,N,&ar,&ai);
        double m2=ar*ar+ai*ai;
        if(m2<1e-30) continue;
        pr[jv[qt]] += m2;
    }
}

/* Constrained-only Born for ODD N:
 * Free params: j[0], j[1], ..., j[N-2] (all but j[N-1]).
 * j[N-1] = constrained = -k_forced[N-2] = j[N-3] - j[N-5] + ... (alternating)
 * Wait — the constraint is on j[0], not j[N-1].
 * For ODD N: constraint is k_forced[1] = -j[0] mod 6.
 * k_forced[1] is determined by j[2], j[4], ..., j[N-1].
 * So: iterate FREE variables j[1], j[2], ..., j[N-1] (N-1 variables),
 * then compute j[0] = -k_forced[1] mod 6 (the constrained variable).
 * Only include if j[0] ∈ the valid range for this circuit.
 */
static void born_constrained(int N, int qt, double pr[6]) {
    memset(pr,0,48);
    if (N % 2 == 0) {
        /* Even N: ALL entries nonzero, magnitude = 1/6^{N/2}, 6^N entries */
        /* Born is trivially 1/6 each for this uniform-output circuit */
        for(int k=0;k<6;k++) pr[k]=1.0/6.0;
        return;
    }
    /* Odd N: iterate over j[1..N-1] (N-1 free variables in Z₆), compute j[0] from constraint. */
    int jv[64];
    long dim_free = 1; for(int i=0;i<N-1;i++) dim_free*=6;
    for(long fi=0;fi<dim_free;fi++){
        /* Decode j[1..N-1] from fi */
        long tmp = fi;
        for(int i=1;i<N;i++){jv[i]=tmp%6;tmp/=6;}
        
        /* Compute k_forced[1] from j[2], j[4], ..., j[N-1] using the Gauss recursion.
         * From the right: k_forced[N-2] = -j[N-1], k_forced[N-4] = -k_forced[N-2]-j[N-3], ...
         * Until we reach k_forced[1]. */
        int k_cur = ((6-jv[N-1])%6+6)%6;   /* k_forced[N-2] */
        for(int step=2; step<=N-3; step+=2){
            int j_mid = jv[N-1-step];
            k_cur = ((6-k_cur-j_mid)%6+6)%6;
        }
        /* k_cur is now k_forced[1] */
        jv[0] = ((6-k_cur)%6+6)%6;   /* j[0] = -k_cur = constraint result */
        
        /* Now evaluate amplitude (will always be nonzero since constraint satisfied) */
        double ar,ai;
        analytic_amp_general(jv,N,&ar,&ai);
        double m2=ar*ar+ai*ai;
        if(m2<1e-30) continue;
        pr[jv[qt]] += m2;
    }
}

int main(void) {
    printf("\n  ═══ SPARSE HEXAGRAM REGISTER — GAUSS SUM FORMULA ═══\n");
    printf("  A(j) = (1/6)^⌊N/2⌋ × δ(constraint) × ω₆^{phase}\n\n");

    /* Verify N=2..7 */
    printf("  Verification vs brute force:\n");
    for(int N=2;N<=7;N++) verify(N);

    /* Born probs using full analytic formula */
    printf("\n  Born probs (analytic, zero storage):\n");
    for(int N=2;N<=10;N++){
        double pr[6]; born_analytic_full(N,0,pr);
        double total=0; for(int k=0;k<6;k++) total+=pr[k];
        printf("    N=%2d: total=%.6f %s (1/6=%.4f each)\n",
               N,total,fabs(total-1.0)<1e-6?"✓":"✗",1.0/6);
    }

    /* Constrained-only iterator for odd N: O(6^{N-1}) instead of O(6^N) */
    printf("\n  Constrained Born (odd N, skips invalid j-vectors):\n");
    for(int N=3;N<=21;N+=2){
        long dim_free=1;for(int i=0;i<N-1;i++)dim_free*=6;
        double pr[6]; born_constrained(N,0,pr);
        double total=0;for(int k=0;k<6;k++)total+=pr[k];
        printf("    N=%2d: 6^(N-1)=%9ld iterations, total=%.6f %s\n",
               N, dim_free, total, fabs(total-1.0)<1e-6?"✓":"✗");
        if(N>=15) break;  /* stop before 6^15 which is huge */
    }

    return 0;
}
