/*
 * ct.c - Exact distribution computation
 * Usage: ./ct_c -N <N> -n <n> [-o <output_dir>]
 *
 * Requires: GMP, OpenMP
 * Build:    make
 *
 * Memory: ~2 * (N+1)^2 * (2N+2) * sizeof(mpz_t)
 *   N=100 -> ~130 MB, N=200 -> ~520 MB, N=300 -> ~1.7 GB
 *
 * Speed tip: run with jemalloc/mimalloc to reduce GMP allocator contention:
 *   LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 ./ct_c -N 100 -n 50
 *
 * Optimizations:
 *   1. mpz_addmul_ui / per-thread mpz acc  -- fast ulong path; for general
 *      path, per-thread reusable temp avoids malloc/free inside mpz_addmul
 *   2. Range tracking per (s,l)            -- iterate only non-zero dtv slice
 *   3. Hoist rlo/rhi out of the pd loop   -- one update per pl, not per pd;
 *      also replaces flat() per pd with base-pointer + index arithmetic
 *   4. Remove redundant mpz_sgn check     -- GMP handles zeros internally;
 *      double-checking adds branch pressure to the hot path
 *   5. LPT work ordering (descending ps)  -- heaviest tasks first, near-
 *      optimal load balance with dynamic,1
 *   6. Parallel clear loop                -- O(N^2) serial work -> parallel
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <gmp.h>
#include <omp.h>
#include <sys/stat.h>
#include <getopt.h>

#define DEFAULT_OUTPUT_DIR "../data/exact_distributions/"

static int  N_val;
static long stride_l, stride_s;

static inline long flat(int s, int l, int d) {
    return (long)s * stride_s + (long)l * stride_l + d;
}

/* Non-zero dtv range for a given (sum, last) pair. lo > hi means empty. */
typedef struct { int lo, hi; } Range;

static void save_distribution(const char *path, mpz_t *arr) {
    int N = N_val;
    int max_dtv = 0;
    for (int last = 0; last <= N; last++)
        for (int dtv = 2 * N; dtv >= 0; dtv--)
            if (mpz_sgn(arr[flat(N, last, dtv)]) != 0) {
                if (dtv > max_dtv) max_dtv = dtv;
                break;
            }

    mpz_t *dtv_sum = malloc((max_dtv + 1) * sizeof(mpz_t));
    for (int d = 0; d <= max_dtv; d++) mpz_init(dtv_sum[d]);

    for (int last = 0; last <= N; last++)
        for (int dtv = 0; dtv <= max_dtv; dtv++)
            mpz_add(dtv_sum[dtv], dtv_sum[dtv], arr[flat(N, last, dtv)]);

    FILE *fp = fopen(path, "w");
    if (!fp) { fprintf(stderr, "Cannot write %s\n", path); goto cleanup; }
    for (int dtv = 0; dtv <= max_dtv; dtv++)
        gmp_fprintf(fp, "%d %Zd\n", dtv, dtv_sum[dtv]);
    fclose(fp);

cleanup:
    for (int d = 0; d <= max_dtv; d++) mpz_clear(dtv_sum[d]);
    free(dtv_sum);
}

static int file_exists(const char *path) {
    FILE *f = fopen(path, "r");
    if (f) { fclose(f); return 1; }
    return 0;
}

static void run_for_N(int N, int n, const char *output_dir) {
    N_val    = N;
    stride_l = 2 * N + 2;
    stride_s = (long)(N + 1) * stride_l;
    long total = (long)(N + 1) * stride_s;

    /* Precompute binom[r*(N+1)+k] = C(r, k) */
    int bs = N + 1;
    mpz_t *binom = malloc((long)bs * bs * sizeof(mpz_t));
    for (long i = 0; i < (long)bs * bs; i++) mpz_init(binom[i]);
    for (int r = 0; r <= N; r++)
        for (int k = 0; k <= r; k++)
            mpz_bin_uiui(binom[r * bs + k], r, k);

    /* Cache whether each binom fits in unsigned long */
    unsigned long *binom_ul   = malloc((long)bs * bs * sizeof(unsigned long));
    int           *binom_fits = malloc((long)bs * bs * sizeof(int));
    for (int r = 0; r <= N; r++) {
        for (int k = 0; k <= r; k++) {
            long idx = r * bs + k;
            binom_fits[idx] = mpz_fits_ulong_p(binom[idx]);
            binom_ul[idx]   = binom_fits[idx] ? mpz_get_ui(binom[idx]) : 0;
        }
    }

    /* Allocate cur and prv arrays */
    mpz_t *cur = malloc(total * sizeof(mpz_t));
    mpz_t *prv = malloc(total * sizeof(mpz_t));
    if (!cur || !prv) {
        fprintf(stderr, "OOM: need ~%ld MB per array (N=%d)\n",
                total * (long)sizeof(mpz_t) / (1024*1024), N);
        exit(1);
    }
    for (long i = 0; i < total; i++) { mpz_init(cur[i]); mpz_init(prv[i]); }

    /* Range arrays: rng[s*(N+1)+l] */
    int rng_n = (N + 1) * (N + 1);
    Range *cur_rng = malloc(rng_n * sizeof(Range));
    Range *prv_rng = malloc(rng_n * sizeof(Range));
    for (int i = 0; i < rng_n; i++) {
        cur_rng[i].lo = INT_MAX; cur_rng[i].hi = -1;
        prv_rng[i].lo = INT_MAX; prv_rng[i].hi = -1;
    }

    /* Init: cur[i][i][0] = C(N, i) */
    for (int i = 0; i <= N; i++) {
        mpz_bin_uiui(cur[flat(i, i, 0)], N, i);
        cur_rng[i * (N+1) + i].lo = cur_rng[i * (N+1) + i].hi = 0;
    }

    /*
     * Build flat (ns, nl) work list in LPT order: descending ps = ns-nl.
     * Heaviest tasks (large ps) are dispatched first, giving near-optimal
     * load balance under dynamic,1 scheduling.
     */
    int n_work = (N + 1) * (N + 2) / 2;
    int *work_ns = malloc(n_work * sizeof(int));
    int *work_nl = malloc(n_work * sizeof(int));
    {
        int w = 0;
        for (int ps = N; ps >= 0; ps--)       /* descending ps = LPT */
            for (int nl = 0; nl <= N - ps; nl++) {
                work_ns[w] = ps + nl;
                work_nl[w] = nl;
                w++;
            }
    }

    char path[4096];
    for (int step = 0; step < n; step++) {
        snprintf(path, sizeof(path), "%s/N_%d_n_%d.txt", output_dir, N, step + 1);
        if (!file_exists(path))
            save_distribution(path, cur);

        if (step == n - 1) break;

        fprintf(stderr, "N=%d step %d/%d\n", N, step + 1, n - 1);

        /* Swap cur/prv and their range arrays */
        { mpz_t *t = cur; cur = prv; prv = t; }
        { Range *t = cur_rng; cur_rng = prv_rng; prv_rng = t; }

        /*
         * Parallel clear: zero cur (= old prv) only over its previous ranges.
         * Each work item owns a unique (s,l), so no races on rng reset.
         */
        #pragma omp parallel for schedule(dynamic, 8)
        for (int w = 0; w < n_work; w++) {
            int s = work_ns[w], l = work_nl[w];
            Range *r = &cur_rng[s * (N+1) + l];
            if (r->lo <= r->hi) {
                mpz_t *base = &cur[flat(s, l, r->lo)];
                int len = r->hi - r->lo + 1;
                for (int i = 0; i < len; i++)
                    mpz_set_ui(base[i], 0);
                r->lo = INT_MAX; r->hi = -1;
            }
        }

        /*
         * Transition: parallel over (ns, nl) work items.
         *
         * Each task writes exclusively to cur[ns][nl][*] -> no races.
         *
         * Per-thread acc reuses GMP limb storage across iterations, avoiding
         * the malloc/free that mpz_addmul does internally on every call.
         *
         * rlo/rhi are hoisted out of the pd loop: computed once per (pl)
         * from the range endpoints, not conditionally per element.
         *
         * src/dst base pointers replace flat() per pd iteration.
         */
        #pragma omp parallel
        {
            mpz_t acc;
            mpz_init(acc);

            #pragma omp for schedule(dynamic, 1)
            for (int w = 0; w < n_work; w++) {
                int ns = work_ns[w];
                int nl = work_nl[w];
                int ps  = ns - nl;
                int rem = N - ps;
                long bidx     = (long)rem * bs + nl;
                int  cb_fits  = binom_fits[bidx];
                unsigned long cb_ul = binom_ul[bidx];
                const mpz_t  *cb   = &binom[bidx];

                int rlo = INT_MAX, rhi = -1;

                for (int pl = 0; pl <= ps; pl++) {
                    Range r = prv_rng[ps * (N+1) + pl];
                    if (r.lo > r.hi) continue;

                    int diff = nl - pl;
                    if (diff < 0) diff = -diff;

                    /* Hoist range update: one update per pl, not per pd */
                    int nd_lo = r.lo + diff;
                    int nd_hi = r.hi + diff;
                    if (nd_lo < rlo) rlo = nd_lo;
                    if (nd_hi > rhi) rhi = nd_hi;

                    /* Base pointers: stride-1 access, no flat() per iteration */
                    mpz_t       *src      = &prv[flat(ps, pl, r.lo)];
                    mpz_t       *dst_base = &cur[flat(ns, nl, nd_lo)];
                    int          len      = r.hi - r.lo + 1;

                    if (cb_fits) {
                        for (int i = 0; i < len; i++)
                            mpz_addmul_ui(dst_base[i], src[i], cb_ul);
                    } else {
                        for (int i = 0; i < len; i++) {
                            mpz_mul(acc, src[i], *cb);
                            mpz_add(dst_base[i], dst_base[i], acc);
                        }
                    }
                }

                if (rlo <= rhi) {
                    cur_rng[ns * (N+1) + nl].lo = rlo;
                    cur_rng[ns * (N+1) + nl].hi = rhi;
                }
            }

            mpz_clear(acc);
        }
    }

    /* Cleanup */
    for (long i = 0; i < total; i++) { mpz_clear(cur[i]); mpz_clear(prv[i]); }
    free(cur); free(prv);
    for (long i = 0; i < (long)bs * bs; i++) mpz_clear(binom[i]);
    free(binom); free(binom_ul); free(binom_fits);
    free(cur_rng); free(prv_rng);
    free(work_ns); free(work_nl);
}

int main(int argc, char **argv) {
    int N = -1, n = -1, u_flag = 0, n_threads = 0, l_start = 1;
    const char *output_dir = DEFAULT_OUTPUT_DIR;

    int opt;
    while ((opt = getopt(argc, argv, "N:n:o:ut:l:")) != -1) {
        switch (opt) {
        case 'N': N = atoi(optarg); break;
        case 'n': n = atoi(optarg); break;
        case 'o': output_dir = optarg; break;
        case 'u': u_flag = 1; break;
        case 't': n_threads = atoi(optarg); break;
        case 'l': l_start = atoi(optarg); break;
        default:
            fprintf(stderr, "Usage: %s -N <N> -n <n> [-o <dir>] [-u] [-l <start>] [-t <threads>]\n", argv[0]);
            return 1;
        }
    }
    if (N < 0 || n < 0) {
        fprintf(stderr, "Usage: %s -N <N> -n <n> [-o <dir>] [-u] [-l <start>] [-t <threads>]\n", argv[0]);
        return 1;
    }
    if (n_threads > 0)
        omp_set_num_threads(n_threads);

    mkdir(output_dir, 0755);
    int N_start = u_flag ? l_start : N;
    for (int Ni = N_start; Ni <= N; Ni++)
        run_for_N(Ni, n, output_dir);
    return 0;
}
