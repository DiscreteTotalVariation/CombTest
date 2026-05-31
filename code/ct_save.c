/*
 * ct_save.c - Exact distribution computation with checkpoint/resume
 *
 * Usage:
 *   Fresh:  ./ct_save -N <N> -n <n> [-o <dir>] [-u] [-l <start>] [-t <threads>]
 *                     [-s <checkpoint>] [-i <interval>]
 *   Resume: ./ct_save -c <checkpoint> [-s <checkpoint>] [-i <interval>] [-t <threads>]
 *
 * Checkpoint options:
 *   -s <path>    Save checkpoint to this file periodically and on SIGINT/SIGTERM
 *   -i <steps>   Save every <steps> iterations (default: signal-only)
 *   -c <path>    Resume from a saved checkpoint (-N/-n/-o/-u/-l loaded from file)
 *
 * Requires: GMP, OpenMP
 * Build:    make ct_save
 *
 * Memory: ~2 * (N+1)^2 * (2N+2) * sizeof(mpz_t)
 *
 * Optimizations:
 *   1. mpz_addmul_ui / per-thread mpz acc
 *   2. Range tracking per (s,l)
 *   3. Hoist rlo/rhi out of the pd loop
 *   4. Remove redundant mpz_sgn check
 *   5. LPT work ordering (descending ps)
 *   6. Parallel clear loop
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <gmp.h>
#include <omp.h>
#include <sys/stat.h>
#include <getopt.h>
#include <signal.h>
#include <stdint.h>

#define DEFAULT_OUTPUT_DIR "../data/exact_distributions/"
#define CKPT_MAGIC 0x43545356u  /* "CTSV" */

/* --- Graceful stop: first signal sets flag, second terminates --------- */
static volatile sig_atomic_t stop_requested = 0;

static void handle_stop(int sig) {
    stop_requested = 1;
    signal(sig, SIG_DFL);
}

/* --- GMP allocator wrappers ------------------------------------------- */
static void *gmp_alloc_wrapper(size_t sz) {
    void *p = malloc(sz);
    if (!p) {
        fprintf(stderr, "OOM: GMP failed to allocate %zu bytes\n", sz);
        exit(1);
    }
    return p;
}

static void *gmp_realloc_wrapper(void *old, size_t old_sz, size_t new_sz) {
    (void)old_sz;
    void *p = realloc(old, new_sz);
    if (!p) {
        fprintf(stderr, "OOM: GMP failed to reallocate %zu bytes\n", new_sz);
        exit(1);
    }
    return p;
}

static void gmp_free_wrapper(void *p, size_t sz) {
    (void)sz;
    free(p);
}

/* --- Global state for 3D->1D flattening ------------------------------- */
static int  N_val;
static long stride_l, stride_s;

static inline long flat(int s, int l, int d) {
    return (long)s * stride_s + (long)l * stride_l + d;
}

/* Non-zero dtv range for a given (sum, last) pair. lo > hi means empty. */
typedef struct { int lo, hi; } Range;

/* --- Distribution output ---------------------------------------------- */
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
    if (!dtv_sum) { fprintf(stderr, "OOM: dtv_sum (N=%d)\n", N); return; }
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

/*
 * Return the first step whose distribution file is missing, or n if all exist.
 * Step 0 produces N_?_n_1.txt, step k produces N_?_n_{k+1}.txt.
 */
static int first_missing_step(int N, int n, const char *output_dir) {
    char path[4096];
    for (int step = 0; step < n; step++) {
        snprintf(path, sizeof(path), "%s/N_%d_n_%d.txt", output_dir, N, step + 1);
        if (!file_exists(path))
            return step;
    }
    return n;
}

/* === Checkpoint save/load ============================================= */

static int save_checkpoint(const char *path, int N, int n, int current_ni,
                           int step, int u_flag, int l_start,
                           const char *output_dir, int n_threads,
                           mpz_t *cur, Range *cur_rng,
                           long total, long rng_n) {
    char tmp_path[4096];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);

    FILE *fp = fopen(tmp_path, "wb");
    if (!fp) {
        fprintf(stderr, "Cannot write checkpoint to %s\n", tmp_path);
        return -1;
    }

    /* Header */
    uint32_t magic = CKPT_MAGIC;
    fwrite(&magic, 4, 1, fp);

    int32_t hdr[7];
    hdr[0] = N; hdr[1] = n; hdr[2] = current_ni; hdr[3] = step;
    hdr[4] = u_flag; hdr[5] = l_start; hdr[6] = n_threads;
    fwrite(hdr, sizeof(int32_t), 7, fp);

    uint32_t dir_len = (uint32_t)strlen(output_dir) + 1;
    fwrite(&dir_len, 4, 1, fp);
    fwrite(output_dir, 1, dir_len, fp);

    /* Non-zero mpz entries */
    int64_t num_nz = 0;
    for (long i = 0; i < total; i++)
        if (mpz_sgn(cur[i]) != 0) num_nz++;
    fwrite(&num_nz, sizeof(int64_t), 1, fp);

    for (long i = 0; i < total; i++) {
        if (mpz_sgn(cur[i]) != 0) {
            int64_t idx = i;
            fwrite(&idx, sizeof(int64_t), 1, fp);
            mpz_out_raw(fp, cur[i]);
        }
    }

    /* Range data */
    int64_t rng_n64 = rng_n;
    fwrite(&rng_n64, sizeof(int64_t), 1, fp);
    for (long i = 0; i < rng_n; i++) {
        int32_t lo = cur_rng[i].lo, hi = cur_rng[i].hi;
        fwrite(&lo, sizeof(int32_t), 1, fp);
        fwrite(&hi, sizeof(int32_t), 1, fp);
    }

    if (fclose(fp) != 0) {
        fprintf(stderr, "Error closing checkpoint %s\n", tmp_path);
        return -1;
    }

    /* Atomic rename */
    if (rename(tmp_path, path) != 0) {
        fprintf(stderr, "Failed to rename %s -> %s\n", tmp_path, path);
        return -1;
    }

    fprintf(stderr, "Checkpoint saved: %s (Ni=%d, step %d/%d)\n",
            path, current_ni, step + 1, n);
    return 0;
}

static int load_checkpoint_header(const char *path,
                                  int *N, int *n, int *current_ni, int *step,
                                  int *u_flag, int *l_start,
                                  char *output_dir, size_t output_dir_size,
                                  int *n_threads) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open checkpoint: %s\n", path);
        return -1;
    }

    uint32_t magic;
    if (fread(&magic, 4, 1, fp) != 1 || magic != CKPT_MAGIC) {
        fprintf(stderr, "Invalid checkpoint (bad magic): %s\n", path);
        fclose(fp);
        return -1;
    }

    int32_t hdr[7];
    if (fread(hdr, sizeof(int32_t), 7, fp) != 7) {
        fprintf(stderr, "Truncated checkpoint header: %s\n", path);
        fclose(fp);
        return -1;
    }
    *N = hdr[0]; *n = hdr[1]; *current_ni = hdr[2]; *step = hdr[3];
    *u_flag = hdr[4]; *l_start = hdr[5]; *n_threads = hdr[6];

    uint32_t dir_len;
    if (fread(&dir_len, 4, 1, fp) != 1 || dir_len > output_dir_size) {
        fprintf(stderr, "Invalid output dir in checkpoint: %s\n", path);
        fclose(fp);
        return -1;
    }
    if (fread(output_dir, 1, dir_len, fp) != dir_len) {
        fprintf(stderr, "Truncated output dir in checkpoint: %s\n", path);
        fclose(fp);
        return -1;
    }

    fclose(fp);
    return 0;
}

static int load_checkpoint_data(const char *path, mpz_t *cur, Range *cur_rng,
                                long total, long rng_n) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open checkpoint for data: %s\n", path);
        return -1;
    }

    /* Skip header: magic(4) + hdr(28) + dir_len(4) + dir_bytes */
    fseek(fp, 4 + 7 * 4, SEEK_SET);
    uint32_t dir_len;
    if (fread(&dir_len, 4, 1, fp) != 1) {
        fprintf(stderr, "Cannot read dir_len from checkpoint\n");
        fclose(fp);
        return -1;
    }
    fseek(fp, dir_len, SEEK_CUR);

    /* Read non-zero mpz entries */
    int64_t num_nz;
    if (fread(&num_nz, sizeof(int64_t), 1, fp) != 1) {
        fprintf(stderr, "Cannot read entry count from checkpoint\n");
        fclose(fp);
        return -1;
    }

    for (int64_t i = 0; i < num_nz; i++) {
        int64_t idx;
        if (fread(&idx, sizeof(int64_t), 1, fp) != 1 ||
            idx < 0 || idx >= total) {
            fprintf(stderr, "Invalid entry index in checkpoint (i=%ld)\n",
                    (long)i);
            fclose(fp);
            return -1;
        }
        if (mpz_inp_raw(cur[idx], fp) == 0) {
            fprintf(stderr, "Failed to read mpz data from checkpoint\n");
            fclose(fp);
            return -1;
        }
    }

    /* Read range data */
    int64_t rng_n64;
    if (fread(&rng_n64, sizeof(int64_t), 1, fp) != 1 || rng_n64 != rng_n) {
        fprintf(stderr, "Range size mismatch in checkpoint "
                "(expected %ld, got %ld)\n", (long)rng_n, (long)rng_n64);
        fclose(fp);
        return -1;
    }
    for (long i = 0; i < rng_n; i++) {
        int32_t lo, hi;
        if (fread(&lo, sizeof(int32_t), 1, fp) != 1 ||
            fread(&hi, sizeof(int32_t), 1, fp) != 1) {
            fprintf(stderr, "Truncated range data in checkpoint\n");
            fclose(fp);
            return -1;
        }
        cur_rng[i].lo = lo;
        cur_rng[i].hi = hi;
    }

    fclose(fp);
    return 0;
}

/* === Main computation ================================================= */

/*
 * Returns 0 on normal completion, 1 if interrupted (checkpoint saved).
 */
static int run_for_N(int N, int n, const char *output_dir,
                     const char *save_path, int save_interval,
                     const char *resume_path, int resume_step,
                     int N_max, int u_flag, int l_start,
                     int n_threads) {
    N_val    = N;
    stride_l = 2L * N + 2;
    stride_s = (long)(N + 1) * stride_l;
    long total = (long)(N + 1) * stride_s;

    /* Precompute binom[r*(N+1)+k] = C(r, k) */
    int bs = N + 1;
    mpz_t *binom = malloc((long)bs * bs * sizeof(mpz_t));
    if (!binom) {
        fprintf(stderr, "OOM: binom (N=%d)\n", N);
        exit(1);
    }
    for (long i = 0; i < (long)bs * bs; i++) mpz_init(binom[i]);
    for (int r = 0; r <= N; r++)
        for (int k = 0; k <= r; k++)
            mpz_bin_uiui(binom[r * bs + k], r, k);

    /* Cache whether each binom fits in unsigned long */
    unsigned long *binom_ul   = malloc((long)bs * bs * sizeof(unsigned long));
    int           *binom_fits = malloc((long)bs * bs * sizeof(int));
    if (!binom_ul || !binom_fits) {
        fprintf(stderr, "OOM: binom cache (N=%d)\n", N);
        exit(1);
    }
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
    long rng_n = (long)(N + 1) * (N + 1);
    Range *cur_rng = malloc((size_t)rng_n * sizeof(Range));
    Range *prv_rng = malloc((size_t)rng_n * sizeof(Range));
    if (!cur_rng || !prv_rng) {
        fprintf(stderr, "OOM: range arrays (N=%d)\n", N);
        exit(1);
    }
    for (long i = 0; i < rng_n; i++) {
        cur_rng[i].lo = INT_MAX; cur_rng[i].hi = -1;
        prv_rng[i].lo = INT_MAX; prv_rng[i].hi = -1;
    }

    /* Init or resume */
    int start_step = 0;
    if (resume_path) {
        if (load_checkpoint_data(resume_path, cur, cur_rng,
                                 total, rng_n) != 0) {
            fprintf(stderr, "Failed to load checkpoint data\n");
            exit(1);
        }
        start_step = resume_step;
        fprintf(stderr, "Resumed: Ni=%d, step %d/%d\n",
                N, start_step + 1, n);
    } else {
        /* Init: cur[i][i][0] = C(N, i) */
        for (int i = 0; i <= N; i++) {
            mpz_bin_uiui(cur[flat(i, i, 0)], N, i);
            cur_rng[i * (N+1) + i].lo = cur_rng[i * (N+1) + i].hi = 0;
        }
    }

    /*
     * Build flat (ns, nl) work list in LPT order: descending ps = ns-nl.
     */
    long n_work = (long)(N + 1) * (N + 2) / 2;
    int *work_ns = malloc((size_t)n_work * sizeof(int));
    int *work_nl = malloc((size_t)n_work * sizeof(int));
    if (!work_ns || !work_nl) {
        fprintf(stderr, "OOM: work arrays (N=%d)\n", N);
        exit(1);
    }
    {
        int w = 0;
        for (int ps = N; ps >= 0; ps--)
            for (int nl = 0; nl <= N - ps; nl++) {
                work_ns[w] = ps + nl;
                work_nl[w] = nl;
                w++;
            }
    }

    int interrupted = 0;
    char path[4096];
    for (int step = start_step; step < n; step++) {
        snprintf(path, sizeof(path), "%s/N_%d_n_%d.txt",
                 output_dir, N, step + 1);
        if (!file_exists(path))
            save_distribution(path, cur);

        if (step == n - 1) break;

        /* Checkpoint: periodic or on signal */
        if (save_path) {
            int do_save = stop_requested;
            if (!do_save && save_interval > 0 &&
                step >= start_step &&
                (step - start_step) % save_interval == 0)
                do_save = 1;
            if (do_save) {
                save_checkpoint(save_path, N_max, n, N, step,
                                u_flag, l_start, output_dir, n_threads,
                                cur, cur_rng, total, rng_n);
                if (stop_requested) {
                    fprintf(stderr, "Interrupted — checkpoint saved.\n");
                    interrupted = 1;
                    goto cleanup;
                }
            }
        } else if (stop_requested) {
            fprintf(stderr, "Interrupted — no -s given, state not saved.\n");
            interrupted = 1;
            goto cleanup;
        }

        fprintf(stderr, "N=%d step %d/%d\n", N, step + 1, n - 1);

        /* Swap cur/prv and their range arrays */
        { mpz_t *t = cur; cur = prv; prv = t; }
        { Range *t = cur_rng; cur_rng = prv_rng; prv_rng = t; }

        /*
         * Merged clear + transition: clear old range then accumulate new
         * values in a single pass, halving page faults when swapping.
         *
         * Each task writes exclusively to cur[ns][nl][*] -> no races.
         */
        #pragma omp parallel
        {
            mpz_t acc;
            mpz_init(acc);

            #pragma omp for schedule(dynamic, 1)
            for (long w = 0; w < n_work; w++) {
                if (stop_requested) continue;  /* early out on signal */

                int ns = work_ns[w];
                int nl = work_nl[w];

                /* Clear old range for this (ns, nl) */
                Range *old_r = &cur_rng[ns * (N+1) + nl];
                if (old_r->lo <= old_r->hi) {
                    mpz_t *base = &cur[flat(ns, nl, old_r->lo)];
                    int len = old_r->hi - old_r->lo + 1;
                    for (int i = 0; i < len; i++)
                        mpz_set_ui(base[i], 0);
                    old_r->lo = INT_MAX; old_r->hi = -1;
                }

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

                    /* Hoist range update */
                    int nd_lo = r.lo + diff;
                    int nd_hi = r.hi + diff;
                    if (nd_lo < rlo) rlo = nd_lo;
                    if (nd_hi > rhi) rhi = nd_hi;

                    /* Base pointers */
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

        /*
         * If interrupted mid-transition, cur is incomplete — save the
         * last known-good state which is in prv/prv_rng (pre-swap data).
         */
        if (stop_requested) {
            if (save_path) {
                save_checkpoint(save_path, N_max, n, N, step,
                                u_flag, l_start, output_dir, n_threads,
                                prv, prv_rng, total, rng_n);
                fprintf(stderr, "Interrupted mid-step — checkpoint saved.\n");
            } else {
                fprintf(stderr, "Interrupted — no -s given, state not saved.\n");
            }
            interrupted = 1;
            goto cleanup;
        }
    }

cleanup:
    for (long i = 0; i < total; i++) { mpz_clear(cur[i]); mpz_clear(prv[i]); }
    free(cur); free(prv);
    for (long i = 0; i < (long)bs * bs; i++) mpz_clear(binom[i]);
    free(binom); free(binom_ul); free(binom_fits);
    free(cur_rng); free(prv_rng);
    free(work_ns); free(work_nl);
    return interrupted;
}

/* === Entry point ====================================================== */

int main(int argc, char **argv) {
    mp_set_memory_functions(gmp_alloc_wrapper, gmp_realloc_wrapper,
                            gmp_free_wrapper);

    int N = -1, n = -1, u_flag = 0, n_threads = 0, l_start = 1;
    char output_dir_buf[4096];
    const char *output_dir = NULL;
    const char *save_path = NULL;
    const char *resume_path = NULL;
    int save_interval = 0;

    int opt;
    while ((opt = getopt(argc, argv, "N:n:o:ut:l:s:i:c:")) != -1) {
        switch (opt) {
        case 'N': N = atoi(optarg); break;
        case 'n': n = atoi(optarg); break;
        case 'o': output_dir = optarg; break;
        case 'u': u_flag = 1; break;
        case 't': n_threads = atoi(optarg); break;
        case 'l': l_start = atoi(optarg); break;
        case 's': save_path = optarg; break;
        case 'i': save_interval = atoi(optarg); break;
        case 'c': resume_path = optarg; break;
        default:
            fprintf(stderr,
                "Usage: %s -N <N> -n <n> [-o <dir>] [-u] [-l <start>] "
                "[-t <threads>] [-s <ckpt>] [-i <interval>]\n"
                "       %s -c <ckpt> [-s <ckpt>] [-i <interval>] "
                "[-t <threads>]\n", argv[0], argv[0]);
            return 1;
        }
    }

    /* Auto-resume: if -s file exists and -c not given, resume from it */
    if (save_path && !resume_path && file_exists(save_path)) {
        resume_path = save_path;
        fprintf(stderr, "Found existing checkpoint %s, auto-resuming\n",
                save_path);
    }

    /* Resume: load parameters from checkpoint file */
    int resume_ni = -1, resume_step = -1;
    int ckpt_n_threads = 0;
    if (resume_path) {
        if (load_checkpoint_header(resume_path,
                &N, &n, &resume_ni, &resume_step,
                &u_flag, &l_start, output_dir_buf, sizeof(output_dir_buf),
                &ckpt_n_threads) != 0)
            return 1;
        if (!output_dir)
            output_dir = output_dir_buf;
        if (n_threads == 0 && ckpt_n_threads > 0)
            n_threads = ckpt_n_threads;
        fprintf(stderr,
            "Checkpoint: N=%d n=%d Ni=%d step=%d/%d u=%d l=%d dir=%s\n",
            N, n, resume_ni, resume_step + 1, n, u_flag, l_start,
            output_dir);
    }

    if (!output_dir)
        output_dir = DEFAULT_OUTPUT_DIR;

    if (N < 0 || n < 0) {
        fprintf(stderr,
            "Usage: %s -N <N> -n <n> [-o <dir>] [-u] [-l <start>] "
            "[-t <threads>] [-s <ckpt>] [-i <interval>]\n"
            "       %s -c <ckpt> [-s <ckpt>] [-i <interval>] "
            "[-t <threads>]\n", argv[0], argv[0]);
        return 1;
    }

    if (n_threads > 0)
        omp_set_num_threads(n_threads);

    /* Install signal handlers for graceful checkpoint-on-stop */
    signal(SIGINT,  handle_stop);
    signal(SIGTERM, handle_stop);

    mkdir(output_dir, 0755);

    int N_start = u_flag ? l_start : N;
    if (resume_path && resume_ni >= 0)
        N_start = resume_ni;

    for (int Ni = N_start; Ni <= N; Ni++) {
        int fms = first_missing_step(Ni, n, output_dir);
        if (fms >= n) {
            fprintf(stderr, "N=%d: all %d distributions exist, skipping\n",
                    Ni, n);
            if (resume_path && Ni == resume_ni)
                resume_path = NULL;
            continue;
        }

        int ret;
        if (resume_path && Ni == resume_ni) {
            ret = run_for_N(Ni, n, output_dir, save_path, save_interval,
                            resume_path, resume_step,
                            N, u_flag, l_start, n_threads);
            resume_path = NULL;  /* only resume once */
        } else {
            ret = run_for_N(Ni, n, output_dir, save_path, save_interval,
                            NULL, -1,
                            N, u_flag, l_start, n_threads);
        }
        if (ret != 0 || stop_requested)
            return 1;
    }

    return 0;
}
