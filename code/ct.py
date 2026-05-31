import argparse
from os import makedirs
from os.path import isfile, join
from math import comb
from tqdm import tqdm

DEFAULT_OUTPUT_DIR = "../data/exact_distributions/"

def run_for_N(N, n, output_dir):
    previous_counts = []
    current_counts = []
    for j in range(N + 1):
        previous_counts2 = []
        current_counts2 = []
        for k in range(N + 1):
            previous_counts2.append([0] * (2 * N + 1 + 1))
            current_counts2.append([0] * (2 * N + 1 + 1))
        previous_counts.append(previous_counts2)
        current_counts.append(current_counts2)

    for i in range(N + 1):
        current_counts[i][i][0] = comb(N, i)

    def compute_distribution(counts):
        dtv_counts = dict()
        for last in range(0, N + 1):
            for dtv in range(0, 2 * N + 1):
                count = counts[N][last][dtv]
                dtv_counts[dtv] = dtv_counts.get(dtv, 0) + count
        distribution = []
        for dtv in range(max(dtv_counts.keys()) + 1):
            distribution.append((dtv, dtv_counts.get(dtv, 0)))
        return distribution

    def save_distribution(distribution, path):
        with open(path, "w", encoding="utf8") as fo:
            for a, b in distribution:
                fo.write(" ".join(map(str, [a, b])) + "\n")

    # idx=0 corresponds to n=1
    idx = 0
    output_path = join(output_dir, "N_%d_n_%d.txt" % (N, idx + 1))
    if not isfile(output_path):
        distribution = compute_distribution(current_counts)
        save_distribution(distribution, output_path)

    if n == 1:
        return

    for idx in tqdm(range(1, n), desc="N=%d" % N):
        previous_counts, current_counts = current_counts, previous_counts
        for i in range(len(current_counts)):
            for j in range(len(current_counts[i])):
                for k in range(len(current_counts[i][j])):
                    current_counts[i][j][k] = 0
        for previous_sum in range(0, N + 1):
            remaining = N - previous_sum
            for previous_last in range(0, previous_sum + 1):
                for previous_dtv in range(0, 2 * previous_sum + 1 + 1):
                    previous_count = previous_counts[previous_sum][previous_last][previous_dtv]
                    if previous_count > 0:
                        for new_last in range(0, remaining + 1):
                            new_sum = previous_sum + new_last
                            new_dtv = previous_dtv + abs(new_last - previous_last)
                            current_counts[new_sum][new_last][new_dtv] += previous_count * comb(remaining, new_last)

        output_path = join(output_dir, "N_%d_n_%d.txt" % (N, idx + 1))
        if not isfile(output_path):
            distribution = compute_distribution(current_counts)
            save_distribution(distribution, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, required=True)
    parser.add_argument("-n", type=int, required=True)
    parser.add_argument("-o", dest="output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-u", action="store_true", help="Run for all N from 1 to the given N")
    args = parser.parse_args()

    output_dir = args.output_dir
    makedirs(output_dir, exist_ok=True)

    N_values = range(1, args.N + 1) if args.u else [args.N]
    for N in N_values:
        run_for_N(N, args.n, output_dir)

if __name__ == "__main__":
    main()
