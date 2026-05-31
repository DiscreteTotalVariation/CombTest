# -*- coding: utf-8 -*-

from os import listdir, makedirs
from os.path import isdir, isfile, join
from tqdm import tqdm

ROOT_DIR=""

EXACT_DISTRIBUTIONS_DIR=join(ROOT_DIR, "../data/exact_distributions")
CDF_EXACT_DIR=join(ROOT_DIR, "../data/cdf_exact")
HISTOGRAM_CONFIGURATION_PATTERN="N_%d_n_%d.txt"

def calculate_exact_cdf(N, n):
    exact_distributions_dir=EXACT_DISTRIBUTIONS_DIR
    cdf_dir=CDF_EXACT_DIR

    makedirs(cdf_dir, exist_ok=True)

    cdf_path=join(cdf_dir, HISTOGRAM_CONFIGURATION_PATTERN%(N, n))
    if isfile(cdf_path) is True:
        return

    # load the distribution
    distribution=dict()
    input_path=join(exact_distributions_dir, HISTOGRAM_CONFIGURATION_PATTERN%(N, n))
    with open(input_path, "r", encoding="utf8") as f:
        lines=[l.strip("\r\n") for l in f.readlines()]
    for l in lines:
        a, b=list(map(int, l.split()))
        distribution[a]=b

    exact_cdf_values=[]
    dtv_values=[]

    # calculate the exact CDF
    numerator=0
    denominator=sum(distribution.values())
    for value, occurrence in sorted(distribution.items()):
        if occurrence==0:
            continue
        numerator+=occurrence
        exact_cdf_value=numerator/denominator

        dtv_values.append(value)
        exact_cdf_values.append(exact_cdf_value)

    # save the CDF values
    with open(cdf_path, "w", encoding="utf8") as fo:
        for dtv_value, exact_cdf_value in zip(dtv_values, exact_cdf_values):
            fo.write(str(dtv_value)+" "+str(exact_cdf_value)+"\n")

def main():

    input_dir=EXACT_DISTRIBUTIONS_DIR

    if isdir(input_dir) is False:
        return

    names=sorted(listdir(input_dir))

    show_status=True

    taker=names
    if show_status is True:
        taker=tqdm(names)
    for name in taker:
        if not name.startswith("N_") or not name.endswith(".txt"):
            continue
        parts=name[:name.rfind(".")].split("_")
        N=int(parts[1])
        n=int(parts[3])

        cdf_output_path=join(CDF_EXACT_DIR, HISTOGRAM_CONFIGURATION_PATTERN%(N, n))

        if isfile(cdf_output_path) is True:
            continue

        calculate_exact_cdf(N=N, n=n)

if __name__=="__main__":
    main()
