#!/bin/bash
cd "$(dirname "$0")"

while getopts "N:n:" opt; do
    case $opt in
        N) N="$OPTARG" ;;
        n) n="$OPTARG" ;;
        *) echo "Usage: $0 -N <N> -n <n>"; exit 1 ;;
    esac
done

if [ -z "$N" ] || [ -z "$n" ]; then
    echo "Usage: $0 -N <N> -n <n>"
    exit 1
fi

file="N_${N}_n_${n}.txt"
results=()

for dir in ../data/fitted/mle_*/; do
    dir="${dir%/}"
    path="${dir}/${file}"
    if [ -f "$path" ]; then
        value=$(sed -n '2p' "$path")
        results+=("${value} ${dir}")
    fi
done

printf '%s\n' "${results[@]}" | sort -g
