#!/bin/bash
cd "$(dirname "$0")"
while true; do
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    if tmux has-session -t ct 2>/dev/null; then
        echo "$ts  ct session running, nothing to do"
    else
        echo "$ts  ct session not found, starting"
        tmux new-session -d -s ct './ct_save -N 500 -n 500 -u -l 478 -o ../data/exact_distributions -s ckpt.bin -i 20'
        #tmux new-session -d -s ct './ct_c -N 500 -n 500 -u -l 468 -o ../data/exact_distributions'
    fi
    sleep 5
done
