#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"

for script in "$DIR"/fit_*.sh; do
    name=$(basename "$script" .sh)       # fit_beta
    session="${name#fit_}"               # beta

    if tmux has-session -t "=$session" 2>/dev/null; then
        echo "Session '$session' already exists, skipping."
    else
        tmux new-session -d -s "$session" -c "$DIR" "bash $script"
        echo "Started session '$session'."
    fi
done
