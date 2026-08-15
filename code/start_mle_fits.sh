#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"

for script in "$DIR"/mle_fit_*.sh; do
    name=$(basename "$script" .sh)       # mle_fit_beta
    session="${name#mle_fit_}"           # beta
    session="mle_${session}"            # mle_beta

    if tmux has-session -t "=$session" 2>/dev/null; then
        echo "Session '$session' already exists, skipping."
    else
        tmux new-session -d -s "$session" -c "$DIR" "bash $script"
        echo "Started session '$session'."
    fi
done
