#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"

for script in "$DIR"/mle_fit_*.sh; do
    name=$(basename "$script" .sh)
    session="${name#mle_fit_}"
    session="mle_${session}"

    if tmux has-session -t "=$session" 2>/dev/null; then
        tmux kill-session -t "=$session"
        echo "Killed session '$session'."
    else
        echo "Session '$session' not running."
    fi
done
