#!/bin/bash
DIR="$(cd "$(dirname "$0")" && pwd)"

for script in "$DIR"/fit_*.sh; do
    name=$(basename "$script" .sh)
    session="${name#fit_}"

    if tmux has-session -t "=$session" 2>/dev/null; then
        tmux kill-session -t "=$session"
        echo "Killed session '$session'."
    else
        echo "Session '$session' not running."
    fi
done
