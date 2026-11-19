#!/bin/sh
# Keep code/alive.py running (started at boot by cron: @reboot line below): after a crash (error exit) it starts
# again in 30 s; a clean stop (pkill -f code/alive.py, Ctrl+C) ends it until the next boot.
#   @reboot sleep 30 && flock -n /tmp/arcby-alive.lock /home/arjein/projects/arcby/code/run_alive.sh
# alive.py logs into ~/scratch/alive/alive.log (rotated); boot.log only gets what goes wrong outside it.
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$HOME/scratch/alive"
# Exit code 3 = could not start (camera busy - e.g. the dashboard's live view -, a model missing): each failed start
# waits twice as long as the one before (60 s ... 10 min), so a lasting problem does not restart (and pop the
# speaker) every 30 s. A run that got going resets the wait.
wait_s=60
while true; do
    echo "$(date '+%F %T') run_alive: starting alive.py" >> "$HOME/scratch/alive/boot.log"
    python3 code/alive.py "$@" >> "$HOME/scratch/alive/boot.log" 2>&1
    code=$?
    echo "$(date '+%F %T') run_alive: alive.py ended with code $code" >> "$HOME/scratch/alive/boot.log"
    [ "$code" -eq 0 ] && exit 0               # stopped on purpose
    if [ "$code" -eq 3 ]; then                # could not start: wait longer each time
        echo "$(date '+%F %T') run_alive: next try in $wait_s s" >> "$HOME/scratch/alive/boot.log"
        sleep "$wait_s"
        wait_s=$((wait_s * 2))
        [ "$wait_s" -gt 600 ] && wait_s=600
    else
        wait_s=60
        sleep 30                              # crashed while running: try again
    fi
done
