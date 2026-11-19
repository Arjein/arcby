#!/usr/bin/env bash
# Copy the Arcby dashboard to the Pi and (re)start it.
# Usage:  pi/dashboard/deploy.sh            (from the Mac, over SSH)
#         PI=arjein@other-ip pi/dashboard/deploy.sh
# When run on the Pi itself it installs locally (no SSH needed).
set -euo pipefail

PI="${PI:-arjein@raspberrypi.local}"
DIR=/srv/arcby/dashboard
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/arcby_dashboard.py $HERE/arcby_archive.py"     # the page + its archive tabs (People, Heard)

# Stops the running instance (via its pid file) and starts a new one in the background.
# flock keeps it to one instance; the @reboot crontab entry uses the same lock.
RESTART=$(cat <<EOF
set -e
cd $DIR
if [ -f dashboard.pid ]; then
  pid=\$(cat dashboard.pid)
  if grep -aq arcby_dashboard /proc/\$pid/cmdline 2>/dev/null; then
    kill \$pid
    for i in 1 2 3 4 5 6 7 8 9 10; do [ -d /proc/\$pid ] || break; sleep 0.3; done
  fi
fi
setsid nohup flock -n $DIR/.lock python3 $DIR/arcby_dashboard.py >> $DIR/dashboard.log 2>&1 < /dev/null &
sleep 1.5
curl -fsS -o /dev/null http://127.0.0.1:8080/api/status && echo "dashboard running: http://\$(hostname -I | cut -d' ' -f1):8080/" || { echo "dashboard did not start - tail of log:"; tail -n 20 $DIR/dashboard.log; exit 1; }
EOF
)

if hostname -I 2>/dev/null | tr ' ' '\n' | grep -qx "${PI#*@}"; then
  mkdir -p "$DIR"
  cp $SRC "$DIR/"
  bash -c "$RESTART"
else
  ssh "$PI" "mkdir -p $DIR"
  scp $SRC "$PI:$DIR/"
  ssh "$PI" "$RESTART"
fi
