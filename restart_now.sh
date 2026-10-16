#!/bin/bash
CWD="/home/u114856156/trading_platform"
PYTHON="$CWD/.venv/bin/python3"
PID_FILE="/home/u114856156/.pid"

# Kill current daemon
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    kill -TERM "$OLD_PID" 2>/dev/null
    sleep 2
    kill -9 "$OLD_PID" 2>/dev/null
fi

# Kill any lingering uvicorn
pkill -f "uvicorn" 2>/dev/null
sleep 1

# Remove old socket
rm -f "$CWD/uvicorn.sock"

# Start fresh daemon
"$PYTHON" "$CWD/start_daemon.py" >> "$CWD/cron.log" 2>&1 &
echo "Daemon reiniciado con PID: $!"
