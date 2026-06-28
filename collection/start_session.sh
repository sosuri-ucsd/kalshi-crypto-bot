#!/bin/bash
cd ~/kalshi || { echo "Can't find ~/kalshi"; exit 1; }

if pgrep -f "ws_kalshi.py" > /dev/null; then
    echo "ws_kalshi.py is already running. Not starting a duplicate."
else
    echo "Starting Kalshi capture..."
    nohup caffeinate -i python3 ws_kalshi.py > capture_heartbeat.log 2>&1 &
    KALSHI_PID=$!
    disown
    echo "  -> PID $KALSHI_PID"
fi

if pgrep -f "ws_binance.py" > /dev/null; then
    echo "ws_binance.py is already running. Not starting a duplicate."
else
    echo "Starting Binance capture..."
    nohup caffeinate -i python3 ws_binance.py > binance_heartbeat.log 2>&1 &
    BINANCE_PID=$!
    disown
    echo "  -> PID $BINANCE_PID"
fi

sleep 2
echo ""
echo "Current status:"
ps aux | grep -E "ws_kalshi|ws_binance" | grep -v grep
