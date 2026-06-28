#!/bin/bash
echo "Stopping ws_kalshi.py and ws_binance.py..."
pkill -f "ws_kalshi.py"
pkill -f "ws_binance.py"
sleep 1
echo "Remaining matches (should be empty):"
ps aux | grep -E "ws_kalshi|ws_binance" | grep -v grep
