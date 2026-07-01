"""
bot/ — Kalshi 15-minute prediction market trading brain.

Module layout
-------------
config.py    Frozen strategy spec (K, T per series), Kelly params, paths.
sizing.py    kalshi_fee() and kelly_fraction() — exact h12 formulas.
brain.py     StrategyBrain: WindowState, on_trade(), decide(), on_settlement().
recal.py     Daily OLS refit of (b0, b1) per series from historical CSV.
run_live.py  Integration entrypoint: WebSocket → brain → decisions.jsonl.

Quick start
-----------
    cd /Users/sourishsuri/kalshi
    python3 -m bot.run_live          # start live decision logging

    # Or use the brain programmatically:
    from bot.recal import fit_all_series
    from bot.brain import StrategyBrain

    coefs = fit_all_series()
    brain = StrategyBrain(coefs=coefs, initial_bankroll=500.0)
    brain.register_window("KXBTC15M-26JUL031445-50000", "KXBTC15M", open_ts=...)
    brain.on_trade("KXBTC15M-26JUL031445-50000", price_dollars=0.52, trade_ts=...)
    decision = brain.decide("KXBTC15M-26JUL031445-50000", yes_ask=0.54, no_ask=0.48, bankroll=500)
    print(decision)
"""
