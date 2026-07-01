import asyncio
import json
import math
import time
import logging
from collections import deque

import websockets

logger = logging.getLogger(__name__)

SYMBOL_MAP = {
    "btcusdt": "KXBTC15M",
    "ethusdt": "KXETH15M",
    "solusdt": "KXSOL15M",
}
SERIES_MAP = {v: k for k, v in SYMBOL_MAP.items()}
WS_URL = (
    "wss://stream.binance.com:9443/stream?streams="
    "btcusdt@aggTrade/ethusdt@aggTrade/solusdt@aggTrade"
)
BUFFER_SIZE = 600


class BinanceFeed:
    def __init__(self):
        self._buffers = {series: deque(maxlen=BUFFER_SIZE) for series in SYMBOL_MAP.values()}
        self._running = False
        self._last_heartbeat = time.time()
        self._buffer_fill_start = {series: None for series in self._buffers}
        self._buffer_first_size = {series: 0 for series in self._buffers}

    async def start(self):
        self._running = True
        await self._connect_loop()

    async def _connect_loop(self):
        while True:
            try:
                await self._run_connection()
            except Exception as e:
                logger.warning("BINANCE RECONNECTING (reason: %s)", e)
                await asyncio.sleep(3)

    async def _run_connection(self):
        async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
            logger.info("BINANCE CONNECTED")
            async for raw in ws:
                msg = json.loads(raw)
                stream = msg.get("stream", "")
                data = msg.get("data", {})
                symbol = stream.split("@")[0]
                series = SYMBOL_MAP.get(symbol)
                if series and "p" in data:
                    if self._buffer_fill_start[series] is None:
                        self._buffer_fill_start[series] = time.time()
                    self._buffers[series].append(float(data["p"]))
                now = time.time()
                if now - self._last_heartbeat >= 60:
                    self._last_heartbeat = now
                    btc = self._latest("KXBTC15M")
                    eth = self._latest("KXETH15M")
                    sol = self._latest("KXSOL15M")
                    logger.info(
                        "BINANCE PRICE BTCUSDT=%.2f ETHUSDT=%.2f SOLUSDT=%.2f",
                        btc, eth, sol,
                    )

    def _latest(self, series: str) -> float:
        buf = self._buffers[series]
        return buf[-1] if buf else 0.0

    def get_price(self, series: str) -> float:
        return self._latest(series)

    def get_sigma(self, series: str) -> float:
        buf = list(self._buffers[series])
        if len(buf) < 10:
            return 0.0
        log_returns = [
            math.log(buf[i + 1] / buf[i])
            for i in range(len(buf) - 1)
            if buf[i] > 0 and buf[i + 1] > 0
        ]
        if len(log_returns) < 9:
            return 0.0
        mean = sum(log_returns) / len(log_returns)
        variance = sum((r - mean) ** 2 for r in log_returns) / len(log_returns)
        return math.sqrt(variance)

    def get_sigma_per_second(self, series: str) -> float:
        raw_sigma = self.get_sigma(series)
        if raw_sigma == 0.0:
            return 0.0
        buf = self._buffers[series]
        if len(buf) < 10:
            return 0.0
        elapsed = time.time() - self._buffer_fill_start[series]
        if elapsed <= 0:
            return raw_sigma
        ticks_per_second = len(buf) / elapsed
        return raw_sigma * math.sqrt(ticks_per_second)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    async def test():
        feed = BinanceFeed()
        task = asyncio.create_task(feed.start())
        for i in range(6):
            await asyncio.sleep(5)
            for series in ["KXBTC15M", "KXETH15M", "KXSOL15M"]:
                print(
                    f"{series}: price={feed.get_price(series):.2f}"
                    f"  sigma_per_sec={feed.get_sigma_per_second(series):.6f}"
                    f"  n={len(feed._buffers[series])}"
                )
        task.cancel()

    asyncio.run(test())
