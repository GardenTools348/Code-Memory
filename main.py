import os
import logging
from contextlib import asynccontextmanager

import anthropic
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pybit.unified_trading import HTTP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("swing-trader")

# Config
BOT_SECRET        = os.environ.get("RAILWAY_BOT_SECRET", "")
BYBIT_API_KEY     = os.environ.get("BYBIT_API_KEY", "")
BYBIT_API_SECRET  = os.environ.get("BYBIT_API_SECRET", "")
BYBIT_DEMO        = os.environ.get("BYBIT_DEMO", "true").lower() == "true"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ORDER_QTY_USDT    = float(os.environ.get("ORDER_QTY_USDT", "100"))
STOP_LOSS_PCT     = float(os.environ.get("STOP_LOSS_PCT", "2"))
TAKE_PROFIT_PCT   = float(os.environ.get("TAKE_PROFIT_PCT", "4"))
SYMBOL            = os.environ.get("SYMBOL", "BTCUSDT")

session = HTTP(demo=BYBIT_DEMO, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
claude  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# --- Indicator helpers ---

def calc_ema(values: list, period: int) -> list:
    k = 2 / (period + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(v * k + result[-1] * (1 - k))
    return result

def calc_sma(values: list, period: int) -> list:
    return [sum(values[i:i+period]) / period for i in range(len(values) - period + 1)]

def calc_rsi(closes: list, period: int = 14) -> list:
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsi_vals = []
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss > 0 else 100
        rsi_vals.append(100 - 100 / (1 + rs))
    return rsi_vals

def find_pivot_lows(lows: list, lb: int = 10) -> list:
    return [lows[i] for i in range(lb, len(lows) - lb)
            if lows[i] == min(lows[i-lb:i+lb+1])]

def find_pivot_highs(highs: list, lb: int = 10) -> list:
    return [highs[i] for i in range(lb, len(highs) - lb)
            if highs[i] == max(highs[i-lb:i+lb+1])]


# --- Strategy signal check ---

def check_signal() -> dict:
    resp    = session.get_kline(category="linear", symbol=SYMBOL, interval="D", limit=200)
    candles = list(reversed(resp["result"]["list"]))  # oldest first

    closes = [float(c[4]) for c in candles]
    highs  = [float(c[2]) for c in candles]
    lows   = [float(c[3]) for c in candles]
    vols   = [float(c[5]) for c in candles]

    ema21_series = calc_ema(closes, 21)
    ma50_series  = calc_sma(closes, 50)
    rsi_series   = calc_rsi(closes, 14)
    vol_ma20     = calc_sma(vols, 20)

    price    = closes[-1]
    ema21    = ema21_series[-1]
    ma50     = ma50_series[-1]
    rsi_now  = rsi_series[-1]
    rsi_prev = rsi_series[-2]
    vol_now  = vols[-1]
    vol_avg  = vol_ma20[-1]

    pivot_lows  = find_pivot_lows(lows)
    pivot_highs = find_pivot_highs(highs)
    support     = pivot_lows[-1] if pivot_lows else None

    hh_hl = (len(pivot_highs) >= 2 and pivot_highs[-1] > pivot_highs[-2] and
              len(pivot_lows)  >= 2 and pivot_lows[-1]  > pivot_lows[-2])

    c1_trend   = price > ema21 and ema21 > ma50 and hh_hl
    c2_support = support is not None and abs(price - support) / support <= 0.015
    c3_ema     = abs(price - ema21) / ema21 <= 0.02
    c4_volume  = vol_now > vol_avg
    rsi_low5   = min(rsi_series[-5:])
    c5_rsi     = 40 <= rsi_low5 <= 50 and rsi_now > rsi_prev

    return {
        "all_conditions_met": c1_trend and c2_support and c3_ema and c4_volume and c5_rsi,
        "price":     price,
        "ema21":     round(ema21, 2),
        "ma50":      round(ma50, 2),
        "rsi":       round(rsi_now, 2),
        "vol_ratio": round(vol_now / vol_avg, 2),
        "support":   round(support, 2) if support else None,
        "conditions": {
            "trend":   c1_trend,
            "support": c2_support,
            "ema":     c3_ema,
            "volume":  c4_volume,
            "rsi":     c5_rsi,
        }
    }


def ask_claude(signal: dict) -> str:
    prompt = f"""You are a trading assistant reviewing a BTC long setup.

All 5 confluence conditions are confirmed on the Daily chart:
- Bullish trend: price ({signal['price']}) > 21 EMA ({signal['ema21']}), 21 EMA > 50 MA ({signal['ma50']}), higher highs/lows confirmed
- Price within 1.5% of swing low support ({signal['support']})
- Price within 2% of 21 EMA
- Volume is {signal['vol_ratio']}x the 20-bar average
- RSI rebounded from 40-50 zone, now {signal['rsi']} and rising

Should we enter a long position on BTCUSDT?
Reply with 'buy' or 'hold' followed by a one-sentence reason."""

    msg = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text.strip().lower()


# --- Scheduled scan ---

async def run_scan():
    logger.info("Swing Trader scan running...")
    try:
        signal = check_signal()
        logger.info("Signal: %s", signal)

        if not signal["all_conditions_met"]:
            logger.info("Conditions not fully met — no trade.")
            return

        logger.info("All 5 conditions met — consulting Claude...")
        response = ask_claude(signal)
        logger.info("Claude: %s", response)

        if not response.startswith("buy"):
            logger.info("Claude said hold — no trade.")
            return

        price       = signal["price"]
        qty         = round(ORDER_QTY_USDT / price, 3)
        stop_loss   = round(price * (1 - STOP_LOSS_PCT / 100), 2)
        take_profit = round(price * (1 + TAKE_PROFIT_PCT / 100), 2)

        close_opposing_position(SYMBOL, "Buy")

        result = session.place_order(
            category="linear",
            symbol=SYMBOL,
            side="Buy",
            orderType="Market",
            qty=str(qty),
            stopLoss=str(stop_loss),
            takeProfit=str(take_profit),
        )
        logger.info("Order placed: %s", result)

    except Exception as e:
        logger.error("Scan error: %s", e)


# --- App startup/shutdown ---

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(run_scan, "interval", hours=2)
    scheduler.start()
    logger.info("Swing Trader started — scanning every 2 hours.")
    yield
    scheduler.shutdown()

app = FastAPI(title="Swing Trader", lifespan=lifespan)


# --- Endpoints ---

class ExecuteRequest(BaseModel):
    symbol: str
    action: str
    price: float | None = None
    reason: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "demo": BYBIT_DEMO}


@app.post("/execute")
async def execute(req: ExecuteRequest, request: Request):
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {BOT_SECRET}":
        raise HTTPException(status_code=401, detail="invalid secret")

    if req.action not in ("buy", "sell"):
        return {"status": "skipped", "reason": f"action is {req.action}"}

    if not req.price:
        raise HTTPException(status_code=400, detail="price is required")

    side        = "Buy" if req.action == "buy" else "Sell"
    qty         = round(ORDER_QTY_USDT / req.price, 3)
    stop_loss   = round(req.price * (1 - STOP_LOSS_PCT / 100) if side == "Buy" else req.price * (1 + STOP_LOSS_PCT / 100), 2)
    take_profit = round(req.price * (1 + TAKE_PROFIT_PCT / 100) if side == "Buy" else req.price * (1 - TAKE_PROFIT_PCT / 100), 2)

    try:
        close_result = close_opposing_position(req.symbol, side)
        result = session.place_order(
            category="linear",
            symbol=req.symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            stopLoss=str(stop_loss),
            takeProfit=str(take_profit),
        )
        logger.info("Order result: %s", result)
        return {"status": "executed", "order": result, "closed_position": close_result}
    except Exception as exc:
        logger.error("Order failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


def close_opposing_position(symbol: str, new_side: str) -> dict | None:
    positions = session.get_positions(category="linear", symbol=symbol)
    for pos in positions["result"]["list"]:
        size = float(pos.get("size", 0))
        if size <= 0:
            continue
        existing_side = pos["side"]
        if existing_side == new_side:
            continue
        close_side = "Sell" if existing_side == "Buy" else "Buy"
        result = session.place_order(
            category="linear",
            symbol=symbol,
            side=close_side,
            orderType="Market",
            qty=str(size),
            reduceOnly=True,
        )
        logger.info("Closed opposing %s position: %s", existing_side, result)
        return result
    return None
