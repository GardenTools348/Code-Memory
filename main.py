import os
import logging
import urllib.request
import urllib.parse
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import anthropic
import pg8000
import pg8000.dbapi
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pybit.unified_trading import HTTP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("swing-trader")

# Config
BOT_SECRET          = os.environ.get("RAILWAY_BOT_SECRET", "")
BYBIT_API_KEY       = os.environ.get("BYBIT_API_KEY", "")
BYBIT_API_SECRET    = os.environ.get("BYBIT_API_SECRET", "")
BYBIT_DEMO          = os.environ.get("BYBIT_DEMO", "true").lower() == "true"
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_BOT_TOKEN  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID", "")
ORDER_QTY_USDT      = float(os.environ.get("ORDER_QTY_USDT", "100"))
STOP_LOSS_PCT       = float(os.environ.get("STOP_LOSS_PCT", "2"))
TAKE_PROFIT_PCT     = float(os.environ.get("TAKE_PROFIT_PCT", "4"))
SYMBOL              = os.environ.get("SYMBOL", "BTCUSDT")
DATABASE_URL        = os.environ.get("DATABASE_URL", "")

session = HTTP(demo=BYBIT_DEMO, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)
claude  = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


# --- Database ---

DB_AVAILABLE = False

def get_db():
    import re
    m = re.match(r'postgres(?:ql)?://([^:]+):(.+)@\[?([^\]/:]+)\]?:(\d+)/(.+)', DATABASE_URL)
    if not m:
        raise ValueError("Could not parse DATABASE_URL")
    user, password, host, port, database = m.groups()
    return pg8000.dbapi.connect(
        host=host, port=int(port),
        database=database.split("?")[0],
        user=user, password=password,
        ssl_context=True
    )

def init_db():
    global DB_AVAILABLE
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set — trade journaling disabled.")
        return
    import re
    m = re.match(r'postgres(?:ql)?://([^:]+):(.+)@\[?([^\]/:]+)\]?:(\d+)/(.+)', DATABASE_URL)
    if m:
        user, _, host, port, database = m.groups()
        logger.info("DB connect → user=%s host=%s port=%s db=%s", user, host, port, database.split("?")[0])
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id            SERIAL PRIMARY KEY,
                    symbol        VARCHAR(20),
                    side          VARCHAR(10),
                    entry_price   FLOAT,
                    qty           FLOAT,
                    stop_loss     FLOAT,
                    take_profit   FLOAT,
                    entry_time    TIMESTAMP,
                    exit_price    FLOAT,
                    exit_time     TIMESTAMP,
                    exit_reason   VARCHAR(20),
                    pnl_usdt      FLOAT,
                    pnl_pct       FLOAT,
                    claude_reason TEXT
                )
            """)
            conn.commit()
        DB_AVAILABLE = True
        logger.info("Database ready.")
    except Exception as e:
        logger.warning("Database unavailable — trade journaling disabled: %s", e)

def log_trade_entry(side: str, entry_price: float, qty: float,
                    stop_loss: float, take_profit: float, claude_reason: str) -> int | None:
    if not DB_AVAILABLE:
        return None
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO trades (symbol, side, entry_price, qty, stop_loss, take_profit, entry_time, claude_reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
            """, (SYMBOL, side, entry_price, qty, stop_loss, take_profit,
                  datetime.now(timezone.utc), claude_reason))
            trade_id = cur.fetchone()[0]
            conn.commit()
        return trade_id
    except Exception as e:
        logger.error("Failed to log trade entry: %s", e)
        return None

def log_trade_exit(symbol: str, side: str, exit_price: float,
                   exit_reason: str, entry_price: float, qty: float):
    if side == "Buy":
        pnl_usdt = (exit_price - entry_price) * qty
        pnl_pct  = (exit_price - entry_price) / entry_price * 100
    else:
        pnl_usdt = (entry_price - exit_price) * qty
        pnl_pct  = (entry_price - exit_price) / entry_price * 100

    if DB_AVAILABLE:
        try:
            with get_db() as conn:
                cur = conn.cursor()
                cur.execute("""
                    UPDATE trades SET exit_price=%s, exit_time=%s, exit_reason=%s,
                        pnl_usdt=%s, pnl_pct=%s
                    WHERE id = (
                        SELECT id FROM trades
                        WHERE symbol=%s AND side=%s AND exit_price IS NULL
                        ORDER BY entry_time DESC LIMIT 1
                    )
                """, (exit_price, datetime.now(timezone.utc), exit_reason,
                      pnl_usdt, pnl_pct, symbol, side))
                conn.commit()
        except Exception as e:
            logger.error("Failed to log trade exit: %s", e)

    return pnl_usdt, pnl_pct


# --- Telegram ---

def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": TELEGRAM_CHAT_ID, "text": message}).encode()
        urllib.request.urlopen(url, data=data, timeout=10)
    except Exception as e:
        logger.error("Telegram notification failed: %s", e)


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
    deltas   = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains    = [max(d, 0) for d in deltas]
    losses   = [max(-d, 0) for d in deltas]
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

def find_tp_levels(entry_price: float, side: str) -> tuple[float, float]:
    resp    = session.get_kline(category="linear", symbol=SYMBOL, interval="240", limit=200)
    candles = list(reversed(resp["result"]["list"]))
    highs   = [float(c[2]) for c in candles]
    lows    = [float(c[3]) for c in candles]
    if side == "Buy":
        pivots = sorted(p for p in find_pivot_highs(highs, lb=5) if p > entry_price)
        tp1 = round(pivots[0], 2) if len(pivots) > 0 else round(entry_price * 1.03, 2)
        tp2 = round(pivots[1], 2) if len(pivots) > 1 else round(entry_price * 1.06, 2)
    else:
        pivots = sorted((p for p in find_pivot_lows(lows, lb=5) if p < entry_price), reverse=True)
        tp1 = round(pivots[0], 2) if len(pivots) > 0 else round(entry_price * 0.97, 2)
        tp2 = round(pivots[1], 2) if len(pivots) > 1 else round(entry_price * 0.94, 2)
    return tp1, tp2

def check_momentum_exhaustion(side: str) -> bool:
    resp    = session.get_kline(category="linear", symbol=SYMBOL, interval="240", limit=50)
    candles = list(reversed(resp["result"]["list"]))
    closes  = [float(c[4]) for c in candles]
    vols    = [float(c[5]) for c in candles]
    rsi     = calc_rsi(closes, 14)
    vol_ma  = calc_sma(vols, 20)
    rsi_now   = rsi[-1]
    vol_ratio = vols[-1] / vol_ma[-1]
    return (rsi_now > 70 and vol_ratio > 1.5) if side == "Buy" else (rsi_now < 30 and vol_ratio > 1.5)


# --- Strategy signal check ---

def check_signals() -> dict:
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
    support     = pivot_lows[-1]  if pivot_lows  else None
    resistance  = pivot_highs[-1] if pivot_highs else None

    # Trend structure
    hh_hl = (len(pivot_highs) >= 2 and pivot_highs[-1] > pivot_highs[-2] and
              len(pivot_lows)  >= 2 and pivot_lows[-1]  > pivot_lows[-2])
    lh_ll = (len(pivot_highs) >= 2 and pivot_highs[-1] < pivot_highs[-2] and
              len(pivot_lows)  >= 2 and pivot_lows[-1]  < pivot_lows[-2])

    # Volume
    c_volume = vol_now > vol_avg

    # --- LONG conditions ---
    long_trend   = price > ema21 and ema21 > ma50 and hh_hl
    long_support = support is not None and abs(price - support) / support <= 0.015
    long_ema     = abs(price - ema21) / ema21 <= 0.02
    rsi_low5     = min(rsi_series[-5:])
    long_rsi     = 40 <= rsi_low5 <= 50 and rsi_now > rsi_prev
    long_signal  = long_trend and long_support and long_ema and c_volume and long_rsi

    # --- SHORT conditions ---
    short_trend      = price < ema21 and ema21 < ma50 and lh_ll
    short_resistance = resistance is not None and abs(price - resistance) / resistance <= 0.02
    short_ema        = abs(price - ema21) / ema21 <= 0.02
    rsi_high5        = max(rsi_series[-5:])
    short_rsi        = 50 <= rsi_high5 <= 60 and rsi_now < rsi_prev
    short_signal     = short_trend and short_resistance and short_ema and c_volume and short_rsi

    return {
        "price":      price,
        "ema21":      round(ema21, 2),
        "ma50":       round(ma50, 2),
        "rsi":        round(rsi_now, 2),
        "vol_ratio":  round(vol_now / vol_avg, 2),
        "support":    round(support, 2)    if support    else None,
        "resistance": round(resistance, 2) if resistance else None,
        "long":  {"signal": long_signal,  "conditions": {"trend": long_trend,  "support": long_support,  "ema": long_ema,  "volume": c_volume, "rsi": long_rsi}},
        "short": {"signal": short_signal, "conditions": {"trend": short_trend, "resistance": short_resistance, "ema": short_ema, "volume": c_volume, "rsi": short_rsi}},
    }


def ask_claude(direction: str, signal: dict) -> str:
    if direction == "long":
        prompt = f"""You are a trading assistant reviewing a BTC long setup.

All 5 confluence conditions are confirmed on the Daily chart:
- Bullish trend: price ({signal['price']}) > 21 EMA ({signal['ema21']}), 21 EMA > 50 MA ({signal['ma50']}), higher highs/lows confirmed
- Price within 1.5% of swing low support ({signal['support']})
- Price within 2% of 21 EMA
- Volume is {signal['vol_ratio']}x the 20-bar average
- RSI rebounded from 40-50 zone, now {signal['rsi']} and rising

Should we enter a LONG position on BTCUSDT?
Reply with 'buy' or 'hold' followed by a one-sentence reason."""
    else:
        prompt = f"""You are a trading assistant reviewing a BTC short setup.

All 5 confluence conditions are confirmed on the Daily chart:
- Bearish trend: price ({signal['price']}) < 21 EMA ({signal['ema21']}), 21 EMA < 50 MA ({signal['ma50']}), lower highs/lows confirmed
- Price within 2% of swing high resistance ({signal['resistance']})
- Price within 2% of 21 EMA (from below)
- Volume is {signal['vol_ratio']}x the 20-bar average
- RSI rejected from 50-60 zone, now {signal['rsi']} and falling

Should we enter a SHORT position on BTCUSDT?
Be cautious of short squeezes. Reply with 'sell' or 'hold' followed by a one-sentence reason."""

    msg = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text.strip().lower()


def execute_trade(side: str, price: float, claude_reason: str = ""):
    MIN_QTY   = 0.001
    total_qty = round(ORDER_QTY_USDT / price, 3)
    close_side = "Sell" if side == "Buy" else "Buy"
    direction  = "LONG" if side == "Buy" else "SHORT"
    stop_loss  = round(price * (1 - STOP_LOSS_PCT / 100) if side == "Buy"
                       else price * (1 + STOP_LOSS_PCT / 100), 2)

    tp1_price, tp2_price = find_tp_levels(price, side)

    # Split 30 / 40 / 30 — fall back to single TP if qty too small to split
    qty_tp1 = round(total_qty * 0.30, 3)
    qty_tp2 = round(total_qty * 0.40, 3)
    qty_tp3 = round(total_qty - qty_tp1 - qty_tp2, 3)
    tiered  = qty_tp1 >= MIN_QTY and qty_tp2 >= MIN_QTY and qty_tp3 >= MIN_QTY

    close_opposing_position(SYMBOL, side)

    # Main order with SL only (TPs placed as separate reduce-only orders)
    result = session.place_order(
        category="linear", symbol=SYMBOL, side=side,
        orderType="Market", qty=str(total_qty),
        stopLoss=str(stop_loss), slTriggerBy="MarkPrice",
    )
    logger.info("Order placed: %s", result)

    tp1_order_id = tp2_order_id = ""
    if tiered:
        r1 = session.place_order(
            category="linear", symbol=SYMBOL, side=close_side,
            orderType="Limit", qty=str(qty_tp1), price=str(tp1_price),
            reduceOnly=True, timeInForce="GTC",
        )
        tp1_order_id = r1["result"].get("orderId", "")
        r2 = session.place_order(
            category="linear", symbol=SYMBOL, side=close_side,
            orderType="Limit", qty=str(qty_tp2), price=str(tp2_price),
            reduceOnly=True, timeInForce="GTC",
        )
        tp2_order_id = r2["result"].get("orderId", "")
    else:
        # Order too small to split — single limit TP at tp2
        session.place_order(
            category="linear", symbol=SYMBOL, side=close_side,
            orderType="Limit", qty=str(total_qty), price=str(tp2_price),
            reduceOnly=True, timeInForce="GTC",
        )

    _open_positions[side] = {
        "side":         side,
        "entry_price":  price,
        "original_qty": total_qty,
        "qty_tp3":      qty_tp3 if tiered else 0,
        "tp1_price":    tp1_price,
        "tp2_price":    tp2_price,
        "stop_loss":    stop_loss,
        "tp1_order_id": tp1_order_id,
        "tp2_order_id": tp2_order_id,
        "tiered":       tiered,
        "tp1_hit":      False,
        "tp2_hit":      False,
        "tp3_hit":      False,
        "be_moved":     False,
    }

    log_trade_entry(side, price, total_qty, stop_loss, tp1_price, claude_reason)

    tp_note = (f"TP1 (30%): ${tp1_price:,.2f}\nTP2 (40%): ${tp2_price:,.2f}\nTP3 (30%): momentum exhaustion"
               if tiered else f"TP: ${tp2_price:,.2f} (single — order too small to split)")
    send_telegram(
        f"{'✅' if side == 'Buy' else '🔻'} {direction} opened on {SYMBOL}\n"
        f"Entry: ${price:,.2f} | Qty: {total_qty} BTC\n"
        f"Stop Loss: ${stop_loss:,.2f}\n{tp_note}"
    )


# --- Scheduled scan ---

async def run_scan():
    logger.info("Swing Trader scan running...")
    try:
        signals = check_signals()
        logger.info("Signals: %s", signals)
        check_closed_positions()

        # Long setup
        if signals["long"]["signal"]:
            logger.info("Long conditions met — consulting Claude...")
            response = ask_claude("long", signals)
            logger.info("Claude (long): %s", response)
            if response.startswith("buy"):
                execute_trade("Buy", signals["price"])
            else:
                send_telegram(f"⏸ Long setup triggered but Claude said hold.\nReason: {response}")

        # Short setup — only if daily trend is clearly bearish
        elif signals["short"]["signal"]:
            logger.info("Short conditions met — consulting Claude...")
            response = ask_claude("short", signals)
            logger.info("Claude (short): %s", response)
            if response.startswith("sell"):
                execute_trade("Sell", signals["price"])
            else:
                send_telegram(f"⏸ Short setup triggered but Claude said hold.\nReason: {response}")

        else:
            logger.info("No conditions met — no trade.")

    except Exception as e:
        logger.error("Scan error: %s", e)
        send_telegram(f"⚠️ Swing Trader error: {e}")


# --- App startup/shutdown ---

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    scheduler.add_job(run_scan, "interval", hours=2)
    scheduler.add_job(check_closed_positions, "interval", minutes=15)
    scheduler.start()
    logger.info("Swing Trader started — scanning every 2 hours, position check every 15 minutes.")
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


@app.get("/performance")
def performance():
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database not available")
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE exit_price IS NOT NULL),
                COUNT(*) FILTER (WHERE pnl_usdt > 0),
                COUNT(*) FILTER (WHERE pnl_usdt <= 0),
                ROUND(AVG(pnl_usdt) FILTER (WHERE exit_price IS NOT NULL)::numeric, 2),
                ROUND(SUM(pnl_usdt) FILTER (WHERE exit_price IS NOT NULL)::numeric, 2),
                ROUND(MAX(pnl_usdt)::numeric, 2),
                ROUND(MIN(pnl_usdt)::numeric, 2)
            FROM trades
        """)
        row   = cur.fetchone()
        total = int(row[0] or 0)
        wins  = int(row[1] or 0)
        stats = {
            "total_trades":   total,
            "wins":           wins,
            "losses":         int(row[2] or 0),
            "avg_pnl_usdt":   float(row[3] or 0),
            "total_pnl_usdt": float(row[4] or 0),
            "best_trade":     float(row[5] or 0),
            "worst_trade":    float(row[6] or 0),
            "win_rate":       f"{round(wins / total * 100, 1)}%" if total > 0 else "N/A",
        }
        cur.execute("""
            SELECT side, entry_price, exit_price, pnl_usdt, pnl_pct,
                   exit_reason, entry_time, exit_time
            FROM trades ORDER BY entry_time DESC LIMIT 20
        """)
        cols = ["side","entry_price","exit_price","pnl_usdt","pnl_pct",
                "exit_reason","entry_time","exit_time"]
        stats["recent_trades"] = [dict(zip(cols, r)) for r in cur.fetchall()]
    return stats


@app.post("/test-long")
async def test_long(request: Request):
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {BOT_SECRET}":
        raise HTTPException(status_code=401, detail="invalid secret")

    try:
        resp    = session.get_kline(category="linear", symbol=SYMBOL, interval="D", limit=200)
        candles = list(reversed(resp["result"]["list"]))
        closes  = [float(c[4]) for c in candles]
        highs   = [float(c[2]) for c in candles]
        lows    = [float(c[3]) for c in candles]
        vols    = [float(c[5]) for c in candles]

        ema21      = calc_ema(closes, 21)[-1]
        ma50       = calc_sma(closes, 50)[-1]
        rsi_series = calc_rsi(closes, 14)
        vol_avg    = calc_sma(vols, 20)[-1]
        support    = find_pivot_lows(lows)
        price      = closes[-1]

        fake_signal = {
            "price":     price,
            "ema21":     round(ema21, 2),
            "ma50":      round(ma50, 2),
            "rsi":       round(rsi_series[-1], 2),
            "vol_ratio": round(vols[-1] / vol_avg, 2),
            "support":   round(support[-1], 2) if support else round(price * 0.985, 2),
        }

        response = ask_claude("long", fake_signal)
        logger.info("Test long — Claude: %s", response)

        if response.startswith("buy"):
            execute_trade("Buy", price)
            return {"status": "executed", "claude": response, "price": price}
        else:
            send_telegram(f"⏸ Test long: Claude said hold.\nReason: {response}")
            return {"status": "held", "claude": response}

    except Exception as e:
        logger.error("Test long error: %s", e, exc_info=True)
        return {"status": "error", "detail": str(e)}


@app.post("/test-short")
async def test_short(request: Request):
    auth = request.headers.get("Authorization", "")
    if auth != f"Bearer {BOT_SECRET}":
        raise HTTPException(status_code=401, detail="invalid secret")

    try:
        resp    = session.get_kline(category="linear", symbol=SYMBOL, interval="D", limit=200)
        candles = list(reversed(resp["result"]["list"]))
        closes  = [float(c[4]) for c in candles]
        highs   = [float(c[2]) for c in candles]
        lows    = [float(c[3]) for c in candles]
        vols    = [float(c[5]) for c in candles]

        ema21      = calc_ema(closes, 21)[-1]
        ma50       = calc_sma(closes, 50)[-1]
        rsi_series = calc_rsi(closes, 14)
        vol_avg    = calc_sma(vols, 20)[-1]
        resistance = find_pivot_highs(highs)
        price      = closes[-1]

        fake_signal = {
            "price":      price,
            "ema21":      round(ema21, 2),
            "ma50":       round(ma50, 2),
            "rsi":        round(rsi_series[-1], 2),
            "vol_ratio":  round(vols[-1] / vol_avg, 2),
            "resistance": round(resistance[-1], 2) if resistance else round(price * 1.02, 2),
        }

        response = ask_claude("short", fake_signal)
        logger.info("Test short — Claude: %s", response)

        if response.startswith("sell"):
            execute_trade("Sell", price)
            return {"status": "executed", "claude": response, "price": price}
        else:
            send_telegram(f"⏸ Test short: Claude said hold.\nReason: {response}")
            return {"status": "held", "claude": response}

    except Exception as e:
        logger.error("Test short error: %s", e, exc_info=True)
        return {"status": "error", "detail": str(e)}


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


_open_positions: dict = {}

def check_closed_positions():
    try:
        resp    = session.get_positions(category="linear", symbol=SYMBOL)
        current = {p["side"]: p for p in resp["result"]["list"] if float(p.get("size", 0)) > 0}

        for side, pos_data in list(_open_positions.items()):
            entry_price  = pos_data.get("entry_price", 0)
            original_qty = pos_data.get("original_qty", pos_data.get("size", 0))
            close_side   = "Sell" if side == "Buy" else "Buy"

            if side not in current:
                # Position fully closed — determine reason
                try:
                    pnl_resp   = session.get_closed_pnl(category="linear", symbol=SYMBOL, limit=1)
                    pnl_record = pnl_resp["result"]["list"][0] if pnl_resp["result"]["list"] else {}
                    exit_price = float(pnl_record.get("avgExitPrice", 0))
                    closed_pnl = float(pnl_record.get("closedPnl", 0))
                    if pos_data.get("tp2_hit"):
                        exit_reason = "tp3"
                    elif pos_data.get("tp1_hit"):
                        exit_reason = "tp2"
                    elif closed_pnl > 0:
                        exit_reason = "tp1"
                    else:
                        exit_reason = "stop_loss"
                except Exception:
                    exit_price  = 0
                    exit_reason = "unknown"

                pnl_usdt, pnl_pct = log_trade_exit(SYMBOL, side, exit_price, exit_reason, entry_price, original_qty)
                emoji = "🟢" if pnl_usdt >= 0 else "🔴"
                send_telegram(
                    f"🔔 {SYMBOL} {side} fully closed ({exit_reason.replace('_', ' ')})\n"
                    f"Exit: ${exit_price:,.2f}\n"
                    f"{emoji} P&L: ${pnl_usdt:+.2f} ({pnl_pct:+.2f}%)"
                )
                del _open_positions[side]
                continue

            # Position still open — check tiered TP progress
            if not pos_data.get("tiered"):
                continue

            current_size = float(current[side]["size"])

            # TP1 hit: size fell to ~70% or below
            if not pos_data["tp1_hit"] and current_size <= original_qty * 0.75:
                pos_data["tp1_hit"] = True
                try:
                    session.set_trading_stop(
                        category="linear", symbol=SYMBOL, positionIdx=0,
                        stopLoss=str(entry_price), tpslMode="Full",
                    )
                    pos_data["be_moved"] = True
                except Exception as e:
                    logger.error("Failed to move SL to break-even: %s", e)
                send_telegram(
                    f"🎯 TP1 hit — {SYMBOL} {side}\n"
                    f"30% closed near ${pos_data['tp1_price']:,.2f}\n"
                    f"SL moved to break-even: ${entry_price:,.2f}\n"
                    f"Watching TP2 at ${pos_data['tp2_price']:,.2f}"
                )

            # TP2 hit: size fell to ~30% or below
            elif pos_data["tp1_hit"] and not pos_data["tp2_hit"] and current_size <= original_qty * 0.35:
                pos_data["tp2_hit"] = True
                send_telegram(
                    f"🎯 TP2 hit — {SYMBOL} {side}\n"
                    f"40% closed near ${pos_data['tp2_price']:,.2f}\n"
                    f"Final 30% riding — watching for momentum exhaustion"
                )

            # TP3: momentum exhaustion check on remaining 30%
            elif pos_data["tp2_hit"] and not pos_data["tp3_hit"]:
                if check_momentum_exhaustion(side):
                    try:
                        session.place_order(
                            category="linear", symbol=SYMBOL, side=close_side,
                            orderType="Market", qty=str(current_size), reduceOnly=True,
                        )
                        pos_data["tp3_hit"] = True
                        send_telegram(
                            f"🎯 TP3 — Momentum Exhaustion — {SYMBOL} {side}\n"
                            f"Final 30% closed at market\n"
                            f"RSI overextended + volume climax confirmed"
                        )
                    except Exception as e:
                        logger.error("TP3 market close failed: %s", e)

        # Update size for still-open tracked positions; register any untracked ones
        for side, p in current.items():
            if side in _open_positions:
                _open_positions[side]["size"] = float(p.get("size", 0))
            else:
                _open_positions[side] = {
                    "side": side, "entry_price": float(p.get("avgPrice", 0)),
                    "original_qty": float(p.get("size", 0)), "tiered": False,
                    "tp1_hit": False, "tp2_hit": False, "tp3_hit": False, "be_moved": False,
                }

    except Exception as e:
        logger.error("Position check error: %s", e)


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
