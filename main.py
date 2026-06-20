import os
import logging

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from pybit.unified_trading import HTTP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("execution-bot")

app = FastAPI(title="Trading Execution Bot")

BOT_SECRET = os.environ.get("RAILWAY_BOT_SECRET", "")
BYBIT_API_KEY = os.environ.get("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.environ.get("BYBIT_API_SECRET", "")
BYBIT_DEMO = os.environ.get("BYBIT_DEMO", "true").lower() == "true"
ORDER_QTY_USDT = float(os.environ.get("ORDER_QTY_USDT", "10"))
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", "2"))
TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", "4"))

session = HTTP(demo=BYBIT_DEMO, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)


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

    logger.info("Execute request: %s", req)

    if req.action not in ("buy", "sell"):
        return {"status": "skipped", "reason": f"action is {req.action}"}

    if not req.price:
        raise HTTPException(status_code=400, detail="price is required to size order")

    side = "Buy" if req.action == "buy" else "Sell"
    qty = round(ORDER_QTY_USDT / req.price, 3)

    try:
        close_result = close_opposing_position(req.symbol, side)

        if side == "Buy":
            stop_loss = req.price * (1 - STOP_LOSS_PCT / 100)
            take_profit = req.price * (1 + TAKE_PROFIT_PCT / 100)
        else:
            stop_loss = req.price * (1 + STOP_LOSS_PCT / 100)
            take_profit = req.price * (1 - TAKE_PROFIT_PCT / 100)

        result = session.place_order(
            category="linear",
            symbol=req.symbol,
            side=side,
            orderType="Market",
            qty=str(qty),
            stopLoss=str(round(stop_loss, 2)),
            takeProfit=str(round(take_profit, 2)),
        )
        logger.info("Order result: %s", result)
        return {"status": "executed", "order": result, "closed_position": close_result}
    except Exception as exc:
        logger.error("Order failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


def close_opposing_position(symbol: str, new_side: str) -> dict | None:
    """If an open position exists on the opposite side of new_side, close it with a reduce-only market order."""
    positions = session.get_positions(category="linear", symbol=symbol)
    for pos in positions["result"]["list"]:
        size = float(pos.get("size", 0))
        if size <= 0:
            continue
        existing_side = pos["side"]  # "Buy" (long) or "Sell" (short)
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
