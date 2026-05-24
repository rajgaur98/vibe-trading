import os
import json
from datetime import datetime
from contextlib import contextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from vibe_trading.data.db import Database
from vibe_trading.runtime.scheduler import TradingScheduler

app = FastAPI(title="Vibe Trading API", description="REST endpoints for Vibe Trading Dashboard")

# Enable CORS for Next.js frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@contextmanager
def get_db_conn():
    """Context manager to yield a temporary read-only DuckDB connection."""
    db_instance = Database(read_only=True)
    db_instance.connect()
    try:
        yield db_instance.conn
    finally:
        db_instance.close()

@app.get("/api/status")
def get_status():
    mode = os.getenv("TRADING_MODE", "PAPER").upper()
    symbols = os.getenv("TRADING_SYMBOLS", "BTC/USDT,ETH/USDT").split(",")
    
    open_count = 0
    db_path = ""
    with get_db_conn() as conn:
        db_path = os.getenv("DATABASE_PATH", "data/vibe_trading.db")
        try:
            open_count = conn.execute("SELECT count(*) FROM open_positions").fetchone()[0]
        except Exception:
            pass

    return {
        "status": "online",
        "mode": mode,
        "symbols": symbols,
        "open_positions_count": open_count,
        "database_path": db_path,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

@app.post("/api/trigger-tick")
def trigger_tick():
    try:
        scheduler = TradingScheduler()
        scheduler.sync_and_evaluate()
        return {"status": "success", "message": "On-demand execution tick completed."}
    except Exception as e:
        err_str = str(e)
        if "lock" in err_str.lower() or "io error" in err_str.lower():
            raise HTTPException(
                status_code=409,
                detail="Database is currently locked by the background trading daemon. Please wait a few seconds and try again."
            )
        raise HTTPException(status_code=500, detail=err_str)

@app.get("/api/metrics")
def get_metrics():
    trades = []
    balance = 10000.0
    peak_balance = 10000.0
    equity_curve = []
    
    with get_db_conn() as conn:
        try:
            trades = conn.execute("SELECT realized_pnl, result FROM trades").fetchall()
        except Exception:
            pass
            
        try:
            balance_res = conn.execute(
                "SELECT balance, peak_balance FROM portfolio_state ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if balance_res:
                balance = balance_res[0]
                peak_balance = balance_res[1]
        except Exception:
            pass
            
        try:
            equity_res = conn.execute(
                "SELECT timestamp, balance FROM portfolio_state ORDER BY timestamp ASC"
            ).fetchall()
            equity_curve = [{"timestamp": r[0].isoformat() + "Z", "balance": r[1]} for r in equity_res]
        except Exception:
            pass
        
    total_trades = len(trades)
    wins = sum(1 for t in trades if t[1] == 'win')
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    total_pnl = sum(t[0] for t in trades)
    avg_return = (total_pnl / total_trades) if total_trades > 0 else 0.0
    
    gross_profits = sum(t[0] for t in trades if t[0] > 0)
    gross_losses = sum(abs(t[0]) for t in trades if t[0] < 0)
    
    if gross_losses > 0:
        profit_factor = gross_profits / gross_losses
    else:
        profit_factor = gross_profits if gross_profits > 0 else 1.0
        
    # Calculate drawdown
    drawdown = 0.0
    if peak_balance > 0:
        drawdown = ((peak_balance - balance) / peak_balance) * 100
        
    if not equity_curve:
        equity_curve = [{"timestamp": datetime.utcnow().isoformat() + "Z", "balance": 10000.0}]
        
    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "avg_return": round(avg_return, 2),
        "profit_factor": round(profit_factor, 2),
        "balance": round(balance, 2),
        "peak_balance": round(peak_balance, 2),
        "drawdown": round(drawdown, 2),
        "equity_curve": equity_curve
    }

@app.get("/api/positions")
def get_positions():
    positions = []
    with get_db_conn() as conn:
        try:
            res = conn.execute(
                "SELECT symbol, side, entry_time, entry_price, size_usd, stop_price, take_profit_price FROM open_positions ORDER BY entry_time DESC"
            ).fetchall()
            for r in res:
                positions.append({
                    "symbol": r[0],
                    "side": r[1],
                    "entry_time": r[2].isoformat() + "Z" if r[2] else None,
                    "entry_price": r[3],
                    "size_usd": r[4],
                    "stop_price": r[5],
                    "take_profit_price": r[6]
                })
        except Exception:
            pass
    return positions

@app.get("/api/trades")
def get_trades():
    trades = []
    with get_db_conn() as conn:
        try:
            res = conn.execute("""
                SELECT trade_id, symbol, action, entry_time, entry_price, close_time, close_price, size_usd, realized_pnl, result 
                FROM trades 
                ORDER BY close_time DESC
            """).fetchall()
            for r in res:
                trades.append({
                    "trade_id": r[0],
                    "symbol": r[1],
                    "action": r[2],
                    "entry_time": r[3].isoformat() + "Z" if r[3] else None,
                    "entry_price": r[4],
                    "close_time": r[5].isoformat() + "Z" if r[5] else None,
                    "close_price": r[6],
                    "size_usd": r[7],
                    "realized_pnl": r[8],
                    "result": r[9]
                })
        except Exception:
            pass
    return trades

@app.get("/api/decisions")
def get_decisions(limit: int = 30):
    decisions = []
    with get_db_conn() as conn:
        try:
            res = conn.execute(f"""
                SELECT decision_id, timestamp, symbol, action, stop_loss_strategy, take_profit_strategy, risk_reward_ratio, reasoning_summary, agent_transcripts 
                FROM decision_log 
                ORDER BY timestamp DESC 
                LIMIT {limit}
            """).fetchall()
            for r in res:
                try:
                    transcripts = json.loads(r[8]) if r[8] else {}
                except Exception:
                    transcripts = {}
                    
                decisions.append({
                    "decision_id": r[0],
                    "timestamp": r[1].isoformat() + "Z" if r[1] else None,
                    "symbol": r[2],
                    "action": r[3],
                    "stop_loss_strategy": r[4],
                    "take_profit_strategy": r[5],
                    "risk_reward_ratio": r[6],
                    "reasoning_summary": r[7],
                    "agent_transcripts": transcripts
                })
        except Exception:
            pass
    return decisions

@app.get("/api/candles")
def get_candles(
    symbol: str = Query(..., description="e.g. BTC/USDT"),
    timeframe: str = Query("4h", description="e.g. 4h, 1d"),
    limit: int = Query(100, description="Number of candles to return")
):
    candles = []
    with get_db_conn() as conn:
        try:
            res = conn.execute("""
                SELECT timestamp, open, high, low, close, volume 
                FROM candles 
                WHERE symbol = ? AND timeframe = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (symbol, timeframe, limit)).fetchall()
            
            # Reverse to chronological order (oldest to newest)
            res.reverse()
            
            for r in res:
                candles.append({
                    "time": int(r[0].timestamp()),
                    "open": r[1],
                    "high": r[2],
                    "low": r[3],
                    "close": r[4],
                    "volume": r[5]
                })
        except Exception:
            pass
    return candles
