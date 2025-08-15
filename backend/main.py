from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import math
import logging
from sqlalchemy import create_engine, Column, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Frontend path
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Database setup
Base = declarative_base()

class StockData(Base):
    __tablename__ = "stocks"
    id = Column(String, primary_key=True)  # Format: "AAPL_2025-08-11"
    symbol = Column(String)
    date = Column(DateTime)
    price = Column(Float)
    change = Column(Float)
    change_percent = Column(Float)
    historical_data = Column(String)  # JSON string

# Update the engine creation with folder creation
db_dir = os.path.join(os.path.dirname(__file__), "database")
db_path = os.path.join(db_dir, "stocks.db")
if not os.path.exists(db_dir):
    os.makedirs(db_dir)  # Create database folder if it doesn't exist
engine = create_engine(f"sqlite:///{db_path}")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def clean_float(value):
    """Ensure float values are finite and JSON-compatible"""
    try:
        value = float(value)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0

def get_cached_stock_data(symbol: str):
    """Retrieve cached data from database"""
    session = Session()
    today = datetime.now().date()
    record_id = f"{symbol}_{today}"
    
    try:
        cached_data = session.query(StockData).filter_by(id=record_id).first()
        if cached_data:
            return {
                "symbol": symbol,
                "name": cached_data.symbol,  # Name not cached, use symbol as fallback
                "current": {
                    "price": clean_float(cached_data.price),
                    "change": clean_float(cached_data.change),
                    "changePercent": clean_float(cached_data.change_percent)
                },
                "historical": json.loads(cached_data.historical_data) if cached_data.historical_data else []
            }
        return None
    except Exception as e:
        logger.error(f"Database error for {symbol}: {str(e)}")
        return None
    finally:
        session.close()

def cache_stock_data(symbol: str, data: dict):
    """Store data in database"""
    session = Session()
    today = datetime.now().date()
    record_id = f"{symbol}_{today}"
    
    try:
        new_record = StockData(
            id=record_id,
            symbol=symbol,
            date=today,
            price=clean_float(data["current"]["price"]),
            change=clean_float(data["current"]["change"]),
            change_percent=clean_float(data["current"]["changePercent"]),
            historical_data=json.dumps(data["historical"])
        )
        session.merge(new_record)
        session.commit()
    except Exception as e:
        logger.error(f"Failed to cache {symbol}: {str(e)}")
        session.rollback()
    finally:
        session.close()

def get_real_stock_data(symbol: str, days: int = 365):
    """Fetch stock data with caching and error handling"""
    # First try cache
    cached_data = get_cached_stock_data(symbol)
    if cached_data:
        logger.info(f"Using cached data for {symbol}")
        return cached_data

    # Fetch fresh data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError("No data returned from API")

        df = df[['Close']].reset_index()
        df.columns = ['date', 'close']
        df['close'] = df['close'].apply(clean_float)

        data = {
            "symbol": symbol,
            "name": stock.info.get('shortName', symbol),
            "current": {
                "price": clean_float(df.iloc[-1]['close']),
                "change": clean_float(df.iloc[-1]['close'] - df.iloc[-2]['close']),
                "changePercent": clean_float((df.iloc[-1]['close'] - df.iloc[-2]['close']) / df.iloc[-2]['close'] * 100)
            },
            "historical": [
                {
                    "date": date.strftime("%Y-%m-%d"), 
                    "close": clean_float(close)
                } for date, close in zip(df['date'], df['close'])
            ]
        }
        
        # Cache the new data
        cache_stock_data(symbol, data)
        return data
        
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {str(e)}")
        return None

@app.get("/api/companies")
async def list_companies():
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "JPM", "NFLX", "BTC-USD"]
    results = []
    
    for symbol in symbols:
        data = get_real_stock_data(symbol, days=7)
        if data:
            results.append({
                "symbol": data["symbol"],
                "name": data["name"],
                "price": data["current"]["price"],
                "change": data["current"]["change"],
                "changePercent": data["current"]["changePercent"]
            })
    return JSONResponse(content={"companies": results})

@app.get("/api/stock/{symbol}")
async def get_stock(symbol: str):
    data = get_real_stock_data(symbol, days=365)
    if not data:
        raise HTTPException(status_code=404, detail="Stock data not available")
    
    try:
        df = pd.DataFrame(data['historical'])
        df['date'] = pd.to_datetime(df['date'])
        df['close'] = df['close'].apply(clean_float)
        
        # Calculate analytics with NaN protection
        df['7d_ma'] = df['close'].rolling(7).mean().fillna(0)
        
        data['analytics'] = {
            '7d_ma': clean_float(df['7d_ma'].iloc[-1]),
            'volatility': clean_float(df['close'].std()),
            '52w_high': clean_float(df['close'].max()),
            '52w_low': clean_float(df['close'].min()),
            'avg_volume': clean_float(yf.Ticker(symbol).info.get('averageVolume', 0))
        }
        
        # Add next-day price prediction using linear regression
        if len(df) >= 2:
            x = np.arange(len(df)).reshape(-1, 1)
            y = df['close'].values
            slope, intercept = np.polyfit(x.flatten(), y, 1)
            next_day_pred = slope * len(df) + intercept
            data['analytics']['next_day_prediction'] = clean_float(next_day_pred)
        else:
            data['analytics']['next_day_prediction'] = clean_float(df['close'].iloc[-1])

        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Data processing error for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail="Data processing error")

@app.get("/api/debug/db")
async def debug_db():
    """Endpoint to check database contents"""
    session = Session()
    try:
        records = session.query(StockData).order_by(StockData.date.desc()).limit(5).all()
        return {
            "count": session.query(StockData).count(),
            "recent": [{
                "id": r.id,
                "price": r.price,
                "date": r.date.isoformat()
            } for r in records]
        }
    finally:
        session.close()

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)