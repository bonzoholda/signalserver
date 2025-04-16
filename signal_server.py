from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import random
import time

#import from binance related
from binance.client import Client
from binance.enums import *
import requests
import gc

session = requests.Session()
# Use the session
session.close()  # Release resources

app = FastAPI()

# ===================
# Load environment variables from .env file
# ===================
from dotenv import load_dotenv
load_dotenv()

# Retrieve API keys from environment variables
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')


# ===================
# Define the RPC URL and Chainlink oracle contract details
# ===================
RPC_URL = "https://polygon.meowrpc.com"  # Replace with your RPC endpoint
CHAINLINK_ORACLE_ADDRESS = "0xAB594600376Ec9fD91F8e885dADF0CE036862dE0"  # Example: MATIC/USD on Polygon
CHAINLINK_ABI = [
    {
        "inputs": [],
        "name": "latestRoundData",
        "outputs": [
            {"internalType": "uint80", "name": "roundId", "type": "uint80"},
            {"internalType": "int256", "name": "answer", "type": "int256"},
            {"internalType": "uint256", "name": "startedAt", "type": "uint256"},
            {"internalType": "uint256", "name": "updatedAt", "type": "uint256"},
            {"internalType": "uint80", "name": "answeredInRound", "type": "uint80"},
        ],
        "stateMutability": "view",
        "type": "function",
    }
]


price_data = []  # Store recent price data for indicators

# Fetch the latest price from Chainlink Oracle
def fetch_price():
    contract = web3.eth.contract(address=CHAINLINK_ORACLE_ADDRESS, abi=[
        {"inputs": [],
         "name": "latestRoundData",
         "outputs": [
             {"internalType": "uint80", "name": "roundId", "type": "uint80"},
             {"internalType": "int256", "name": "answer", "type": "int256"},
             {"internalType": "uint256", "name": "startedAt", "type": "uint256"},
             {"internalType": "uint256", "name": "updatedAt", "type": "uint256"},
             {"internalType": "uint80", "name": "answeredInRound", "type": "uint80"}
         ],
         "stateMutability": "view", "type": "function"}
    ])
    latest_data = contract.functions.latestRoundData().call()
    price = latest_data[1] / 1e8
    return float(price)


# =====================
# USE BINANCE PRICE FEED TO GENERATE SIGNALS
# =====================


client = Client(API_KEY, API_SECRET)

# Function to get historical data incrementally
def get_historical_data(symbol='POLUSDT', interval=Client.KLINE_INTERVAL_5MINUTE, limit=1000):
    try:
        # Fetch initial historical data
        klines = client.get_historical_klines(symbol, interval, limit=limit)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                           'close_time', 'quote_asset_volume', 'number_of_trades', 
                                           'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        
        # Convert and optimize data types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = df['close'].astype('float32')
        df['volume'] = df['volume'].astype('float32')
        return df
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return pd.DataFrame()

# Update DataFrame with new data
def update_historical_data(df, symbol='POLUSDT', interval=Client.KLINE_INTERVAL_5MINUTE):
    try:
        last_timestamp = int(df['timestamp'].iloc[-1].timestamp() * 1000)  # Convert to milliseconds
        
        # Fetch new data since the last timestamp
        klines = client.get_historical_klines(symbol, interval, start_str=last_timestamp)
        
        # Convert to DataFrame
        new_df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                               'close_time', 'quote_asset_volume', 'number_of_trades', 
                                               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        
        # Optimize new data types
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
        new_df['close'] = new_df['close'].astype('float32')
        new_df['volume'] = new_df['volume'].astype('float32')
        
        # Append only new rows
        updated_df = pd.concat([df, new_df]).drop_duplicates(subset='timestamp').reset_index(drop=True)
        
        # Release memory of the temporary DataFrame
        del new_df
        gc.collect()  # Force garbage collection
        
        return updated_df
    except Exception as e:
        print(f"Error updating historical data: {e}")
        return df

# Function to calculate RSI
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Release memory of temporary variables
    del delta, gain, loss, rs
    gc.collect()
    
    return df


# Example RSI calculation function (ensure you have your own or a proper implementation)
def calculate_rsi(df, period=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

######################divergence block 
def detect_rsi_divergence(df, rsi_period=14, window=5):
    """
    Detect RSI divergences (Bullish and Bearish) based on price action.
    :param df: DataFrame containing 'close' prices and RSI.
    :param rsi_period: Period for RSI calculation.
    :param window: Number of bars to consider for local extrema.
    :return: DataFrame with divergence signals.
    """
    
    # Ensure RSI is calculated
    df = calculate_rsi(df, period=rsi_period)
    
    # Identify local highs and lows in price
    df['local_low'] = df['close'].rolling(window=window, center=True).min()
    df['local_high'] = df['close'].rolling(window=window, center=True).max()
    
    # Identify local highs and lows in RSI
    df['rsi_low'] = df['rsi'].rolling(window=window, center=True).min()
    df['rsi_high'] = df['rsi'].rolling(window=window, center=True).max()
    
    # Detect bullish divergence (Price lower low, RSI higher low)
    bullish_condition = (
        (df['close'] < df['close'].shift(window)) &  # Lower low in price
        (df['rsi'] > df['rsi'].shift(window))  # Higher low in RSI
    )
    
    # Detect bearish divergence (Price higher high, RSI lower high)
    bearish_condition = (
        (df['close'] > df['close'].shift(window)) &  # Higher high in price
        (df['rsi'] < df['rsi'].shift(window))  # Lower high in RSI
    )
    
    # Apply signals
    df['divergence_signal'] = 0
    df['divergence_type'] = None
    df.loc[bullish_condition, 'divergence_signal'] = 1  # Buy signal
    df.loc[bullish_condition, 'divergence_type'] = 'bullish'
    df.loc[bearish_condition, 'divergence_signal'] = -1  # Sell signal
    df.loc[bearish_condition, 'divergence_type'] = 'bearish'
    
    gc.collect()
    return df

# Integrate RSI divergence into the main signal function
def generate_signals_with_rsi_divergence(df, local_window=5, rsi_period=14):
    df = generate_signals(df, local_window)
    df = detect_rsi_divergence(df, rsi_period, local_window)
    return df





##########################



def generate_signals(df, local_window=5):
    # Calculate moving averages
    df['sma10'] = df['close'].rolling(window=10).mean()
    df['sma200'] = df['close'].rolling(window=200).mean()
    df['sma9'] = df['close'].rolling(window=9).mean()
    
    # Calculate SMA and Envelopes
    period = 20
    deviation = 0.01
    df['sma'] = df['close'].rolling(window=period).mean()
    df['upper_band'] = df['sma'] * (1 + deviation)
    df['lower_band'] = df['sma'] * (1 - deviation)    

    # Calculate RSI
    df = calculate_rsi(df, period=14)

    # --- New logic for local extremes ---
    # Compute the rolling minimum and maximum over a defined window.
    # We use min_periods=local_window to ensure we only flag signals once we have enough data
    df['rolling_min'] = df['close'].rolling(window=local_window, min_periods=local_window).min()
    df['rolling_max'] = df['close'].rolling(window=local_window, min_periods=local_window).max()

    # To avoid potential floating point issues when checking for equality,
    # you can use np.isclose. Here we use a tolerance value
    tolerance = 1e-8

    # Buy signal: when the previous bar's close was the local minimum and now price increases
    buy_condition = (
        np.isclose(df['close'].shift(1), df['rolling_min'].shift(1), atol=tolerance) &
        (df['close'] > df['close'].shift(1))
    )
    
    # Sell signal: when the previous bar's close was the local maximum and now price decreases
    sell_condition = (
        np.isclose(df['close'].shift(1), df['rolling_max'].shift(1), atol=tolerance) &
        (df['close'] < df['close'].shift(1))
    )

    # Optionally, if you still want to use the envelope bands as an extra filter,
    # you could combine the conditions. For example:
    buy_condition = buy_condition & (df['close'] < df['lower_band'])
    sell_condition = sell_condition & (df['close'] > df['upper_band'])
    
    # Generate signals
    df['signal'] = 0  # Default: no signal
    df['signal_type'] = None  # Default: no signal type
    df.loc[buy_condition, 'signal'] = 1
    df.loc[buy_condition, 'signal_type'] = 'buy'
    df.loc[sell_condition, 'signal'] = -1
    df.loc[sell_condition, 'signal_type'] = 'sell'

    # Clean up (if needed)
    gc.collect()
    
    print("Signals generated based on local extremes and reversals.")
    return df


@app.get("/")
def root():
    return {"status": "Signal Server is running"}

@app.get("/api/signal")
def get_signal(pair: str = "POLUSDT"):
    # Simulated signal logic (replace with real TA logic)

    current_price = fetch_price()

    historical_data = get_historical_data()
    df = update_historical_data(historical_data)
    df_with_signals = generate_signals(df)
    latest_signal = df_with_signals.iloc[-1]['signal_type']
    last_signal = latest_signal
        
    df_withRSIdvg = generate_signals_with_rsi_divergence(df)
    confirmed_signal = df_withRSIdvg.iloc[-1]['divergence_type']
    dvg_signal = confirmed_signal
    
    if (last_signal == 'buy' or dvg_signal == 'bullish'):
        signal = 'long'
    elif (last_signal == 'sell' or dvg_signal== 'bearish' ):
        signal = 'short'
    else :
        signal = 'no-signals'

    return JSONResponse({
        "pair": pair.upper(),
        "signal": signal,
        "price": current_price,
        "timestamp": int(time.time())
    })