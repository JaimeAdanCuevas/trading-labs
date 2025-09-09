import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import mplfinance as mpf
import numpy as np

# ==============================
# 1. Descargar datos
# ==============================
tickers = ["AAPL", "MSFT", "^GSPC", "EURUSD=X", "BTC-USD", "INTC", "QQQ"]
data = yf.download(tickers, start="2022-01-01", end="2023-12-31", interval="1d")
ticker_name = "QQQ"
ticker = "QQQ"

# Solo usamos AAPL como ejemplo para estrategia
df = data["Close"][ticker].to_frame()
df.columns = ["Close"]

# ==============================
# 2. Calcular SMAs
# ==============================
df["SMA20"] = df["Close"].rolling(window=20).mean() #Promedio aritmetico de cierre durante un numero fijo de periodos
df["SMA50"] = df["Close"].rolling(window=50).mean()

# ==============================
# 3. Estrategia de cruces
# ==============================
df["Signal"] = 0
df["Signal"] = np.where(df["SMA20"] > df["SMA50"], 1, -1)  # 1 = compra, -1 = venta
df["Return"] = df["Close"].pct_change(fill_method=None).fillna(0)
df["Strategy"] = df["Signal"].shift(1) * df["Return"]

# ==============================
# 4. Evaluación de resultados
# ==============================
cumulative_return = (1 + df["Strategy"]).cumprod()
total_return = cumulative_return.iloc[-1] - 1

# Drawdown
cum_max = cumulative_return.cummax()
drawdown = (cumulative_return - cum_max) / cum_max
max_drawdown = drawdown.min()

print("📊 Resultados de estrategia en " + ticker + ":")
print(f"Ganancia/Pérdida acumulada: {total_return:.2%}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# ==============================
# 5. Graficar velas + SMAs
# ==============================
# Preparar datos OHLC de AAPL
ticker = yf.download(ticker, start="2024-01-01", end="2025-09-09", interval="1d")

# Si hay MultiIndex, simplificar columnas
if isinstance(ticker.columns, pd.MultiIndex):
    ticker.columns = ticker.columns.get_level_values(0)

ticker = ticker[["Open", "High", "Low", "Close", "Volume"]]
ticker = ticker.dropna()

ticker[["Open","High","Low","Close","Volume"]] = ticker[["Open","High","Low","Close","Volume"]].astype(float)


ticker.index = pd.to_datetime(ticker.index).tz_localize(None)

print(ticker.dtypes)
print(ticker.isna().sum())

mpf.plot(
    ticker,
    type="candle",
    style="charles",
    title=f"{ticker_name} con SMA20 y SMA50",
    mav=(20, 50),
    mavcolors=("green", "red"),
    volume=True
)

# ==============================
# 6. Graficar estrategia
# ==============================
plt.figure(figsize=(12,6))
plt.plot(cumulative_return, label="Estrategia (SMA20/50)")
plt.plot(drawdown, label="Drawdown", color="red", alpha=0.5)
plt.legend()
plt.title("Rendimiento Estrategia y Drawdown")
plt.show()