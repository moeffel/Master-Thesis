import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 

# 1) FETCH from Yahoo in EUR
tickers = ["BTC-EUR", "ETH-EUR", "DOGE-EUR", "SOL-EUR"]
start_date = "2009-06-22" 
end_date   = "2025-06-22"

print("Lade Preisdaten in EUR...")
raw = yf.download(tickers, start=start_date, end=end_date, progress=False)["Close"]

# 2) CLEANUP: Spalten umbenennen
price_df = raw.rename(columns={
    "BTC-EUR": "BTC",
    "ETH-EUR": "ETH",
    "DOGE-EUR": "DOGE",
    "SOL-EUR": "SOL"
})

# 3) ENSURE proper index
price_df.index = pd.to_datetime(price_df.index)
price_df = price_df.sort_index().dropna()

# 4) CALCULATE Market Cap
print("Berechne Marktkapitalisierung...")
market_cap_df = pd.DataFrame(index=price_df.index)

for coin in price_df.columns:
    try:
        ticker_name = f"{coin}-USD" 
        ticker_info = yf.Ticker(ticker_name)
        
        circ_supply = ticker_info.info['circulatingSupply']
        
        market_cap_df[coin] = price_df[coin] * circ_supply
        print(f"  > Marktkapitalisierung für {coin} berechnet")

    except Exception as e:
        print(f"Daten für {coin} konnten nicht abgerufen werden: {e}")
        
print("Berechnung abgeschlossen.\n")


# 5) PLOT: 4 Zeilen, geteilte x-Achse, unabhängige y-Achsen
fig, axes = plt.subplots(
    nrows=4,
    ncols=1,
    sharex=True,
    figsize=(12, 10),
    constrained_layout=True
)

# // GEÄNDERT: Funktion zur Formatierung der Y-Achse in MILLIARDEN Euro
def billions_formatter_de(x, pos):
    """Formatiert große Zahlen in Milliarden ('Mrd.') mit Euro-Symbol."""
    # Wir verwenden .1f, um eine Dezimalstelle zu erlauben (z.B. 45,5 Mrd. €)
    return f'{x/1e9:.1f} Mrd. €'

for ax, coin in zip(axes, market_cap_df.columns):
    ax.plot(market_cap_df.index, market_cap_df[coin], label=coin)
    
    ax.set_ylabel(f"{coin} Marktkapitalisierung")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    # // GEÄNDERT: Wenden den neuen Milliarden-Formatierer an
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(billions_formatter_de))

axes[-1].set_xlabel("Datum") 
fig.suptitle("Marktkapitalisierung von Kryptowährungen (2020–2025)", fontsize=16)
plt.show()