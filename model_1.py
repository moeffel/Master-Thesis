# -*- coding: utf-8 -*-
import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timezone, timedelta
import ta
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation
import requests
import time
import traceback
import streamlit as st
from bs4 import BeautifulSoup
import re

# Stellen Sie sicher, dass scikit-learn installiert ist: pip install scikit-learn
try:
    import sklearn # Wird von PyPortfolioOpt für manche Kovarianz-Methoden benötigt
except ImportError:
    st.error("FEHLER: Bibliothek 'scikit-learn' nicht gefunden. Bitte installieren Sie sie mit `pip install scikit-learn` und starten Sie Streamlit neu.")
    st.stop()


# --- Globale Hilfsfunktion für HTML-Links ---
def make_html_link_clickable(link_val, text_to_display=None):
    if text_to_display is None: text_to_display = link_val
    if link_val and isinstance(link_val, str) and link_val.startswith("http"):
        return f'<a target="_blank" href="{link_val}">{text_to_display}</a>'
    return "N/A" if pd.isna(link_val) else str(link_val)

# --- Dashboard Titel und Seitenkonfiguration ---
st.set_page_config(layout="wide", page_title="CryptoPro Optimizer & Trading AI")
st.title("🚀 CryptoPro Optimizer & Trading AI")

# --- Sidebar für Konfigurationen ---
st.sidebar.header("⚙️ Konfigurationen")

# Grundkonfiguration
NUMBER_OF_TOP_COINS_TO_SCAN = st.sidebar.slider("Anzahl Top Coins (CoinGecko)", 30, 300, 50, 10, help="CoinGecko Top Coins für Analyse (Max 300 wg. API Limits). Weniger Coins = schnellere Ladezeit.")
TARGET_EXCHANGE = 'binance'
DEFAULT_CURRENT_PORTFOLIO_ALLOCATION_STR = "{'BTC/USDT': 0.40, 'ETH/USDT': 0.30, 'USDC': 0.30}"
DEFAULT_CURRENT_PORTFOLIO_VALUE_USD = 10000

current_allocation_str = st.sidebar.text_area("Akt. Portfolio Allokation (Python Dict)", DEFAULT_CURRENT_PORTFOLIO_ALLOCATION_STR, height=100, help="Format: {'BTC/USDT': 0.5, 'USDC': 0.5}. 'USDC' oder 'USDT' als Cash-Symbol verwenden.")
try:
    CURRENT_PORTFOLIO_ALLOCATION = eval(current_allocation_str)
    if not isinstance(CURRENT_PORTFOLIO_ALLOCATION, dict):
        st.sidebar.error("Ungültiges Format für Portfolio Allokation."); CURRENT_PORTFOLIO_ALLOCATION = eval(DEFAULT_CURRENT_PORTFOLIO_ALLOCATION_STR)
except Exception as e:
    st.sidebar.error(f"Fehler Parsen Portfolio Allokation: {e}"); CURRENT_PORTFOLIO_ALLOCATION = eval(DEFAULT_CURRENT_PORTFOLIO_ALLOCATION_STR)

CURRENT_PORTFOLIO_VALUE_USD = st.sidebar.number_input("Aktueller Portfolio Wert (USD)", min_value=100, value=DEFAULT_CURRENT_PORTFOLIO_VALUE_USD, step=100, help="Gesamtwert deines Krypto-Portfolios in USD.")

MAX_ASSETS_IN_OPTIMIZED_PORTFOLIO = st.sidebar.slider("Max. Assets im optimierten Portfolio", 2, 10, 5, 1, help="Maximale Anzahl Assets im optimierten Zielportfolio.")
MIN_USD_ALLOCATION_PER_ASSET = st.sidebar.number_input("Min. USD Allokation pro Asset", min_value=20, max_value=1000, value=100, step=10, help="Minimaler USD-Betrag pro Asset im optimierten Portfolio (für diskrete Allokation).")
RISK_FREE_RATE = st.sidebar.slider("Risikofreier Zinssatz (p.a.)", 0.00, 0.05, 0.02, 0.005, format="%.3f", help="Wird für Sharpe Ratio Berechnung benötigt.")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Portfolio Optimierung - Zeitfenster (Kurzfristig)")
OPTIMIZATION_MU_SPAN_DAYS = st.sidebar.slider("Optimierung: Span für erw. Renditen (EMA, Tage)", 10, 90, 22, 1, help="Kürzerer Span = stärkerer Fokus auf kurzfristige Rendite-Trends für erwartete Renditen (mu).")
OPTIMIZATION_COV_HISTORY_DAYS = st.sidebar.slider("Optimierung: Historie für Kovarianz (Tage)", 20, 120, 44, 1, help="Anzahl Tage für die Kovarianzmatrix-Berechnung. Kürzere Periode = kurzfristigere Korrelationen.")
MAX_EXPECTED_ANNUAL_RETURN_CAP = st.sidebar.slider("Optimierung: Max. erw. Jahresrendite (Cap, %)", 50, 1000, 300, 10, help="Obergrenze für die annualisierte erwartete Rendite eines Assets, um unrealistische Werte zu kappen (z.B. 300% = 3.0).") / 100

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Trading Setup Parameter")
RISK_PER_TRADE_PORTFOLIO_PCT = st.sidebar.slider("Risiko pro Trade (% des Portfolio-Werts)", 0.1, 2.0, 0.5, 0.1, help="Maximaler Verlust pro Trade als Prozentsatz des gesamten Portfoliowerts.") / 100
USE_ATR_FOR_SL_TP = st.sidebar.checkbox("ATR für Stop-Loss & Take-Profit nutzen?", True, help="Stop-Loss und Take-Profit basierend auf ATR.")
ATR_PERIOD = st.sidebar.slider("ATR Periode für Vola-Penalty & Basis-TA (Tage)", 7, 28, 14, 1, help="Periode für ATR-Berechnung für Volatilitäts-Penalty und allgemeine TA-Anzeige.")
ATR_PERIOD_TRADE = st.sidebar.slider("ATR Periode für Trading Setups (Tage)", 7, 28, 14, 1, help="Periode für ATR-Berechnung speziell für Trading-Setups.", key="atr_trade_setup", disabled=not USE_ATR_FOR_SL_TP)
ATR_SL_MULTIPLIER_TRADE = st.sidebar.slider("ATR Stop-Loss Multiplikator für Trades", 1.0, 4.0, 2.0, 0.1, help="Vielfaches des ATR für Stop-Loss bei Trading Setups.", disabled=not USE_ATR_FOR_SL_TP, key="atr_sl_trade_setup")
DEFAULT_SL_PCT_TRADE = st.sidebar.slider("Fallback Stop-Loss (%) für Trades", 1.0, 10.0, 3.0, 0.1, format="%.1f", help="Prozentualer Stop-Loss für Trading Setups, wenn ATR nicht genutzt wird.", disabled=USE_ATR_FOR_SL_TP) / 100
TAKE_PROFIT_RR_RATIO_1 = st.sidebar.slider("Take-Profit 1 (Risk/Reward Ratio)", 1.0, 3.0, 1.5, 0.1, help="Erstes Take-Profit-Ziel als Vielfaches des Risikos (Stop-Loss-Distanz).")
TAKE_PROFIT_RR_RATIO_2 = st.sidebar.slider("Take-Profit 2 (Risk/Reward Ratio)", 1.5, 5.0, 2.5, 0.1, help="Zweites Take-Profit-Ziel als Vielfaches des Risikos.")


st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Portfolio Optimierung - Vola Penalty (Asset Selektion)")
ENABLE_VOLA_PENALTY = st.sidebar.checkbox("Vola-Penalty für Asset-Selektion aktivieren?", True, help="Reduziert erw. Rendite für Assets mit aktuell hoher relativer Volatilität (basierend auf norm. ATR) bei der Portfolio-Zusammenstellung.")
VOLA_PENALTY_MEDIAN_THRESHOLD = st.sidebar.slider("Vola Penalty: Schwelle ggü. Median (x-fach)", 1.1, 3.0, 1.75, 0.05, help="Asset-Vola muss > X mal Median-Vola sein für Penalty 1.", disabled=not ENABLE_VOLA_PENALTY)
VOLA_PENALTY_SELF_THRESHOLD = st.sidebar.slider("Vola Penalty: Schwelle ggü. Eigener Historie (x-fach)", 1.1, 3.0, 1.5, 0.05, help="Asset-Vola muss > X mal eigener roll. Vola sein für Penalty 2.", disabled=not ENABLE_VOLA_PENALTY)
VOLA_PENALTY_FACTOR_MEDIAN = st.sidebar.slider("Vola Penalty: Faktor ggü. Median", 0.01, 0.20, 0.05, 0.01, format="%.2f", help="Stärke der Strafe für Bedingung 1.", disabled=not ENABLE_VOLA_PENALTY)
VOLA_PENALTY_FACTOR_SELF = st.sidebar.slider("Vola Penalty: Faktor ggü. Eigener Historie", 0.01, 0.25, 0.07, 0.01, format="%.2f", help="Stärke der Strafe für Bedingung 2.", disabled=not ENABLE_VOLA_PENALTY)
VOLA_PENALTY_MAX_PENALTY_MEDIAN = st.sidebar.slider("Vola Penalty: Max Strafe Median (%)", 5, 50, 30, 1, help="Maximale Reduktion der erw. Rendite durch Penalty 1.", disabled=not ENABLE_VOLA_PENALTY) / 100
VOLA_PENALTY_MAX_PENALTY_SELF = st.sidebar.slider("Vola Penalty: Max Strafe Eigene Historie (%)", 5, 60, 40, 1, help="Maximale Reduktion der erw. Rendite durch Penalty 2.", disabled=not ENABLE_VOLA_PENALTY) / 100
VOLA_PENALTY_ATR_HISTORY_WINDOW = st.sidebar.slider("Vola Penalty: Fenster Eigene Historie (Tage)", 5, 50, 20, 1, help="Zeitraum für den Vergleich der aktuellen Vola zur historischen.", disabled=not ENABLE_VOLA_PENALTY)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Dynamisches Cash Management (F&G)")
ENABLE_DYNAMIC_CASH = st.sidebar.checkbox("Autom. Cash-Management (via F&G-TA) aktivieren?", True, help="Passt Cash-Anteil basierend auf F&G Index TA an.")
FNG_HISTORY_DAYS = 90
FNG_SMA_SHORT = st.sidebar.slider("F&G SMA Kurz (Tage)", 3, 14, 7, 1, help="Kurzer gleitender Durchschnitt für F&G Index.", disabled=not ENABLE_DYNAMIC_CASH)
FNG_SMA_LONG = st.sidebar.slider("F&G SMA Lang (Tage)", 15, 50, 21, 1, help="Langer gleitender Durchschnitt für F&G Index.", disabled=not ENABLE_DYNAMIC_CASH)
FNG_TA_EXTREME_GREED_SELL_THRESHOLD = st.sidebar.slider("F&G Gier-Schwelle für Cash-Erhöhung", 65, 85, 75, 1, help="Oberhalb dieser Schwelle + Signal -> Cash erhöhen.", disabled=not ENABLE_DYNAMIC_CASH)
FNG_TA_EXTREME_FEAR_BUY_THRESHOLD = st.sidebar.slider("F&G Angst-Schwelle für Cash-Reduktion", 15, 35, 25, 1, help="Unterhalb dieser Schwelle + Signal -> Cash reduzieren.", disabled=not ENABLE_DYNAMIC_CASH)
DYNAMIC_CASH_ADJUSTMENT_PCT = st.sidebar.slider("Autom. Cash: Anpassung Cash-Anteil (%)", 0.05, 0.30, 0.10, 0.01, format="%.2f", help="Schrittweite der Cash-Anpassung bei Signal.", disabled=not ENABLE_DYNAMIC_CASH)
MIN_TOTAL_CASH_PCT = st.sidebar.slider("Min. Gesamt-Cash-Anteil (%)", 0.05, 0.30, 0.10, 0.01, format="%.2f", help="Minimaler Cash-Anteil, der nicht unterschritten wird.", disabled=not ENABLE_DYNAMIC_CASH)
MAX_TOTAL_CASH_PCT = st.sidebar.slider("Max. Gesamt-Cash-Anteil (%)", 0.50, 0.90, 0.75, 0.01, format="%.2f", help="Maximaler Cash-Anteil, der nicht überschritten wird.", disabled=not ENABLE_DYNAMIC_CASH)

st.sidebar.markdown("---")
MIN_DATA_POINTS_FOR_ANALYSIS_GLOBAL = st.sidebar.slider("Min. Datenpunkte für TA (Basis, Tage)", 20, 90, 35, 1, help="Grundlegendes Minimum an Handelstagen für TA-Indikatoren. Optimierungsfenster können dies überschreiben.")
MC_SIMULATIONS = st.sidebar.slider("Monte Carlo Simulationen", 100, 10000, 500, 100, help="Mehr Simulationen = genauer, aber langsamer.")
MC_DAYS_TO_SIMULATE_LONG_TERM = 252
MC_DAYS_TO_SIMULATE_SHORT_TERM = [1, 7]

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Coin-Filterung")
if 'multiselect_coin_options' not in st.session_state:
    st.session_state.multiselect_coin_options = []
if 'coins_to_exclude_user_selection' not in st.session_state:
    st.session_state.coins_to_exclude_user_selection = []
EXCLUDE_COINS_MANUALLY = st.sidebar.multiselect(
    "Coins manuell von Analyse ausschließen:",
    options=st.session_state.multiselect_coin_options,
    default=st.session_state.coins_to_exclude_user_selection,
    help="Wähle Coins, die trotz Top-Ranking nicht berücksichtigt werden sollen. Analyse muss danach neu gestartet werden."
)
if EXCLUDE_COINS_MANUALLY != st.session_state.coins_to_exclude_user_selection:
    st.session_state.coins_to_exclude_user_selection = EXCLUDE_COINS_MANUALLY

STABLECOINS_TO_IGNORE_BASE = ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'USDP', 'GUSD', 'PAX', 'USDD', 'FDUSD', 'EURS', 'PYUSD']
PAIRS_TO_EXCLUDE_FROM_OPTIMIZATION_STATIC = ['PAXG/USDT', 'XAUT/USDT', 'WBTC/BTC', 'USDC/USDT']
LAST_DATA_DOWNLOAD_TIMESTAMP_UTC = None
exchange_instance = None

# --- HILFSFUNKTIONEN ---
def initialize_exchange():
    global exchange_instance
    if exchange_instance is None:
        try:
            exchange_instance = getattr(ccxt, TARGET_EXCHANGE)(); exchange_instance.load_markets(True)
        except Exception as e:
            st.sidebar.error(f"Fehler Börsenverbindung: {e}"); st.error(f"Börsenverbindung zu {TARGET_EXCHANGE} fehlgeschlagen: {e}"); st.stop()
    return exchange_instance

@st.cache_data(ttl=1800)
def get_fear_and_greed_index_data_cached(days=FNG_HISTORY_DAYS):
    current_value, current_classification = None, "Unbekannt (API Problem)"
    df_hist = pd.DataFrame()
    try:
        url_now = "https://api.alternative.me/fng/?limit=1&format=json"
        r_now = requests.get(url_now, timeout=10); r_now.raise_for_status(); data_now_list = r_now.json().get('data')
        if data_now_list and isinstance(data_now_list, list) and len(data_now_list) > 0:
            data_now = data_now_list[0]
            try:
                current_value_str = data_now.get('value')
                match = re.search(r"^\d+", str(current_value_str))
                if match: current_value = int(match.group(0))
                else: current_value = int(current_value_str)
                current_classification = data_now.get('value_classification', "Unbekannt")
            except (ValueError, TypeError): pass
        url_hist = f"https://api.alternative.me/fng/?limit={days}&date_format=unix"
        r_hist = requests.get(url_hist, timeout=10); r_hist.raise_for_status(); data_hist = r_hist.json()
        if data_hist and 'data' in data_hist and data_hist['data']:
            df_hist_raw = pd.DataFrame(data_hist['data'])
            if 'value' in df_hist_raw.columns and 'timestamp' in df_hist_raw.columns:
                df_hist_raw['value'] = pd.to_numeric(df_hist_raw['value'], errors='coerce')
                df_hist_raw['timestamp'] = pd.to_numeric(df_hist_raw['timestamp'], errors='coerce')
                df_hist_raw.dropna(subset=['value', 'timestamp'], inplace=True)
                if not df_hist_raw.empty:
                    df_hist = df_hist_raw.copy(); df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'].astype(int), unit='s')
                    df_hist = df_hist.sort_values('timestamp').set_index('timestamp')
        if current_value is None and not df_hist.empty and 'value' in df_hist.columns:
            current_value = int(df_hist['value'].iloc[-1])
            current_classification = "Aus Historie"
        elif current_value is None:
            current_value, current_classification = np.random.randint(40,60), "Neutral (Simuliert wg. API Problem)"
        return current_value, current_classification, df_hist
    except requests.exceptions.RequestException as e_req: print(f"F&G Index API Request Fehler: {e_req}"); st.sidebar.warning(f"F&G API Fehler (Netzwerk/Request). Nutze simulierte/hist. Werte.")
    except Exception as e_json: print(f"F&G Index API Gesamtfehler: {e_json}"); st.sidebar.warning(f"F&G API Fehler (Datenverarbeitung). Nutze simulierte/hist. Werte.")
    return np.random.randint(40,60), "Neutral (Simuliert wg. API Fehler)", pd.DataFrame()

def get_market_sentiment_data():
    fng_val, fng_cls, fng_hist_df = get_fear_and_greed_index_data_cached(days=FNG_HISTORY_DAYS)
    return {"fear_greed_value": fng_val, "fear_greed_classification": fng_cls, "fng_history_df": fng_hist_df}

@st.cache_data(ttl=86400)
def get_coingecko_coin_list_cached():
    try:
        url = "https://api.coingecko.com/api/v3/coins/list?include_platform=false"
        response = requests.get(url, timeout=10); response.raise_for_status()
        return {coin['symbol'].upper(): coin['id'] for coin in response.json()}
    except Exception as e: print(f"Fehler Laden CoinGecko Coin-Liste: {e}"); return {}

@st.cache_data(ttl=86400)
def get_coingecko_coin_details_cached(coin_id, max_retries=2, retry_delay_seconds=5):
    if not coin_id: return {"categories": ["Unbekannt"], "description": "Keine CoinGecko ID gefunden.", "homepage": None}
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&tickers=false&market_data=false&community_data=false&developer_data=false&sparkline=false"
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            categories = data.get('categories', [])
            filtered_categories = [cat for cat in categories if cat and isinstance(cat, str) and cat.lower() not in ["cryptocurrency", "token", "finance", "asset-backed-tokens", "stablecoins"]]
            final_categories = filtered_categories[:2] if filtered_categories else (categories[:1] if categories else ["Unbekannt"])
            description_html = data.get('description', {}).get('en', '')
            short_description = "Keine Beschreibung verfügbar."
            if description_html:
                soup = BeautifulSoup(description_html, 'html.parser')
                description_text = soup.get_text(separator=' ', strip=True)
                sentences = re.split(r'(?<=[.!?])\s+', description_text)
                short_desc_parts = []; current_len = 0; max_desc_len = 700
                for s_idx, s_val in enumerate(sentences):
                    if current_len + len(s_val) < max_desc_len or s_idx < 3 :
                        short_desc_parts.append(s_val); current_len += len(s_val) + 1
                        if s_idx >= 2 and current_len >= max_desc_len *0.60 : break
                    else: break
                short_description = " ".join(short_desc_parts)
                if short_desc_parts and not short_description.endswith(('.', '!', '?')): short_description += "."
                if not short_desc_parts and len(description_text) > max_desc_len : short_description = description_text[:max_desc_len] + "..."
                elif not short_desc_parts: short_description = description_text
            homepage_links = data.get('links', {}).get('homepage', [])
            homepage_link = homepage_links[0] if homepage_links and homepage_links[0] else None
            return {"categories": final_categories, "description": short_description, "homepage": homepage_link}
        except requests.exceptions.HTTPError as e_http:
            if e_http.response.status_code == 429:
                if attempt < max_retries:
                    actual_retry_delay = retry_delay_seconds * (2 ** attempt)
                    print(f"CoinGecko Rate Limit für {coin_id}. Warte {actual_retry_delay}s. Versuch {attempt + 2}/{max_retries +1}.")
                    time.sleep(actual_retry_delay)
                    continue
                else:
                    print(f"CoinGecko Rate Limit für {coin_id} auch nach {max_retries + 1} Versuchen. Gebe Fehler-Details zurück.")
                    return {"categories": ["API Limit"], "description": "Zu viele Anfragen an CoinGecko. Details nicht geladen.", "homepage": None}
            else:
                print(f"CoinGecko Detail HTTP Fehler {coin_id}: {e_http}")
                return {"categories": ["HTTP Fehler"], "description": f"Fehler beim Laden: {e_http.response.status_code}", "homepage": None}
        except Exception as e:
            print(f"CoinGecko Detail Allgemeiner Fehler {coin_id}: {e}")
            return {"categories": ["Fehler"], "description": f"Allgemeiner Fehler beim Laden der Coin-Details.", "homepage": None}
    return {"categories": ["Fehler"], "description": "Maximale Abrufversuche erreicht.", "homepage": None}


@st.cache_data(ttl=3600)
def get_top_n_coins_from_coingecko_cached(n=100, quote='USDT', ignore_stables_base=None, _exchange_markets_tuple=None):
    current_exchange = initialize_exchange()
    if ignore_stables_base is None: ignore_stables_base = STABLECOINS_TO_IGNORE_BASE
    req_n = min(n + len(ignore_stables_base) + 50, 250)
    url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page={req_n}&page=1&sparkline=false"
    symbols = []
    try:
        r = requests.get(url, timeout=15); r.raise_for_status(); coins = r.json()
    except Exception as e: print(f"CoinGecko API Fehler (Top Coins): {e}"); return []
    found_count = 0
    for coin_data in coins:
        if found_count >= n: break
        base = coin_data['symbol'].upper()
        if base in ignore_stables_base: continue
        pair = f"{base}/{quote}"
        market_info = current_exchange.markets.get(pair)
        if market_info and market_info.get('active', False):
            symbols.append(pair); found_count +=1
    return symbols

@st.cache_data(ttl=1800)
def fetch_crypto_data_cached(
    symbols,
    min_data_points_for_analysis_param: int,
    optimization_mu_span_days_param: int,
    optimization_cov_history_days_param: int,
    vola_penalty_atr_history_window_param: int,
    timeframe='1d',
    limit=365
    ):
    global LAST_DATA_DOWNLOAD_TIMESTAMP_UTC
    current_exchange = initialize_exchange()
    data = {}; count = 0
    download_start_time = datetime.now(timezone.utc)
    if not symbols: return {}

    min_required_for_ta = min_data_points_for_analysis_param
    min_required_for_opt_mu = optimization_mu_span_days_param + 10
    min_required_for_opt_cov = optimization_cov_history_days_param + 10
    min_required_for_vola_penalty = vola_penalty_atr_history_window_param + 5
    effective_limit = max(limit, min_required_for_ta, min_required_for_opt_mu, min_required_for_opt_cov, min_required_for_vola_penalty, 30)

    progress_bar = st.sidebar.progress(0); status_text = st.sidebar.empty()
    for i, sym in enumerate(symbols):
        count += 1
        status_text.text(f"Lade Daten: {sym} ({count}/{len(symbols)})")
        progress_bar.progress((i + 1) / len(symbols))
        try:
            if current_exchange.rateLimit: time.sleep(current_exchange.rateLimit / 1000)
            ohlcv = current_exchange.fetch_ohlcv(sym, timeframe=timeframe, limit=effective_limit)
            if ohlcv is None or len(ohlcv) < min_data_points_for_analysis_param: continue
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms'); df.set_index('timestamp', inplace=True)
            df = df[df['volume'] > 0]
            if len(df) < min_data_points_for_analysis_param: continue
            data[sym] = df
        except KeyboardInterrupt: raise
        except ccxt.NetworkError as e_net: print(f"Netzwerkfehler {sym}: {e_net}")
        except ccxt.ExchangeError as e_exc: print(f"Börsenfehler {sym}: {e_exc}")
        except Exception as e: print(f"-> Allgemeiner Fehler beim Laden von {sym}: {str(e)[:100]}...");
    status_text.text(f"Daten für {len(data)} von {len(symbols)} Assets geladen."); progress_bar.empty()
    if data: LAST_DATA_DOWNLOAD_TIMESTAMP_UTC = download_start_time
    return data

def perform_technical_analysis(data_dict, atr_period_global_param: int, atr_period_trade_param: int):
    long_candidates = []; short_candidates = []
    assets_with_ta_data = {}
    required_len = max(50, atr_period_global_param + 5, MIN_DATA_POINTS_FOR_ANALYSIS_GLOBAL)
    for symbol, df_orig in data_dict.items():
        if df_orig.empty or len(df_orig) < required_len:
            assets_with_ta_data[symbol] = df_orig; continue
        df = df_orig.copy()
        try:
            df['SMA20'] = ta.trend.SMAIndicator(df['close'], window=20, fillna=False).sma_indicator()
            df['SMA50'] = ta.trend.SMAIndicator(df['close'], window=50, fillna=False).sma_indicator()
            df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14, fillna=False).rsi()
            macd_indicator = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9, fillna=False)
            df['MACD_line'] = macd_indicator.macd()
            df['MACD_signal'] = macd_indicator.macd_signal()
            df['MACD_hist'] = macd_indicator.macd_diff()
            bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2, fillna=False)
            df['BB_low'] = bb_indicator.bollinger_lband();  df['BB_high'] = bb_indicator.bollinger_hband()
            df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=atr_period_global_param, fillna=False).average_true_range()
            df['ATR_TRADE'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=atr_period_trade_param, fillna=False).average_true_range()

        except Exception as e: print(f"TA Fehler bei Berechnung für {symbol}: {e}"); assets_with_ta_data[symbol] = df_orig; continue
        assets_with_ta_data[symbol] = df
        if df.empty or any(pd.isna(df.iloc[-1][col]) for col in ['RSI', 'MACD_hist', 'BB_low', 'BB_high', 'ATR_TRADE', 'SMA20', 'SMA50', 'close']):
            continue
        if df['ATR_TRADE'].iloc[-1] < 1e-8 : continue
        latest = df.iloc[-1]; previous = df.iloc[-2] if len(df) > 1 else latest
        if latest['RSI'] < 38 and previous['close'] < previous['BB_low'] and latest['close'] > latest['BB_low']:
            long_candidates.append({"symbol": symbol, "reason": "RSI<38,BB_Rev", "price": latest['close'], "rsi": latest['RSI'], "atr": latest['ATR_TRADE']})
        if previous['MACD_hist'] <= 0 and latest['MACD_hist'] > 0 and latest['RSI'] < 65 and latest['close'] > latest['SMA20']:
             long_candidates.append({"symbol": symbol, "reason": "MACD_H_Bull,RSI<65", "price": latest['close'], "rsi": latest['RSI'], "atr": latest['ATR_TRADE']})
        if latest['RSI'] > 75 and previous['close'] > previous['BB_high'] and latest['close'] < latest['BB_high']:
            short_candidates.append({"symbol": symbol, "reason": "RSI>75,BB_Rev_Short", "price": latest['close'], "rsi": latest['RSI'], "atr": latest['ATR_TRADE']})
        if previous['MACD_hist'] >= 0 and latest['MACD_hist'] < 0 and latest['RSI'] > 40 and latest['close'] < latest['SMA50'] and latest.get('MACD_line', 0) < 0 :
             short_candidates.append({"symbol": symbol, "reason": "MACD_H_Bear_Strong", "price": latest['close'], "rsi": latest['RSI'], "atr": latest['ATR_TRADE']})
    long_candidates = sorted(list({tc['symbol']: tc for tc in long_candidates}.values()), key=lambda x: x.get('rsi', 100))
    short_candidates = sorted(list({tc['symbol']: tc for tc in short_candidates}.values()), key=lambda x: x.get('rsi', 0), reverse=True)
    return {"long": long_candidates, "short": short_candidates}, assets_with_ta_data

def calculate_portfolio_metrics(portfolio_data_dict, allocation_dict, risk_free_rate):
    valid_assets_for_calc = [
        asset for asset, weight in allocation_dict.items()
        if asset in portfolio_data_dict and
           not portfolio_data_dict[asset].empty and
           weight > 0 and
           asset.split('/')[0].upper() not in STABLECOINS_TO_IGNORE_BASE
    ]
    if not valid_assets_for_calc: return None, None, None
    filtered_allocation = {asset: allocation_dict[asset] for asset in valid_assets_for_calc}
    total_weight_invested_assets = sum(filtered_allocation.values())
    if total_weight_invested_assets < 1e-6 : return None, None, None
    normalized_allocation = {asset: weight / total_weight_invested_assets for asset, weight in filtered_allocation.items()}
    common_index = None
    for asset in normalized_allocation.keys():
        if common_index is None: common_index = portfolio_data_dict[asset].index
        else: common_index = common_index.intersection(portfolio_data_dict[asset].index)
    if common_index is None or len(common_index) < 2: return None, None, None
    portfolio_returns_list = []
    for asset, weight in normalized_allocation.items():
        returns = portfolio_data_dict[asset].loc[common_index]['close'].pct_change().dropna()
        if not returns.empty:
            portfolio_returns_list.append(returns * weight)
    if not portfolio_returns_list: return None, None, None
    df_portfolio_contributions = pd.concat(portfolio_returns_list, axis=1)
    df_portfolio_total_returns = df_portfolio_contributions.sum(axis=1, min_count=1).ffill().fillna(0)
    if df_portfolio_total_returns.empty or len(df_portfolio_total_returns) < 2: return None, None, None
    mean_daily_return = df_portfolio_total_returns.mean(); std_daily_return = df_portfolio_total_returns.std()
    annual_return = mean_daily_return * 252; annual_volatility = std_daily_return * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 1e-6 else 0
    return annual_return, annual_volatility, sharpe_ratio

def format_price(price):
    if pd.isna(price) or not isinstance(price, (int, float)) or price == 0: return "N/A"
    if abs(price) < 0.000001 and price != 0: return f"{price:.8e}"
    if abs(price) < 0.001: return f"{price:.8f}"
    elif abs(price) < 0.1: return f"{price:.6f}"
    elif abs(price) < 1: return f"{price:.4f}"
    else: return f"{price:,.2f}"

def optimize_portfolio_sharpe(
    assets_ta_data_dict,
    risk_free_rate_param,
    assets_for_optimization_list,
    max_assets_limit_param,
    coingecko_ids_map_param,
    coingecko_categories_cache_param,
    current_tickers_dict_param
    ):
    mu_span_days = OPTIMIZATION_MU_SPAN_DAYS
    cov_history_days = OPTIMIZATION_COV_HISTORY_DAYS
    min_hist_for_opt = max(MIN_DATA_POINTS_FOR_ANALYSIS_GLOBAL, VOLA_PENALTY_ATR_HISTORY_WINDOW + 10, mu_span_days + 10, cov_history_days + 10, 30)

    prices_dict_for_opt = {}
    valid_symbols_after_filter = []
    for symbol in assets_for_optimization_list:
        if symbol in assets_ta_data_dict:
            df_asset = assets_ta_data_dict[symbol]
            if not df_asset.empty and \
               len(df_asset) >= min_hist_for_opt and \
               all(col in df_asset.columns for col in ['close', 'ATR']) and \
               not df_asset[['close', 'ATR']].isnull().values.any():
                prices_dict_for_opt[symbol] = df_asset['close']
                valid_symbols_after_filter.append(symbol)
    if not prices_dict_for_opt or len(valid_symbols_after_filter) < 2:
        st.sidebar.warning(f"Zu wenig valide Assets ({len(valid_symbols_after_filter)}) für Optimierung nach Filterung. Benötigt: mind. {min_hist_for_opt} Tage & gültige TA-Daten für mind. 2 Assets.");
        return None, None, None, pd.DataFrame()

    prices_df_for_opt_full_history = pd.DataFrame(prices_dict_for_opt).ffill().bfill()
    prices_df_for_opt_full_history.dropna(axis=1, how='any', inplace=True)
    valid_symbols_after_filter = list(prices_df_for_opt_full_history.columns)

    if prices_df_for_opt_full_history.empty or len(valid_symbols_after_filter) < 2:
        st.sidebar.warning("Zu wenig valide Assets nach Preis-DataFrame-Bereinigung für Optimierung.");
        return None, None, prices_df_for_opt_full_history, pd.DataFrame()

    if cov_history_days > 0 and len(prices_df_for_opt_full_history) >= cov_history_days:
        prices_df_for_cov_calc = prices_df_for_opt_full_history.iloc[-cov_history_days:]
    else:
        prices_df_for_cov_calc = prices_df_for_opt_full_history

    try:
        mu_original = expected_returns.ema_historical_return(prices_df_for_opt_full_history, compounding=False, span=mu_span_days, frequency=252)
        mu_original = mu_original.clip(lower=-MAX_EXPECTED_ANNUAL_RETURN_CAP, upper=MAX_EXPECTED_ANNUAL_RETURN_CAP)
        S_matrix = risk_models.CovarianceShrinkage(prices_df_for_cov_calc, frequency=252).ledoit_wolf()
    except Exception as e:
        st.sidebar.error(f"Fehler bei Berechnung von mu/S: {e}");
        return None, None, prices_df_for_opt_full_history, pd.DataFrame()

    mu_adjusted_for_penalty = mu_original.copy()
    penalty_applied_log = {}
    if ENABLE_VOLA_PENALTY:
        all_normalized_atrs = []
        asset_norm_atr_details = {}
        for symbol_iter in valid_symbols_after_filter:
            if symbol_iter not in assets_ta_data_dict: continue
            df_asset_iter = assets_ta_data_dict[symbol_iter].dropna(subset=['ATR', 'close'])
            if not df_asset_iter.empty and len(df_asset_iter) > VOLA_PENALTY_ATR_HISTORY_WINDOW:
                current_atr_val = df_asset_iter['ATR'].iloc[-1]
                current_close_val = df_asset_iter['close'].iloc[-1]
                if current_close_val > 1e-8 and current_atr_val > 1e-8:
                    norm_atr_current = current_atr_val / current_close_val
                    all_normalized_atrs.append(norm_atr_current)
                    asset_norm_atr_details[symbol_iter] = {'current': norm_atr_current}
                    df_asset_iter['norm_atr_hist'] = (df_asset_iter['ATR'] / df_asset_iter['close'].replace(0, np.nan)).ffill().fillna(0)
                    rolling_norm_atr_series = df_asset_iter['norm_atr_hist'].rolling(window=VOLA_PENALTY_ATR_HISTORY_WINDOW).mean()
                    if not pd.isna(rolling_norm_atr_series.iloc[-1]) and rolling_norm_atr_series.iloc[-1] > 1e-8:
                       asset_norm_atr_details[symbol_iter]['rolling_avg'] = rolling_norm_atr_series.iloc[-1]
        if all_normalized_atrs:
            median_norm_atr_all = np.median([a for a in all_normalized_atrs if not pd.isna(a) and a > 0])
            if not pd.isna(median_norm_atr_all) and median_norm_atr_all > 1e-8:
                for symbol_iter in mu_adjusted_for_penalty.index:
                    current_penalty = 0.0
                    penalty_reasons = []
                    asset_data_iter = asset_norm_atr_details.get(symbol_iter)
                    if asset_data_iter:
                        current_norm_atr_val = asset_data_iter['current']
                        if current_norm_atr_val > median_norm_atr_all * VOLA_PENALTY_MEDIAN_THRESHOLD:
                            penalty_vs_median = min(((current_norm_atr_val / median_norm_atr_all) - 1) * VOLA_PENALTY_FACTOR_MEDIAN, VOLA_PENALTY_MAX_PENALTY_MEDIAN)
                            current_penalty = max(current_penalty, penalty_vs_median)
                            if penalty_vs_median > 0: penalty_reasons.append(f"Median ({penalty_vs_median:.1%})")
                        rolling_avg_norm_atr_val = asset_data_iter.get('rolling_avg')
                        if rolling_avg_norm_atr_val and current_norm_atr_val > rolling_avg_norm_atr_val * VOLA_PENALTY_SELF_THRESHOLD:
                             penalty_vs_self = min(((current_norm_atr_val / rolling_avg_norm_atr_val) - 1) * VOLA_PENALTY_FACTOR_SELF, VOLA_PENALTY_MAX_PENALTY_SELF)
                             current_penalty = max(current_penalty, penalty_vs_self)
                             if penalty_vs_self > 0: penalty_reasons.append(f"Self ({penalty_vs_self:.1%})")
                        if current_penalty > 0:
                            mu_adjusted_for_penalty[symbol_iter] *= (1 - current_penalty)
                            penalty_applied_log[symbol_iter] = f"-{current_penalty:.1%} ({', '.join(penalty_reasons)})"
            else: st.sidebar.caption("Info: Median der norm. ATR für Penalty nicht berechenbar.")
        else: st.sidebar.caption("Info: Keine norm. ATR-Werte für Penalty-Berechnung.")
    if penalty_applied_log:
        with st.sidebar.expander("ℹ️ Info: Angewandte Vola-Penalties", expanded=False):
            for sym_log, info_log in penalty_applied_log.items():
                st.caption(f"{sym_log}: {info_log}")

    final_weights_dict = None; optimized_performance_data = None
    ef_instance = None
    try:
        common_symbols_for_ef = mu_adjusted_for_penalty.index.intersection(S_matrix.index)
        mu_for_ef = mu_adjusted_for_penalty.loc[common_symbols_for_ef]
        S_for_ef = S_matrix.loc[common_symbols_for_ef, common_symbols_for_ef]

        if len(common_symbols_for_ef) < 2:
             st.sidebar.error("Nicht genug Assets (nach Penalty/Index-Abgleich) für Optimierung.")
             return None, None, prices_df_for_opt_full_history, pd.DataFrame()

        ef_instance = EfficientFrontier(mu_for_ef, S_for_ef, weight_bounds=(0.0, 1.0))
        try:
            raw_weights = ef_instance.max_sharpe(risk_free_rate=risk_free_rate_param)
        except ValueError as e_val_sharpe:
             st.sidebar.warning(f"Max Sharpe Optimierung nicht möglich: {e_val_sharpe}. Fallback auf Min Volatility.")
             try:
                 ef_instance_fallback = EfficientFrontier(mu_for_ef, S_for_ef, weight_bounds=(0.0, 1.0))
                 raw_weights = ef_instance_fallback.min_volatility()
                 ef_instance = ef_instance_fallback
             except Exception as e_min_vol:
                 st.sidebar.error(f"Fallback Min Volatility auch fehlgeschlagen: {e_min_vol}")
                 return None, None, prices_df_for_opt_full_history, pd.DataFrame()
        except Exception as e_max_s_other:
            st.sidebar.error(f"Unerwarteter Fehler bei max_sharpe: {e_max_s_other}")
            return None, None, prices_df_for_opt_full_history, pd.DataFrame()

        if raw_weights is None:
            st.sidebar.warning("Keine Gewichte nach Optimierung erhalten."); return None, None, prices_df_for_opt_full_history, pd.DataFrame()
        cleaned_weights = ef_instance.clean_weights(cutoff=0.0001)
        if not cleaned_weights:
            st.sidebar.warning("Keine Assets nach clean_weights()."); return None, None, prices_df_for_opt_full_history, pd.DataFrame()

        if max_assets_limit_param > 0 and len(cleaned_weights) > max_assets_limit_param:
            sorted_cleaned_weights = dict(sorted(cleaned_weights.items(), key=lambda item: item[1], reverse=True))
            top_n_unnormalized = dict(list(sorted_cleaned_weights.items())[:max_assets_limit_param])
            total_top_n_weight = sum(top_n_unnormalized.values())
            if total_top_n_weight > 1e-6:
                final_weights_dict = {k: v / total_top_n_weight for k, v in top_n_unnormalized.items()}
        else:
            total_cleaned_weight = sum(cleaned_weights.values())
            if total_cleaned_weight > 1e-6:
                final_weights_dict = {k: v / total_cleaned_weight for k, v in cleaned_weights.items()}
            else: final_weights_dict = cleaned_weights
        if not final_weights_dict:
            st.sidebar.warning("Keine finalen Gewichte nach Begrenzung/Normalisierung."); return None, None, prices_df_for_opt_full_history, pd.DataFrame()

        weights_series = pd.Series(final_weights_dict)
        common_idx_perf = weights_series.index.intersection(mu_for_ef.index)
        weights_series_aligned = weights_series.loc[common_idx_perf]
        mu_for_ef_aligned = mu_for_ef.loc[common_idx_perf]
        S_for_ef_aligned = S_for_ef.loc[common_idx_perf, common_idx_perf]

        ret_opt = np.sum(mu_for_ef_aligned * weights_series_aligned)
        vol_opt_squared = np.dot(weights_series_aligned.T, np.dot(S_for_ef_aligned, weights_series_aligned))
        vol_opt = np.sqrt(vol_opt_squared) if vol_opt_squared >= 0 else 0
        sharpe_opt = (ret_opt - risk_free_rate_param) / vol_opt if vol_opt > 1e-6 else 0
        optimized_performance_data = {
            "annual_return": ret_opt,
            "annual_volatility": vol_opt,
            "sharpe_ratio": sharpe_opt,
            "daily_return": ret_opt / 252
        }
    except Exception as e_opt:
        st.sidebar.error(f"Fehler während Portfolio-Optimierung: {e_opt}");
        traceback.print_exc()
        return None, None, prices_df_for_opt_full_history, pd.DataFrame()

    asset_details_list_for_df = []
    if final_weights_dict and optimized_performance_data:
        for asset_symbol, weight_val in sorted(final_weights_dict.items(), key=lambda x: x[1], reverse=True):
            if weight_val > 0.0001:
                asset_base_symbol = asset_symbol.split('/')[0]
                coingecko_id = coingecko_ids_map_param.get(asset_base_symbol)
                coin_details = {"categories":["Unbekannt"],"description":"N/A", "homepage": None}
                if coingecko_id:
                    if coingecko_id not in coingecko_categories_cache_param:
                        coingecko_categories_cache_param[coingecko_id] = get_coingecko_coin_details_cached(coingecko_id)
                    coin_details = coingecko_categories_cache_param.get(coingecko_id, coin_details)
                asset_current_price = current_tickers_dict_param.get(asset_symbol, {}).get('last', 0)
                if asset_current_price is None or asset_current_price <= 0 :
                   if asset_symbol in prices_df_for_opt_full_history.columns:
                       asset_current_price = prices_df_for_opt_full_history[asset_symbol].iloc[-1]
                       if pd.isna(asset_current_price): asset_current_price = 0
                rsi_display = "N/A"; perf_30d_display = "N/A"; ta_signal_display = "Neutral"
                if asset_symbol in assets_ta_data_dict and not assets_ta_data_dict[asset_symbol].empty:
                    df_asset_for_details = assets_ta_data_dict[asset_symbol]
                    if 'RSI' in df_asset_for_details.columns and not pd.isna(df_asset_for_details['RSI'].iloc[-1]):
                        rsi_num = df_asset_for_details['RSI'].iloc[-1]
                        rsi_display = f"{rsi_num:.1f}"
                        if rsi_num > 75: rsi_display += " (🔥 SEHR Überkauft!)"
                        elif rsi_num > 70: rsi_display += " (Überkauft)"
                        elif rsi_num < 25: rsi_display += " (🧊 SEHR Überverkauft!)"
                        elif rsi_num < 30: rsi_display += " (Überverkauft)"
                    if 'close' in df_asset_for_details.columns and len(df_asset_for_details['close']) >= 31 and asset_current_price > 0:
                        price_30d_ago_val = df_asset_for_details['close'].iloc[-31]
                        if price_30d_ago_val > 0:
                            perf_30d_num = ((asset_current_price - price_30d_ago_val) / price_30d_ago_val)
                            perf_30d_display = f"{perf_30d_num:.1%}"
                    latest_ta_row = df_asset_for_details.iloc[-1]
                    if not latest_ta_row.isnull().all():
                        if latest_ta_row.get('RSI', 50) < 40 and latest_ta_row.get('MACD_hist', 0) > 0 and latest_ta_row.get('close',0) > latest_ta_row.get('SMA20', float('inf')):
                            ta_signal_display = "Leicht Bullisch"
                        elif latest_ta_row.get('RSI', 50) > 60 and latest_ta_row.get('MACD_hist', 0) < 0 and latest_ta_row.get('close',0) < latest_ta_row.get('SMA20', 0):
                            ta_signal_display = "Leicht Bärisch"
                expected_return_display = f"{mu_original.get(asset_symbol, 0)*100:.2f}%"
                if asset_symbol in penalty_applied_log:
                     expected_return_display += f" (nach Penalty {penalty_applied_log[asset_symbol]})"
                asset_details_list_for_df.append({
                    "Asset": asset_symbol, "Gewichtung": f"{weight_val:.2%}",
                    "Akt. Preis": f"${format_price(asset_current_price)}",
                    "Akt. RSI(14)": rsi_display, "Perf. 30T": perf_30d_display,
                    "TA Signal (Portfolio)": ta_signal_display,
                    "Erw. Rend. (p.a.)": expected_return_display,
                    "Asset Vola (p.a.)": f"{np.sqrt(S_matrix.loc[asset_symbol,asset_symbol])*100:.2f}%" if asset_symbol in S_matrix.index else "N/A",
                    "Sektor(en)": ", ".join(coin_details["categories"]),
                    "Homepage": coin_details["homepage"],
                    "Beschreibung": coin_details["description"]
                })
    return final_weights_dict, optimized_performance_data, prices_df_for_opt_full_history, pd.DataFrame(asset_details_list_for_df)


def generate_trade_recommendations_from_portfolio(
    optimized_weights: dict,
    portfolio_value_usd: float,
    assets_ta_data: dict,
    current_tickers: dict
    ):
    trading_setups = []
    if not optimized_weights:
        return pd.DataFrame()

    investable_capital = portfolio_value_usd * (1 - st.session_state.get('target_cash_allocation_final_pct', 0.30))

    for asset, weight in optimized_weights.items():
        if weight <= 0.0001:
            continue

        allocated_capital_for_asset = investable_capital * weight
        if allocated_capital_for_asset < MIN_USD_ALLOCATION_PER_ASSET:
            # print(f"Info Trading Setup: {asset} wird übersprungen, da die allokierte Summe (${allocated_capital_for_asset:,.2f}) unter dem Minimum von ${MIN_USD_ALLOCATION_PER_ASSET:,.2f} liegt.")
            continue

        df_asset = assets_ta_data.get(asset)
        if df_asset is None or df_asset.empty or 'ATR_TRADE' not in df_asset.columns or pd.isna(df_asset['ATR_TRADE'].iloc[-1]):
            print(f"Warnung: Keine gültigen TA-Daten (insb. ATR_TRADE) für {asset} im Trading Setup.")
            continue

        entry_price = current_tickers.get(asset, {}).get('last')
        if entry_price is None or entry_price <= 0:
            if df_asset is not None and not df_asset.empty:
                entry_price = df_asset['close'].iloc[-1]
            else:
                print(f"Warnung: Kein Entry-Preis für {asset} in Trading Setups, obwohl Allokation > Minimum.")
                continue
        if entry_price is None or entry_price <= 0:
            print(f"Warnung: Kein gültiger Entry-Preis für {asset} nach Fallback.")
            continue
        
        risk_amount_usd_for_this_trade = portfolio_value_usd * RISK_PER_TRADE_PORTFOLIO_PCT

        stop_loss_price = 0
        sl_type = "N/A"
        atr_trade_value = df_asset['ATR_TRADE'].iloc[-1]

        if USE_ATR_FOR_SL_TP and atr_trade_value > 1e-8:
            stop_loss_price = entry_price - (ATR_SL_MULTIPLIER_TRADE * atr_trade_value)
            sl_type = f"ATR ({format_price(atr_trade_value)})"
        else:
            stop_loss_price = entry_price * (1 - DEFAULT_SL_PCT_TRADE)
            sl_type = f"Std % ({DEFAULT_SL_PCT_TRADE:.1%})"

        if stop_loss_price <= 0 or stop_loss_price >= entry_price:
            stop_loss_price = entry_price * (1 - max(DEFAULT_SL_PCT_TRADE, 0.01))
            sl_type = f"Std % (Fallback, {max(DEFAULT_SL_PCT_TRADE, 0.01):.1%})"
            if stop_loss_price <= 0 or stop_loss_price >= entry_price:
                 print(f"Kritische Warnung: Ungültiger SL für {asset} auch nach aggressivem Fallback. Überspringe Trade-Setup.")
                 continue

        risk_per_unit = entry_price - stop_loss_price
        if risk_per_unit <= 1e-8:
            print(f"Warnung: Risiko pro Einheit zu klein oder negativ für {asset} (Entry: {entry_price}, SL: {stop_loss_price}). Überspringe Trade-Setup.")
            continue

        position_size_units = risk_amount_usd_for_this_trade / risk_per_unit
        position_size_usd_calculated = position_size_units * entry_price
        position_size_usd_final = min(position_size_usd_calculated, allocated_capital_for_asset)
        
        MIN_TRADE_SIZE_USD = 10.0
        if position_size_usd_final < MIN_TRADE_SIZE_USD:
            # print(f"Info Trading Setup: Berechnete Positionsgröße für {asset} (${position_size_usd_final:,.2f}) liegt unter dem Minimum von ${MIN_TRADE_SIZE_USD:,.2f}. Wird nicht als Trade angezeigt.")
            continue

        position_size_units_final = position_size_usd_final / entry_price if entry_price > 0 else 0

        tp1_price = entry_price + (risk_per_unit * TAKE_PROFIT_RR_RATIO_1)
        tp2_price = entry_price + (risk_per_unit * TAKE_PROFIT_RR_RATIO_2)

        trading_setups.append({
            "Asset": asset,
            "Richtung": "Long",
            "Gewichtung im Portfolio": f"{weight:.2%}",
            "Entry Preis": f"${format_price(entry_price)}",
            "Stop-Loss": f"${format_price(stop_loss_price)} ({sl_type})",
            "Take-Profit 1": f"${format_price(tp1_price)} (RR: {TAKE_PROFIT_RR_RATIO_1:.1f})",
            "Take-Profit 2": f"${format_price(tp2_price)} (RR: {TAKE_PROFIT_RR_RATIO_2:.1f})",
            "Positionsgröße (USD)": f"${position_size_usd_final:,.2f}",
            "Positionsgröße (Einheiten)": f"{position_size_units_final:.4f}",
            "Kalk. Risiko für diesen Trade (USD)": f"${risk_amount_usd_for_this_trade:,.2f}",
            "TV Chart": f"https://www.tradingview.com/chart/?symbol={TARGET_EXCHANGE.upper()}:{asset.replace('/', '')}"
        })
    return pd.DataFrame(trading_setups)


def analyze_weekday_performance(assets_ta_data_dict, assets_to_analyze_list):
    weekday_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]; analysis_results = {}
    for symbol_analyze in assets_to_analyze_list:
        if symbol_analyze not in assets_ta_data_dict or assets_ta_data_dict[symbol_analyze].empty or len(assets_ta_data_dict[symbol_analyze]) < 20: continue
        df_analyze = assets_ta_data_dict[symbol_analyze].copy()
        df_analyze['daily_return'] = df_analyze['close'].pct_change()
        df_analyze['weekday'] = df_analyze.index.weekday
        try:
            weekday_stats = df_analyze.groupby('weekday')['daily_return'].agg(['mean', 'count', 'std']).reindex(range(7))
            weekday_stats['weekday_name'] = weekday_names
            weekday_stats.set_index('weekday_name', inplace=True)
            analysis_results[symbol_analyze] = weekday_stats
        except Exception: pass
    if not analysis_results: return None
    output_df_data_wd = {sym_res: stats_df_res['mean'].apply(lambda x: x*100 if pd.notna(x) else np.nan) for sym_res, stats_df_res in analysis_results.items()}
    if output_df_data_wd: return pd.DataFrame(output_df_data_wd)
    return None

def run_monte_carlo_simulation(
    portfolio_expected_annual_return, portfolio_annual_volatility,
    num_simulations, num_days_to_simulate, initial_portfolio_value=100, title_suffix=""
    ):
    if portfolio_expected_annual_return is None or portfolio_annual_volatility is None or portfolio_annual_volatility <= 1e-6 :
        st.warning(f"Ungültige Eingaben für Monte-Carlo ({title_suffix}). Simulation übersprungen."); return None, None
    daily_return_avg = portfolio_expected_annual_return / 252
    daily_volatility = portfolio_annual_volatility / np.sqrt(252)
    if daily_volatility <= 1e-8:
        st.warning(f"Tägliche Volatilität für Monte-Carlo ({title_suffix}) zu gering. Simulation übersprungen."); return None, None
    simulation_results_array = np.zeros((num_days_to_simulate + 1, num_simulations))
    simulation_results_array[0, :] = initial_portfolio_value
    for i in range(num_simulations):
        drift = daily_return_avg - (daily_volatility**2 / 2)
        random_shocks = np.random.normal(0, 1, num_days_to_simulate)
        daily_log_returns = drift + daily_volatility * random_shocks
        cumulative_log_returns = np.cumsum(daily_log_returns)
        simulation_results_array[1:, i] = initial_portfolio_value * np.exp(cumulative_log_returns)
    final_portfolio_values = simulation_results_array[-1, :]
    fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
    sns.histplot(final_portfolio_values, kde=True, bins=50, ax=ax_mc, color="skyblue")
    ax_mc.set_title(f'Verteilung Portfolio-Endwerte nach {num_days_to_simulate} Tagen ({title_suffix})')
    ax_mc.set_xlabel(f'Portfolio Endwert (Start={initial_portfolio_value})'); ax_mc.set_ylabel('Häufigkeit')
    mean_final_value = np.mean(final_portfolio_values); median_final_value = np.median(final_portfolio_values)
    percentile_5 = np.percentile(final_portfolio_values, 5); percentile_95 = np.percentile(final_portfolio_values, 95)
    ax_mc.axvline(mean_final_value, color='r', ls='--', lw=1.5, label=f'Mean: {format_price(mean_final_value)}')
    ax_mc.axvline(median_final_value, color='g', ls='--', lw=1.5, label=f'Median: {format_price(median_final_value)}')
    ax_mc.axvline(percentile_5, color='darkblue', ls=':', lw=1.2, label=f'5% Pctl: {format_price(percentile_5)}')
    ax_mc.axvline(percentile_95, color='darkblue', ls=':', lw=1.2, label=f'95% Pctl: {format_price(percentile_95)}')
    ax_mc.legend(); ax_mc.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    mc_summary_stats = {"mean": mean_final_value, "median": median_final_value, "p5": percentile_5, "p95": percentile_95, "days": num_days_to_simulate}
    return fig_mc, mc_summary_stats

def get_correlation_matrix(assets_ta_data_dict_param, assets_list_for_corr):
    valid_assets_for_corr = [
        a for a in assets_list_for_corr
        if a in assets_ta_data_dict_param and
           not assets_ta_data_dict_param[a].empty and
           'close' in assets_ta_data_dict_param[a].columns and
           a.split('/')[0].upper() not in STABLECOINS_TO_IGNORE_BASE and
           a.upper() not in PAIRS_TO_EXCLUDE_FROM_OPTIMIZATION_STATIC
    ]
    if len(valid_assets_for_corr) < 2: return None
    common_time_index = None
    closes_data_dict = {}
    for asset_corr in valid_assets_for_corr:
        df_asset_corr = assets_ta_data_dict_param[asset_corr]
        closes_data_dict[asset_corr] = df_asset_corr['close']
        if common_time_index is None: common_time_index = df_asset_corr.index
        else: common_time_index = common_time_index.intersection(df_asset_corr.index)
    if common_time_index is None or len(common_time_index) < 2 or not closes_data_dict: return None
    aligned_closes_df_list = []
    for asset_corr_key in valid_assets_for_corr:
        if asset_corr_key in closes_data_dict:
            aligned_closes_df_list.append(closes_data_dict[asset_corr_key].reindex(common_time_index).rename(asset_corr_key))
    if not aligned_closes_df_list or len(aligned_closes_df_list) < 2: return None
    all_closes_combined_df = pd.concat(aligned_closes_df_list, axis=1).ffill().bfill()
    all_closes_combined_df.dropna(axis=1, how='all', inplace=True)
    all_closes_combined_df.dropna(how='all', inplace=True)
    if all_closes_combined_df.shape[1] < 2: return None
    return all_closes_combined_df.pct_change().corr()

def plot_portfolio_allocation_st(allocation_dict_param, plot_title="Portfolio Allokation", max_items_to_plot=15):
    if not allocation_dict_param: st.warning(f"Keine Daten für Plot: {plot_title}"); return None
    plot_data_dict = allocation_dict_param.copy()
    if "optimiert" in plot_title.lower():
        plot_data_dict = {
            k:v for k,v in plot_data_dict.items()
            if not (k.upper() in STABLECOINS_TO_IGNORE_BASE and v < 0.0001)
        }
    labels_list = list(plot_data_dict.keys()); sizes_list = list(plot_data_dict.values())
    if not labels_list or not sizes_list or sum(sizes_list) < 1e-5:
        st.warning(f"Keine plotbaren Daten für '{plot_title}' (nach Filter)."); return None
    sorted_allocations = sorted(zip(labels_list, sizes_list), key=lambda x: x[1], reverse=True)
    plot_labels_final = [item[0] for item in sorted_allocations[:max_items_to_plot]]
    plot_sizes_final = [item[1] for item in sorted_allocations[:max_items_to_plot]]
    if len(sorted_allocations) > max_items_to_plot:
        plot_labels_final.append('Andere (<0.01% oder >Max)')
        plot_sizes_final.append(sum(item[1] for item in sorted_allocations[max_items_to_plot:]))
    if not plot_sizes_final or sum(plot_sizes_final) < 1e-5:
        st.warning(f"Keine plotbaren Größen für '{plot_title}' (nach Konsolidierung)."); return None
    fig_pie, ax_pie = plt.subplots(figsize=(8,4.5))
    ax_pie.pie(plot_sizes_final, labels=plot_labels_final, autopct='%1.1f%%', startangle=90, pctdistance=0.85,
               wedgeprops={'edgecolor': 'white', 'linewidth': 0.5})
    ax_pie.axis('equal'); ax_pie.set_title(plot_title, fontsize=14);
    plt.tight_layout()
    return fig_pie

def plot_correlation_matrix_heatmap_st(correlation_matrix_param, heatmap_title="Korrelationsmatrix"):
    if correlation_matrix_param is None or correlation_matrix_param.empty:
        st.info(f"Keine Daten für Korrelations-Plot: {heatmap_title}"); return None
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(max(8, len(correlation_matrix_param.columns)*0.8), max(6, len(correlation_matrix_param.columns)*0.6)))
    sns.heatmap(correlation_matrix_param, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
                annot_kws={"size":8 if len(correlation_matrix_param.columns) < 10 else 6},
                ax=ax_heatmap, vmin=-1, vmax=1)
    ax_heatmap.set_title(heatmap_title, fontsize=14);
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    return fig_heatmap

def plot_asset_chart_st(df_asset_to_plot, symbol_to_plot, indicators_to_plot=None, chart_title_prefix="Preisverlauf"):
    if df_asset_to_plot is None or df_asset_to_plot.empty or 'close' not in df_asset_to_plot.columns or len(df_asset_to_plot) < 5:
        st.caption(f"Nicht genügend Daten für Chart: {symbol_to_plot}"); return None
    fig_asset, ax_asset = plt.subplots(figsize=(10, 4.5))
    ax_asset.plot(df_asset_to_plot.index, df_asset_to_plot['close'], label='Schlusskurs', color='dodgerblue', lw=1.5, alpha=0.9)
    if indicators_to_plot:
        if 'SMA20' in indicators_to_plot and 'SMA20' in df_asset_to_plot.columns and not df_asset_to_plot['SMA20'].isnull().all():
            ax_asset.plot(df_asset_to_plot.index, df_asset_to_plot['SMA20'], label='SMA20', color='darkorange', ls='--', lw=1.2, alpha=0.8)
        if 'SMA50' in indicators_to_plot and 'SMA50' in df_asset_to_plot.columns and not df_asset_to_plot['SMA50'].isnull().all():
            ax_asset.plot(df_asset_to_plot.index, df_asset_to_plot['SMA50'], label='SMA50', color='green', ls=':', lw=1.2, alpha=0.8)
        if 'BB' in indicators_to_plot and all(c in df_asset_to_plot.columns for c in ['BB_high', 'BB_low']) \
           and not df_asset_to_plot['BB_high'].isnull().all() and not df_asset_to_plot['BB_low'].isnull().all():
            ax_asset.fill_between(df_asset_to_plot.index, df_asset_to_plot['BB_low'], df_asset_to_plot['BB_high'], color='silver', alpha=0.3, label='Bollinger Bänder')
    ax_asset.set_title(f"{chart_title_prefix}: {symbol_to_plot}", fontsize=14)
    ax_asset.set_ylabel(f"Preis ({symbol_to_plot.split('/')[1] if '/' in symbol_to_plot else 'USD'})", fontsize=10)
    ax_asset.legend(fontsize='small'); ax_asset.grid(True, linestyle=':', alpha=0.6); fig_asset.autofmt_xdate()
    ax_asset.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%y')); plt.xticks(rotation=30, ha='right', fontsize=9);
    plt.yticks(fontsize=9)
    plt.tight_layout()
    return fig_asset

def print_footer_st():
    st.markdown("---")
    st.caption("HINWEIS: Diese Analyse ist experimentell und dient nur zu Informationszwecken. KEINE ANLAGEBERATUNG. Kryptowährungen sind hochvolatil und risikoreich. Das Skript führt KEINE ECHTEN TRADES AUS. Alle Entscheidungen liegen in Ihrer Verantwortung.")

# --- HAUPTABLAUF des Streamlit Dashboards ---
def run_analysis():
    global LAST_DATA_DOWNLOAD_TIMESTAMP_UTC, exchange_instance
    st.session_state.setdefault('analysis_in_progress', False)
    st.session_state.setdefault('results_ready', False)
    st.session_state.setdefault('last_opt_weights', {})
    st.session_state.setdefault('last_target_cash_pct', CURRENT_PORTFOLIO_ALLOCATION.get(next((cs for cs in ['USDC','USDT'] if cs in CURRENT_PORTFOLIO_ALLOCATION), 'USDC'), 0.30))
    if 'exchange_initialized_for_session' not in st.session_state:
        initialize_exchange(); st.session_state.exchange_initialized_for_session = True
    if 'coingecko_ids_map' not in st.session_state:
        with st.spinner("Lade CoinGecko Coin-Liste..."): st.session_state.coingecko_ids_map = get_coingecko_coin_list_cached()
    if 'coingecko_categories_cache' not in st.session_state:
        st.session_state.coingecko_categories_cache = {}

    if not st.session_state.multiselect_coin_options:
        temp_quote_currency = 'USDT'
        temp_markets_tuple = tuple(sorted(exchange_instance.markets.keys())) if exchange_instance and exchange_instance.markets else tuple()
        initial_universe_for_multiselect = get_top_n_coins_from_coingecko_cached(
            n=NUMBER_OF_TOP_COINS_TO_SCAN + 50,
            quote=temp_quote_currency,
            ignore_stables_base=STABLECOINS_TO_IGNORE_BASE,
            _exchange_markets_tuple=temp_markets_tuple
        )
        if initial_universe_for_multiselect:
            st.session_state.multiselect_coin_options = sorted(list(set(initial_universe_for_multiselect)))

    if st.sidebar.button("🚀 Analyse Starten / Aktualisieren", type="primary", use_container_width=True):
        st.session_state['analysis_in_progress'] = True; st.session_state['results_ready'] = False
        keys_to_clear_on_rerun = [
            'fig_fng', 'mc_figs_dict', 'fig_alloc', 'fig_corr', 'weekday_df', 'corr_matrix',
            'opt_asset_details_df', 'trading_setups_df', 'opt_weights_current_run',
            'opt_perf_data_current_run', 'rebalancing_advice_df', 'sentiment_data',
            'trading_universe_filtered', 'hist_data_ohlcv', 'top_candidates_dict', 'assets_with_ta_data_final',
            'current_portfolio_sharpe', 'prices_df_for_discrete_alloc', 'optimized_assets_list_final',
            'target_cash_allocation_final_pct', 'mc_all_stats_dict', 'current_exchange_tickers'
        ]
        for key_clear in keys_to_clear_on_rerun:
            if key_clear in st.session_state: del st.session_state[key_clear]

        with st.spinner("Führe vollständige Analyse durch..."):
            st.session_state.sentiment_data = get_market_sentiment_data()
            quote_currency_used = 'USDT'
            markets_tuple_for_cache_key = tuple(sorted(exchange_instance.markets.keys())) if exchange_instance and exchange_instance.markets else tuple()
            base_trading_universe = get_top_n_coins_from_coingecko_cached(
                n=NUMBER_OF_TOP_COINS_TO_SCAN, quote=quote_currency_used,
                ignore_stables_base=STABLECOINS_TO_IGNORE_BASE,
                _exchange_markets_tuple=markets_tuple_for_cache_key
            )
            if not base_trading_universe: st.error("Kein Basis-Handelsuniversum gefunden. Analyse gestoppt."); st.stop()
            
            current_options = set(st.session_state.multiselect_coin_options)
            new_options_from_scan = set(base_trading_universe)
            updated_options = sorted(list(current_options.union(new_options_from_scan)))
            if updated_options != st.session_state.multiselect_coin_options:
                 st.session_state.multiselect_coin_options = updated_options
            
            user_excluded_coins = st.session_state.get('coins_to_exclude_user_selection', [])
            st.session_state.trading_universe_filtered = [
                coin for coin in base_trading_universe if coin not in user_excluded_coins
            ]
            if not st.session_state.trading_universe_filtered: st.error("Nach Filterung durch Benutzerauswahl sind keine Coins mehr im Handelsuniversum. Analyse gestoppt."); st.stop()

            current_portfolio_asset_pairs = [s for s in CURRENT_PORTFOLIO_ALLOCATION.keys() if '/' in s]
            symbols_to_fetch_ohlcv = list(set(st.session_state.trading_universe_filtered + current_portfolio_asset_pairs))
            final_symbols_to_fetch = []
            for s_fetch in symbols_to_fetch_ohlcv:
                is_statically_excluded = s_fetch.upper() in PAIRS_TO_EXCLUDE_FROM_OPTIMIZATION_STATIC
                is_stablecoin_base = s_fetch.split('/')[0].upper() in STABLECOINS_TO_IGNORE_BASE and s_fetch.split('/')[1].upper() in STABLECOINS_TO_IGNORE_BASE
                if not is_statically_excluded and not is_stablecoin_base:
                    final_symbols_to_fetch.append(s_fetch)
            
            min_days_needed_for_fetch = max(
                OPTIMIZATION_MU_SPAN_DAYS + 10, OPTIMIZATION_COV_HISTORY_DAYS + 10,
                VOLA_PENALTY_ATR_HISTORY_WINDOW + 5, MIN_DATA_POINTS_FOR_ANALYSIS_GLOBAL, 30
            )
            st.session_state.hist_data_ohlcv = fetch_crypto_data_cached(
                symbols=final_symbols_to_fetch,
                min_data_points_for_analysis_param=MIN_DATA_POINTS_FOR_ANALYSIS_GLOBAL,
                optimization_mu_span_days_param=OPTIMIZATION_MU_SPAN_DAYS,
                optimization_cov_history_days_param=OPTIMIZATION_COV_HISTORY_DAYS,
                vola_penalty_atr_history_window_param=VOLA_PENALTY_ATR_HISTORY_WINDOW,
                timeframe='1d', limit=min_days_needed_for_fetch + 20
            )
            
            if not st.session_state.hist_data_ohlcv: st.error("Fehler beim Laden der historischen OHLCV-Daten. Analyse gestoppt."); st.stop()
            
            assets_for_fetching_tickers = list(st.session_state.hist_data_ohlcv.keys())
            if assets_for_fetching_tickers:
                try: st.session_state.current_exchange_tickers = exchange_instance.fetch_tickers(assets_for_fetching_tickers)
                except Exception as e_fetch_tickers: st.sidebar.warning(f"Fehler beim Laden der Ticker-Preise: {e_fetch_tickers}. Nutze letzte OHLC-Preise."); st.session_state.current_exchange_tickers = {}
            else: st.session_state.current_exchange_tickers = {}
            
            st.session_state.top_candidates_dict, st.session_state.assets_with_ta_data_final = \
                perform_technical_analysis(
                    st.session_state.hist_data_ohlcv,
                    atr_period_global_param=ATR_PERIOD,
                    atr_period_trade_param=ATR_PERIOD_TRADE
                )

            if not st.session_state.assets_with_ta_data_final: st.warning("Keine validen Daten nach TA-Berechnung vorhanden. Optimierung könnte unvollständig sein.");
            data_for_current_pf_metrics = { ast_curr: st.session_state.assets_with_ta_data_final[ast_curr] for ast_curr in current_portfolio_asset_pairs if ast_curr in st.session_state.assets_with_ta_data_final and not st.session_state.assets_with_ta_data_final[ast_curr].empty}
            allocation_current_pf_invested = {k:v for k,v in CURRENT_PORTFOLIO_ALLOCATION.items() if k in data_for_current_pf_metrics}
            if allocation_current_pf_invested: _, _, st.session_state.current_portfolio_sharpe = calculate_portfolio_metrics(data_for_current_pf_metrics, allocation_current_pf_invested, RISK_FREE_RATE)
            else: st.session_state.current_portfolio_sharpe = None

            assets_eligible_for_optimization = [
                s_opt for s_opt in st.session_state.trading_universe_filtered
                if s_opt in st.session_state.assets_with_ta_data_final and
                   not st.session_state.assets_with_ta_data_final[s_opt].empty and
                   s_opt.split('/')[0].upper() not in STABLECOINS_TO_IGNORE_BASE and
                   s_opt.upper() not in PAIRS_TO_EXCLUDE_FROM_OPTIMIZATION_STATIC
            ]
            st.session_state.opt_weights_current_run = None; st.session_state.opt_perf_data_current_run = None
            st.session_state.prices_df_for_discrete_alloc = None; st.session_state.opt_asset_details_df = pd.DataFrame()
            if assets_eligible_for_optimization:
                 st.session_state.opt_weights_current_run, st.session_state.opt_perf_data_current_run, st.session_state.prices_df_for_discrete_alloc, st.session_state.opt_asset_details_df = \
                     optimize_portfolio_sharpe(
                         assets_ta_data_dict=st.session_state.assets_with_ta_data_final,
                         risk_free_rate_param=RISK_FREE_RATE,
                         assets_for_optimization_list=assets_eligible_for_optimization,
                         max_assets_limit_param=MAX_ASSETS_IN_OPTIMIZED_PORTFOLIO,
                         coingecko_ids_map_param=st.session_state.coingecko_ids_map,
                         coingecko_categories_cache_param=st.session_state.coingecko_categories_cache,
                         current_tickers_dict_param=st.session_state.current_exchange_tickers
                      )
            else: st.sidebar.warning("Keine Assets für die Optimierung nach Filterung übrig. Zielportfolio bleibt leer.")
            st.session_state.optimized_assets_list_final = list(st.session_state.opt_weights_current_run.keys()) if st.session_state.opt_weights_current_run else []
            initial_cash_symbol = next((cs_key for cs_key in CURRENT_PORTFOLIO_ALLOCATION if cs_key.split('/')[0].upper() in STABLECOINS_TO_IGNORE_BASE), 'USDC')
            target_cash_percentage_current = CURRENT_PORTFOLIO_ALLOCATION.get(initial_cash_symbol, MIN_TOTAL_CASH_PCT)
            fng_history_df_for_cash = st.session_state.sentiment_data.get("fng_history_df")
            if ENABLE_DYNAMIC_CASH and fng_history_df_for_cash is not None and not fng_history_df_for_cash.empty:
                fng_df_for_cash_logic = fng_history_df_for_cash.copy()
                if len(fng_df_for_cash_logic) >= FNG_SMA_LONG + 5:
                    fng_df_for_cash_logic['sma_short'] = ta.trend.SMAIndicator(fng_df_for_cash_logic['value'], window=FNG_SMA_SHORT, fillna=False).sma_indicator()
                    fng_df_for_cash_logic['sma_long'] = ta.trend.SMAIndicator(fng_df_for_cash_logic['value'], window=FNG_SMA_LONG, fillna=False).sma_indicator()
                    if not fng_df_for_cash_logic.empty and all(col in fng_df_for_cash_logic.columns for col in ['value', 'sma_short', 'sma_long']) and \
                       not fng_df_for_cash_logic[['value', 'sma_short', 'sma_long']].iloc[-1].isnull().any():
                        latest_fng_row = fng_df_for_cash_logic.iloc[-1]; cash_change_reasons = []
                        if latest_fng_row['value'] >= FNG_TA_EXTREME_GREED_SELL_THRESHOLD and latest_fng_row['value'] < latest_fng_row['sma_short']:
                            target_cash_percentage_current = min(target_cash_percentage_current + DYNAMIC_CASH_ADJUSTMENT_PCT, MAX_TOTAL_CASH_PCT)
                            cash_change_reasons.append(f"F&G Gier ({latest_fng_row['value']:.0f}) & <SMA{FNG_SMA_SHORT}")
                        elif latest_fng_row['value'] <= FNG_TA_EXTREME_FEAR_BUY_THRESHOLD and latest_fng_row['value'] > latest_fng_row['sma_short']:
                            target_cash_percentage_current = max(target_cash_percentage_current - DYNAMIC_CASH_ADJUSTMENT_PCT, MIN_TOTAL_CASH_PCT)
                            cash_change_reasons.append(f"F&G Angst ({latest_fng_row['value']:.0f}) & >SMA{FNG_SMA_SHORT}")
                        if cash_change_reasons: st.sidebar.info(f"Dyn. Cash-Anpassung: Ziel auf {target_cash_percentage_current:.0%}. Grund: {', '.join(cash_change_reasons)}")
            st.session_state.target_cash_allocation_final_pct = target_cash_percentage_current
            if 'opt_asset_details_df' in st.session_state and not st.session_state.opt_asset_details_df.empty:
                final_invest_capital = CURRENT_PORTFOLIO_VALUE_USD * (1 - st.session_state.target_cash_allocation_final_pct)
                temp_df_opt_details = st.session_state.opt_asset_details_df.copy()
                if 'Gewichtung' in temp_df_opt_details.columns:
                    temp_df_opt_details['Gewichtung_float'] = temp_df_opt_details['Gewichtung'].astype(str).str.rstrip('%').astype('float') / 100.0
                    temp_df_opt_details["Wert (USD)"] = (temp_df_opt_details['Gewichtung_float'] * final_invest_capital).apply(lambda x: f"${x:,.2f}")
                    temp_df_opt_details.drop(columns=['Gewichtung_float'], inplace=True, errors='ignore')
                st.session_state.opt_asset_details_df = temp_df_opt_details
            
            if st.session_state.opt_weights_current_run:
                st.session_state.trading_setups_df = generate_trade_recommendations_from_portfolio(
                    optimized_weights=st.session_state.opt_weights_current_run,
                    portfolio_value_usd=CURRENT_PORTFOLIO_VALUE_USD,
                    assets_ta_data=st.session_state.assets_with_ta_data_final,
                    current_tickers=st.session_state.current_exchange_tickers
                )
            else:
                st.session_state.trading_setups_df = pd.DataFrame()

            if st.session_state.optimized_assets_list_final:
                st.session_state.weekday_df = analyze_weekday_performance(st.session_state.assets_with_ta_data_final, st.session_state.optimized_assets_list_final[:7])
                corr_plot_assets_list = st.session_state.optimized_assets_list_final[:10]
                if corr_plot_assets_list: st.session_state.corr_matrix = get_correlation_matrix(st.session_state.assets_with_ta_data_final, corr_plot_assets_list)

            if 'last_opt_weights' in st.session_state and st.session_state.last_opt_weights and st.session_state.opt_weights_current_run:
                rebalancing_advice_list = []
                all_assets_for_rebal = set(st.session_state.last_opt_weights.keys()) | set(st.session_state.opt_weights_current_run.keys())
                last_target_cash_percentage = st.session_state.get('last_target_cash_pct', st.session_state.target_cash_allocation_final_pct)
                current_target_cash_percentage = st.session_state.target_cash_allocation_final_pct
                old_investment_capital = CURRENT_PORTFOLIO_VALUE_USD * (1 - last_target_cash_percentage)
                new_investment_capital = CURRENT_PORTFOLIO_VALUE_USD * (1 - current_target_cash_percentage)
                for asset_rebal in all_assets_for_rebal:
                    old_weight_rebal = st.session_state.last_opt_weights.get(asset_rebal, 0)
                    new_weight_rebal = st.session_state.opt_weights_current_run.get(asset_rebal, 0)
                    old_value_rebal = old_weight_rebal * old_investment_capital
                    new_value_rebal = new_weight_rebal * new_investment_capital
                    change_in_value = new_value_rebal - old_value_rebal
                    price_for_rebal_units_calc = st.session_state.current_exchange_tickers.get(asset_rebal, {}).get('last')
                    if price_for_rebal_units_calc is None or price_for_rebal_units_calc <= 0:
                        if st.session_state.prices_df_for_discrete_alloc is not None and asset_rebal in st.session_state.prices_df_for_discrete_alloc.columns:
                             price_for_rebal_units_calc = st.session_state.prices_df_for_discrete_alloc[asset_rebal].iloc[-1]
                        else: price_for_rebal_units_calc = 0
                    change_in_units_str = f"{change_in_value / price_for_rebal_units_calc:.4f}" if price_for_rebal_units_calc > 0 else "N/A"
                    if abs(change_in_value) > MIN_USD_ALLOCATION_PER_ASSET * 0.10 :
                         rebalancing_advice_list.append({
                             "Asset": asset_rebal,
                             "Alte Gew. (Investiert)": f"{old_weight_rebal:.2%}",
                             "Neue Gew. (Investiert)": f"{new_weight_rebal:.2%}",
                             "Alter Wert USD": f"${old_value_rebal:,.0f}",
                             "Neuer Wert USD": f"${new_value_rebal:,.0f}",
                             "Änderung USD": f"${change_in_value:,.0f}",
                             "Aktion (Stück)": f"{'KAUFEN' if change_in_value > 0 else 'VERKAUFEN'}: {change_in_units_str}"
                         })
                st.session_state.rebalancing_advice_df = pd.DataFrame(rebalancing_advice_list)
            st.session_state.last_opt_weights = st.session_state.opt_weights_current_run.copy() if st.session_state.opt_weights_current_run else {}
            st.session_state.last_target_cash_pct = st.session_state.target_cash_allocation_final_pct
            st.session_state['results_ready'] = True; st.session_state['analysis_in_progress'] = False
            st.success("Analyse abgeschlossen!")
            st.rerun()

    if st.session_state.get('results_ready', False) and not st.session_state.get('analysis_in_progress', False):
        st.header(f"📊 Ergebnisse der Analyse vom {datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M')} UTC")
        col_summary1, col_summary2 = st.columns(2)
        with col_summary1:
            st.subheader("📄 Executive Summary")
            if LAST_DATA_DOWNLOAD_TIMESTAMP_UTC: st.caption(f"Datenstand (hist. OHLCV): Bis ca. {LAST_DATA_DOWNLOAD_TIMESTAMP_UTC.strftime('%d.%m.%Y %H:%M')} UTC")
            fng_val_display = st.session_state.sentiment_data.get('fear_greed_value')
            fng_cls_display = st.session_state.sentiment_data.get('fear_greed_classification')
            st.metric("Fear & Greed Index", f"{fng_val_display} ({fng_cls_display})", delta=None, help="Aktueller Wert des Crypto Fear & Greed Index von alternative.me")
            st.markdown(f"[Quelle: alternative.me](https://alternative.me/crypto/fear-and-greed-index/)", unsafe_allow_html=True)
            current_sharpe_display = st.session_state.get('current_portfolio_sharpe')
            if current_sharpe_display is not None: st.metric("Sharpe Aktuell (investiert)", f"{current_sharpe_display:.2f}", help="Berechnete Sharpe Ratio des aktuellen, investierten Portfolioteils (ohne Cash).")
            else: st.metric("Sharpe Aktuell (investiert)", "N/A", help="Konnte nicht berechnet werden (z.B. keine investierten Assets).")
            st.info(f"**Automatischer Ziel-Cash-Anteil:** {st.session_state.target_cash_allocation_final_pct:.0%}")
        with col_summary2:
            st.subheader("⭐ Performance des optimierten Zielportfolios")
            opt_perf_data_display = st.session_state.get('opt_perf_data_current_run', {})
            opt_weights_display = st.session_state.get('opt_weights_current_run')
            if opt_perf_data_display and opt_perf_data_display.get("annual_return") is not None and opt_weights_display:
                st.metric("Progn. Sharpe Ratio", f"{opt_perf_data_display['sharpe_ratio']:.2f}", help="Erwartete Sharpe Ratio des optimierten Portfolios (maximiert).")
                st.metric("Progn. Jährl. Rendite", f"{opt_perf_data_display['annual_return']:.2%}", help="Erwartete annualisierte Rendite.")
                st.metric("Progn. Tagesrendite", f"{opt_perf_data_display['daily_return']:.4%}", help="Erwartete durchschnittliche Tagesrendite.")
                st.metric("Progn. Jährl. Volatilität", f"{opt_perf_data_display['annual_volatility']:.2%}", help="Erwartete annualisierte Volatilität.")
                st.caption(f"Basiert auf {len(opt_weights_display)} Assets von max. {MAX_ASSETS_IN_OPTIMIZED_PORTFOLIO} erlaubten Assets.")
            else: st.warning("Optimiertes Zielportfolio konnte nicht berechnet werden oder hat keine gültige Performance.")
        st.markdown("---")
        
        trading_setups_df_display = st.session_state.get('trading_setups_df')
        if trading_setups_df_display is not None and not trading_setups_df_display.empty:
            st.subheader("📈 Optimierte Trading-Portfolio Setups")
            df_display_trading_setups = trading_setups_df_display.copy()
            if 'TV Chart' in df_display_trading_setups.columns:
                 df_display_trading_setups['TV Chart'] = df_display_trading_setups['TV Chart'].apply(lambda x_link: make_html_link_clickable(x_link, "Chart"))
            
            cols_to_show_trading = ["Richtung", "Gewichtung im Portfolio", "Entry Preis", "Stop-Loss", "Take-Profit 1", "Take-Profit 2", "Positionsgröße (USD)", "Kalk. Risiko für diesen Trade (USD)", "TV Chart"]
            final_cols_for_trading_html = [col for col in cols_to_show_trading if col in df_display_trading_setups.columns]

            if 'Asset' in df_display_trading_setups.columns:
                st.markdown(df_display_trading_setups.set_index('Asset')[final_cols_for_trading_html].to_html(escape=False, render_links=True, classes=["dataframe", "trades-table"]), unsafe_allow_html=True)
            else:
                st.warning("Keine 'Asset'-Spalte in Trading-Setups gefunden.")

            st.markdown("---")
            st.write("**Charts zu den Trading Setups (letzte 120 Tage):**")
            num_setups = len(df_display_trading_setups)
            cols_per_row_setup_charts = min(num_setups, 2)
            if num_setups > 0:
                for i_setup in range(0, num_setups, cols_per_row_setup_charts):
                    cols_setup_charts = st.columns(cols_per_row_setup_charts)
                    for j_setup_col_idx in range(cols_per_row_setup_charts):
                        if i_setup + j_setup_col_idx < num_setups:
                            with cols_setup_charts[j_setup_col_idx]:
                                setup_rec = df_display_trading_setups.iloc[i_setup + j_setup_col_idx]
                                asset_symbol_for_setup_chart = setup_rec['Asset']
                                if asset_symbol_for_setup_chart in st.session_state.assets_with_ta_data_final:
                                    df_for_setup_chart = st.session_state.assets_with_ta_data_final[asset_symbol_for_setup_chart].copy().tail(120)
                                    if not df_for_setup_chart.empty:
                                        with st.expander(f"Chart für {asset_symbol_for_setup_chart} (Trading Setup)", expanded=False):
                                            fig_trade_asset_chart = plot_asset_chart_st(df_for_setup_chart, asset_symbol_for_setup_chart, indicators_to_plot=['SMA20', 'SMA50', 'BB'], chart_title_prefix=f"Trading Setup für")
                                            if fig_trade_asset_chart: st.pyplot(fig_trade_asset_chart); plt.close(fig_trade_asset_chart)
                                            else: st.caption("Chart für dieses Setup konnte nicht erstellt werden.")
                                else: st.caption(f"Keine Chartdaten für {asset_symbol_for_setup_chart} in TA-Daten gefunden.")
            else:
                st.info("Keine Trading Setups mit Charts anzuzeigen.")
        else:
            st.info("Keine Trading Setups aus dem optimierten Portfolio generiert.")
        st.markdown("---")


        st.subheader("😨 Fear & Greed Index Verlauf & Technische Analyse")
        fng_hist_df_for_plotting = st.session_state.sentiment_data.get("fng_history_df")
        if fng_hist_df_for_plotting is not None and not fng_hist_df_for_plotting.empty and 'value' in fng_hist_df_for_plotting.columns:
            fig_fng_plot = None
            df_fng_to_plot = fng_hist_df_for_plotting.copy()
            if len(df_fng_to_plot) >= max(FNG_SMA_SHORT, FNG_SMA_LONG, 15):
                try:
                    df_fng_to_plot['sma_short'] = ta.trend.SMAIndicator(df_fng_to_plot['value'], window=FNG_SMA_SHORT, fillna=False).sma_indicator()
                    df_fng_to_plot['sma_long'] = ta.trend.SMAIndicator(df_fng_to_plot['value'], window=FNG_SMA_LONG, fillna=False).sma_indicator()
                    fig_fng_plot, ax_fng_plot = plt.subplots(figsize=(12,5))
                    ax_fng_plot.plot(df_fng_to_plot.index, df_fng_to_plot['value'], label='F&G Index', color='black', lw=1.5)
                    if df_fng_to_plot['sma_short'].notna().any(): ax_fng_plot.plot(df_fng_to_plot.index, df_fng_to_plot['sma_short'], label=f'SMA({FNG_SMA_SHORT})', color='blue', ls='--', lw=1)
                    if df_fng_to_plot['sma_long'].notna().any(): ax_fng_plot.plot(df_fng_to_plot.index, df_fng_to_plot['sma_long'], label=f'SMA({FNG_SMA_LONG})', color='orange', ls=':', lw=1)
                    if ENABLE_DYNAMIC_CASH:
                        ax_fng_plot.axhline(FNG_TA_EXTREME_FEAR_BUY_THRESHOLD, color='green', linestyle='-.', linewidth=0.8, label=f'Angst Zone (<{FNG_TA_EXTREME_FEAR_BUY_THRESHOLD})')
                        ax_fng_plot.axhline(FNG_TA_EXTREME_GREED_SELL_THRESHOLD, color='red', linestyle='-.', linewidth=0.8, label=f'Gier Zone (>{FNG_TA_EXTREME_GREED_SELL_THRESHOLD})')
                    ax_fng_plot.set_title("Fear & Greed Index mit SMAs und Zonen (falls Dyn. Cash aktiv)", fontsize=14); ax_fng_plot.set_ylabel("Index Wert")
                    ax_fng_plot.legend(loc='upper left'); ax_fng_plot.grid(True, linestyle=':', alpha=0.7); fig_fng_plot.autofmt_xdate()
                    ax_fng_plot.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%y'))
                    plt.tight_layout()
                except Exception as e_fng_chart: st.caption(f"Fehler beim Erstellen des F&G Detail-Charts: {e_fng_chart}"); fig_fng_plot = None
            else: st.info("Zu wenig historische F&G-Daten für SMA-Berechnung im Chart.")
            if fig_fng_plot: st.pyplot(fig_fng_plot); plt.close(fig_fng_plot)
            elif not df_fng_to_plot.empty and fig_fng_plot is None:
                fig_fng_raw_plot, ax_fng_raw_plot = plt.subplots(figsize=(12,5))
                ax_fng_raw_plot.plot(df_fng_to_plot.index, df_fng_to_plot['value'], label='F&G Index (Roh)', color='black', lw=1.5)
                ax_fng_raw_plot.set_title("Fear & Greed Index (Rohdaten)", fontsize=14); ax_fng_raw_plot.set_ylabel("Index Wert")
                ax_fng_raw_plot.legend(loc='upper left'); ax_fng_raw_plot.grid(True, linestyle=':', alpha=0.7); fig_fng_raw_plot.autofmt_xdate()
                ax_fng_raw_plot.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.%y')); plt.tight_layout(); st.pyplot(fig_fng_raw_plot); plt.close(fig_fng_raw_plot)
        else: st.info("Fear & Greed Index Chart nicht verfügbar (Historie nicht geladen oder 'value'-Spalte fehlt).")
        st.markdown("---")
        
        opt_weights_for_discrete = st.session_state.get('opt_weights_current_run')
        prices_df_for_discrete = st.session_state.get('prices_df_for_discrete_alloc')
        if opt_weights_for_discrete and prices_df_for_discrete is not None and not prices_df_for_discrete.empty:
            st.subheader("💰 Diskrete Allokation (Stückzahlen für Zielportfolio)")
            try:
                latest_prices_for_discrete_alloc = {}
                valid_tickers_for_discrete = st.session_state.get('current_exchange_tickers', {})
                for asset_da in opt_weights_for_discrete.keys():
                    if asset_da in valid_tickers_for_discrete and valid_tickers_for_discrete[asset_da].get('last') is not None and valid_tickers_for_discrete[asset_da]['last'] > 0:
                        latest_prices_for_discrete_alloc[asset_da] = valid_tickers_for_discrete[asset_da]['last']
                    elif asset_da in prices_df_for_discrete.columns and not pd.isna(prices_df_for_discrete[asset_da].iloc[-1]) and prices_df_for_discrete[asset_da].iloc[-1] > 0 :
                        latest_prices_for_discrete_alloc[asset_da] = prices_df_for_discrete[asset_da].iloc[-1]
                capital_to_invest_discrete = CURRENT_PORTFOLIO_VALUE_USD * (1 - st.session_state.target_cash_allocation_final_pct)
                st.write(f"**Verfügbares Kapital für Investitionen (nach Cash-Anteil):** ${capital_to_invest_discrete:,.2f}")
                eligible_weights_for_da = {}
                for asset_da_filt, weight_da_filt in opt_weights_for_discrete.items():
                    if asset_da_filt in latest_prices_for_discrete_alloc:
                        if weight_da_filt * capital_to_invest_discrete >= MIN_USD_ALLOCATION_PER_ASSET:
                            eligible_weights_for_da[asset_da_filt] = weight_da_filt
                
                if not eligible_weights_for_da:
                    st.info(f"Keine Assets erfüllen die Mindestallokation von ${MIN_USD_ALLOCATION_PER_ASSET:,.2f} oder es fehlen Preise.")
                else:
                    sum_eligible_weights = sum(eligible_weights_for_da.values())
                    if sum_eligible_weights > 1e-6:
                        normalized_discrete_weights = {k: v / sum_eligible_weights for k, v in eligible_weights_for_da.items()}
                        actual_capital_for_discrete_alloc = capital_to_invest_discrete * sum_eligible_weights
                        
                        st.write(f"Assets nach Mindestfilter & Preisverfügbarkeit: {len(normalized_discrete_weights)}, "
                                 f"Kapital dafür: ${actual_capital_for_discrete_alloc:,.2f}")

                        prices_series_for_da_final = pd.Series({
                            asset: latest_prices_for_discrete_alloc[asset]
                            for asset in normalized_discrete_weights.keys()
                            if asset in latest_prices_for_discrete_alloc
                        })

                        if not normalized_discrete_weights or prices_series_for_da_final.empty or actual_capital_for_discrete_alloc < 1:
                            st.info("Nicht genügend Daten oder Kapital für die diskrete Allokation nach Filterung.")
                        else:
                            missing_prices_for_norm = [asset for asset in normalized_discrete_weights if asset not in prices_series_for_da_final]
                            if missing_prices_for_norm:
                                st.warning(f"Fehlende Preise für folgende normalisierte Assets: {missing_prices_for_norm}. Diskrete Allokation könnte unvollständig sein.")
                                normalized_discrete_weights = {k:v for k,v in normalized_discrete_weights.items() if k in prices_series_for_da_final}
                                if not normalized_discrete_weights:
                                    st.error("Nach Preisprüfung keine Assets mehr für diskrete Allokation übrig.")
                                    # return # Funktion hier beenden oder Fallback

                            if normalized_discrete_weights: # Erneute Prüfung
                                da_instance = DiscreteAllocation(
                                    normalized_discrete_weights,
                                    prices_series_for_da_final.loc[list(normalized_discrete_weights.keys())],
                                    total_portfolio_value=actual_capital_for_discrete_alloc,
                                    short_ratio=None
                                )
                                allocation_units, leftover_cash = da_instance.greedy_portfolio()
                                discrete_alloc_results_list = []
                                total_allocated_value_check = 0
                                for sym_alloc, num_units in sorted(allocation_units.items(), key=lambda item: item[1] * prices_series_for_da_final.get(item[0],0), reverse=True):
                                    price_for_value_calc = prices_series_for_da_final.get(sym_alloc)
                                    if price_for_value_calc and price_for_value_calc > 0:
                                        value_of_units = num_units * price_for_value_calc; total_allocated_value_check += value_of_units
                                        discrete_alloc_results_list.append({"Asset": sym_alloc, "Stückzahl": f"{num_units:.4f}", "Zielwert (USD)": f"${value_of_units:,.2f}"})
                                if discrete_alloc_results_list:
                                    st.dataframe(pd.DataFrame(discrete_alloc_results_list).set_index('Asset'))
                                    st.write(f"**Summe Zielwerte diskrete Allokation:** ${total_allocated_value_check:,.2f}")
                                    st.write(f"**Nicht zugewiesenes Kapital (Rest aus diskreter Allokation):** ${leftover_cash:.2f}")
                                    st.caption(f"Hinweis: Der Restbetrag kann aufgrund von Rundungen und der 'greedy' Natur der Allokation entstehen.")
                                else: st.info("Keine diskrete Allokation nach Filterung möglich.")
                    else:
                        st.info("Summe der Gewichte der berechtigten Assets ist zu gering für eine sinnvolle Allokation.")
            except Exception as e_discrete_alloc: st.error(f"FEHLER bei diskreter Allokation: {e_discrete_alloc}"); traceback.print_exc()
        st.markdown("---")

        rebal_df_display = st.session_state.get('rebalancing_advice_df')
        if rebal_df_display is not None and not rebal_df_display.empty:
            with st.expander("🔄 Rebalancing Vorschläge (Vergleich zur vorherigen Optimierung)", expanded=False):
                st.dataframe(rebal_df_display.set_index('Asset'))
                st.caption("Zeigt vorgeschlagene Käufe/Verkäufe, um vom vorherigen zum aktuellen optimierten Portfolio zu gelangen.")
        st.markdown("---")
        
        opt_asset_details_df_display = st.session_state.get('opt_asset_details_df')
        if opt_asset_details_df_display is not None and not opt_asset_details_df_display.empty:
            with st.expander("🎯 Details zum optimierten Portfolio (Asset-Selektion)", expanded=False):
                df_display_optimized_assets = opt_asset_details_df_display.copy()
                if 'Homepage' in df_display_optimized_assets.columns: df_display_optimized_assets['Homepage'] = df_display_optimized_assets['Homepage'].apply(lambda x: make_html_link_clickable(x, "Website"))
                if 'Asset' in df_display_optimized_assets.columns: df_display_optimized_assets['TV Chart'] = df_display_optimized_assets['Asset'].apply(lambda x: make_html_link_clickable(f"https://www.tradingview.com/chart/?symbol={TARGET_EXCHANGE.upper()}:{x.replace('/', '')}", "Chart"))
                cols_to_show_in_html_details = ["Gewichtung", "Wert (USD)", "Akt. Preis", "Akt. RSI(14)", "Perf. 30T", "TA Signal (Portfolio)", "Sektor(en)", "Homepage", "TV Chart", "Erw. Rend. (p.a.)", "Asset Vola (p.a.)", "Beschreibung"]
                final_cols_for_html_render = [col_html for col_html in cols_to_show_in_html_details if col_html in df_display_optimized_assets.columns]
                if 'Asset' in df_display_optimized_assets.columns:
                    df_to_render_as_html = df_display_optimized_assets.set_index('Asset')[final_cols_for_html_render]
                    st.markdown(df_to_render_as_html.to_html(escape=False, render_links=True, classes=["dataframe", "details-table"]), unsafe_allow_html=True)

        print_footer_st()
    elif st.session_state.get('analysis_in_progress', False):
        pass
    else:
        st.info("Willkommen! Bitte passe die Konfigurationen in der Sidebar an (falls gewünscht) und klicke auf '🚀 Analyse Starten / Aktualisieren', um die Analyse zu beginnen.")
        print_footer_st()

if __name__ == "__main__":
    try:
        run_analysis()
    except Exception as e_main:
        st.error(f"Ein unerwarteter Fehler ist im Hauptablauf aufgetreten: {e_main}")
        st.error("Bitte überprüfen Sie die Konsole für Details und versuchen Sie es erneut oder starten Sie die App neu.")
        traceback.print_exc()