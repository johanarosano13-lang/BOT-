"""
Bot de alertas swing (Telegram). Solo avisa, no opera.
  Semanal: precio entrando en zona de Order Block grande (sin tocar desde su formación)
  Diario : divergencia grande del RSI
  Horario: cruce de medias (EMA rápida/lenta), con confluencia
Hace una pasada por ejecución (GitHub Actions lo corre cada hora). Datos: Yahoo Finance.
  python ema_alert_bot.py         -> revisa todo y avisa
  python ema_alert_bot.py --test  -> manda un mensaje de prueba a Telegram
"""
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests

# ---------------- ACTIVOS (tickers de Yahoo Finance) ----------------
FOREX = [f"{p}=X" for p in (
    "EURUSD GBPUSD USDJPY USDCHF AUDUSD USDCAD NZDUSD "
    "EURGBP EURJPY EURCHF EURAUD EURCAD EURNZD GBPJPY GBPCHF GBPAUD GBPCAD GBPNZD "
    "AUDJPY AUDCAD AUDCHF AUDNZD CADJPY CADCHF CHFJPY NZDJPY NZDCAD NZDCHF "
    "USDMXN USDZAR USDTRY USDSEK USDNOK USDDKK USDSGD USDHKD USDPLN USDCNH").split()]
INDICES = ["^GSPC", "^NDX", "^DJI", "^RUT", "^FTSE", "^GDAXI", "^FCHI", "^STOXX50E",
           "^N225", "^HSI", "^AXJO", "^IBEX", "^AEX", "^SSMI"]
COMMODITIES = ["GC=F", "SI=F", "HG=F", "PL=F", "PA=F", "CL=F", "BZ=F", "NG=F",
               "KC=F", "SB=F", "CC=F", "ZC=F", "ZW=F", "ZS=F"]
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "LTC-USD",
          "BNB-USD", "AVAX-USD", "LINK-USD", "DOT-USD", "BCH-USD"]
STOCKS = ("AAPL MSFT AMZN GOOGL META NVDA TSLA AMD NFLX INTC AVGO ORCL CRM ADBE QCOM MU PLTR COIN MSTR "
          "UBER SHOP PYPL HOOD ARM TSM BABA NIO RIVN BA DIS JPM BAC GS V MA WMT KO PEP XOM CVX PFE "
          "JNJ NKE MCD SBUX F GM").split()
WATCHLIST = FOREX + INDICES + COMMODITIES + CRYPTO + STOCKS
# (para quitar un activo, borralo de la lista de arriba; para sumar, agregá su ticker de Yahoo)

NAMES = {"^GSPC": "S&P 500", "^NDX": "Nasdaq 100", "^DJI": "Dow Jones", "^RUT": "Russell 2000",
         "^FTSE": "FTSE 100", "^GDAXI": "DAX", "^FCHI": "CAC 40", "^STOXX50E": "Euro Stoxx 50",
         "^N225": "Nikkei 225", "^HSI": "Hang Seng", "^AXJO": "ASX 200", "^IBEX": "IBEX 35",
         "^AEX": "AEX", "^SSMI": "SMI", "GC=F": "Oro", "SI=F": "Plata", "HG=F": "Cobre",
         "PL=F": "Platino", "PA=F": "Paladio", "CL=F": "WTI", "BZ=F": "Brent", "NG=F": "Gas natural"}

# ---------------- PARÁMETROS ----------------
# SEMANAL: order block
OB_LOOKBACK = 90         # velas semanales hacia atrás
OB_IMPULSE_ATR = 3.0     # "grande": el impulso posterior mueve >= 3 ATR
OB_IMPULSE_CANDLES = 5   # ...dentro de 5 velas
OB_ZONE = "body"         # "body" = cuerpo de la vela / "range" = rango completo con mechas
OB_ONLY_FRESH = True     # solo OB que el precio NO volvió a tocar desde que se formó
NEAR_PCT = 0.03          # el log lista los OB a menos de 3% del precio
# DIARIO: divergencia RSI
RSI_LEN = 14
PIVOT_N = 3              # un pivote se confirma con 3 velas a cada lado (3 días de demora)
DIV_LOOKBACK = 120
DIV_MIN_GAP = 10         # velas mínimas entre los dos pivotes
DIV_MIN_RSI_DIFF = 5     # diferencia mínima de RSI entre pivotes
DIV_MIN_PRICE_ATR = 0.5  # el precio debe superar el pivote anterior por >= 0.5 ATR
# HORARIO: cruce de medias
FAST, SLOW = 5, 10
GATE_1H = True           # True: avisa el cruce solo si coincide con OB semanal o divergencia diaria
GATE_DIV_HOURS = 72      # cuánto vale una divergencia diaria como contexto

STATE_FILE = "alert_state.json"
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
# --------------------------------------------

NAME = {"bull": "ALCISTA", "bear": "BAJISTA"}
DELTA = {"1h": pd.Timedelta(hours=1), "1d": pd.Timedelta(days=1), "1w": pd.Timedelta(days=7)}
YF_PERIOD = {"1h": "60d", "1d": "5y"}
SEND_FAILED = False
NEAR = []


def label(sym):
    return NAMES.get(sym) and f"{NAMES[sym]} ({sym})" or sym.replace("=X", "")


def send(msg):
    global SEND_FAILED
    try:
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          json={"chat_id": CHAT_ID, "text": msg}, timeout=15)
        if r.ok:
            return True
        print(f"ERROR Telegram {r.status_code}: {r.text}")
    except Exception as e:
        print("ERROR Telegram:", str(e).replace(TOKEN or "", "***"))
    SEND_FAILED = True
    return False


def fetch_one(sym, tf):
    """Velas cerradas de un activo (None si Yahoo no responde)."""
    import yfinance as yf
    for attempt in range(2):
        try:
            df = yf.Ticker(sym).history(period=YF_PERIOD[tf], interval=tf, auto_adjust=True)
            if df is not None and len(df):
                df.columns = [c.lower() for c in df.columns]
                df = df[["open", "high", "low", "close"]].dropna()
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                return df[df.index + DELTA[tf] <= pd.Timestamp.now(tz="UTC")]
        except Exception:
            pass
        time.sleep(1.5)
    return None


def fetch_all(tickers, tf):
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(lambda t: fetch_one(t, tf), tickers))
    return dict(zip(tickers, res))


def to_weekly(d):
    """Velas semanales (lunes a domingo) a partir de las diarias."""
    w = d.resample("W-MON", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return w[w.index + DELTA["1w"] <= pd.Timestamp.now(tz="UTC")]


def atr(df, n=14):
    c = df["close"]
    tr = pd.concat([df.high - df.low, (df.high - c.shift()).abs(),
                    (df.low - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def rsi(c, n=RSI_LEN):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def pivots(x, n, kind):
    out = []
    for i in range(n, len(x) - n):
        w = x[i - n:i + n + 1]
        if (kind == "low" and x[i] == w.min()) or (kind == "high" and x[i] == w.max()):
            if not out or i - out[-1] > n:  # evita pivotes duplicados por empates
                out.append(i)
    return out


# ---------- SEMANAL: order blocks ----------
def find_obs(df):
    """OB alcista: última vela bajista antes de un impulso alcista grande (espejo para bajista).
    Se descarta si un cierre posterior lo atraviesa o (OB_ONLY_FRESH) si el precio ya volvió a tocarlo.
    Devuelve (dirección, zona_baja, zona_alta, semana, impulso_en_ATR)."""
    d = df.iloc[-OB_LOOKBACK:]
    a = atr(d).values
    o, h, l, c = (d[k].values for k in ("open", "high", "low", "close"))
    n = len(d)
    obs = []
    for i in range(14, n - 2):
        if np.isnan(a[i]):
            continue
        end = min(i + 1 + OB_IMPULSE_CANDLES, n)
        if c[i] < o[i] and c[i + 1] > o[i + 1]:
            dirn, move = "bull", h[i + 1:end].max() - c[i]
        elif c[i] > o[i] and c[i + 1] < o[i + 1]:
            dirn, move = "bear", c[i] - l[i + 1:end].min()
        else:
            continue
        if move < OB_IMPULSE_ATR * a[i]:
            continue
        lo, hi = (min(o[i], c[i]), max(o[i], c[i])) if OB_ZONE == "body" else (l[i], h[i])
        if dirn == "bull":
            if (c[i + 1:] < l[i]).any():
                continue
            if OB_ONLY_FRESH and (l[i + 2:] <= hi).any():
                continue
        else:
            if (c[i + 1:] > h[i]).any():
                continue
            if OB_ONLY_FRESH and (h[i + 2:] >= lo).any():
                continue
        obs.append((dirn, lo, hi, d.index[i], move / a[i]))
    return obs


def check_ob(sym, px, obs, st, ctx):
    sent = st["ob_sent"].setdefault(sym, [])
    ctx["ob"] = None
    status = f"fuera de zona ({len(obs)} OB vigentes)"
    for dirn, lo, hi, ts, mult in obs:
        inside = lo <= px <= hi
        dist = 0.0 if inside else min(abs(px - lo), abs(px - hi)) / px
        if dist <= NEAR_PCT:
            NEAR.append((sym, dirn, dist))
        if not inside:
            continue
        ctx["ob"] = dirn
        oid = f"{dirn}|{ts:%Y-%m-%d}"
        status = f"dentro de OB {NAME[dirn]}"
        if oid in sent:
            status += " (ya avisado)"
            continue
        zona = "cuerpo de la vela" if OB_ZONE == "body" else "rango completo"
        ok = send(f"{label(sym)} · SEMANAL\nPrecio dentro de OB {NAME[dirn]} grande (sin tocar desde su formación)\n"
                  f"Zona ({zona}): {lo:.5g} – {hi:.5g}\nVela del OB: semana del {ts:%Y-%m-%d}\n"
                  f"Impulso posterior: {mult:.1f} ATR\nPrecio: {px:.5g}")
        if ok:
            sent.append(oid)
            status += " (alerta enviada)"
        else:
            status += " (envío FALLÓ)"
    return status


# ---------- DIARIO: divergencia RSI ----------
def find_divergence(df):
    r = rsi(df.close).iloc[-DIV_LOOKBACK:].values
    d = df.iloc[-DIV_LOOKBACK:]
    a = atr(df).iloc[-1]
    n = len(d)
    if n < 40 or np.isnan(a):
        return None
    recent = lambda i: i >= n - 1 - PIVOT_N - 2  # pivote recién confirmado
    lo, hi = d.low.values, d.high.values
    pl = pivots(lo, PIVOT_N, "low")
    if len(pl) >= 2:
        i1, i2 = pl[-2], pl[-1]
        if (recent(i2) and i2 - i1 >= DIV_MIN_GAP and lo[i1] - lo[i2] >= DIV_MIN_PRICE_ATR * a
                and r[i2] - r[i1] >= DIV_MIN_RSI_DIFF and r[i1] < 40):
            return "bull", d.index[i2], r[i1], r[i2], lo[i1], lo[i2]
    ph = pivots(hi, PIVOT_N, "high")
    if len(ph) >= 2:
        i1, i2 = ph[-2], ph[-1]
        if (recent(i2) and i2 - i1 >= DIV_MIN_GAP and hi[i2] - hi[i1] >= DIV_MIN_PRICE_ATR * a
                and r[i1] - r[i2] >= DIV_MIN_RSI_DIFF and r[i1] > 60):
            return "bear", d.index[i2], r[i1], r[i2], hi[i1], hi[i2]
    return None


def check_div(sym, d1, st, ctx):
    res = find_divergence(d1)
    if not res:
        return "sin divergencia"
    dirn, ts, r1, r2, p1, p2 = res
    ctx["div"] = {"dir": dirn, "seen": pd.Timestamp.now(tz="UTC").isoformat()}
    key = f"{dirn}|{ts}"
    if st["last"].get(f"{sym}|1d") == key:
        return f"{NAME[dirn]} (ya avisada)"
    ok = send(f"{label(sym)} · DIARIO\nDivergencia {NAME[dirn]} grande en RSI\n"
              f"Precio: {p1:.5g} → {p2:.5g}\nRSI: {r1:.1f} → {r2:.1f}\nPivote: {ts:%Y-%m-%d}")
    if ok:
        st["last"][f"{sym}|1d"] = key
    return f"{NAME[dirn]} (alerta {'enviada' if ok else 'FALLÓ'})"


# ---------- HORARIO: cruce de medias ----------
def check_cross(sym, h1, st, ctx):
    c = h1.close
    gap = c.ewm(span=FAST, adjust=False).mean() - c.ewm(span=SLOW, adjust=False).mean()
    g0, g1 = gap.iloc[-2], gap.iloc[-1]
    if g0 <= 0 < g1:
        dirn = "bull"
    elif g0 >= 0 > g1:
        dirn = "bear"
    else:
        return "sin cruce"
    ts = h1.index[-1]
    key = f"{dirn}|{ts}"
    k = f"{sym}|1h"
    prev = st["last"].get(k)
    if prev == key:
        return f"cruce {NAME[dirn]} (ya avisado)"
    if prev == key + "|filtrado":
        return f"cruce {NAME[dirn]} sin confluencia (ya descartado, no se avisa)"
    why = []
    if ctx.get("ob") == dirn:
        why.append("precio en OB semanal " + NAME[dirn].lower())
    div = ctx.get("div")
    if div and div["dir"] == dirn and \
            pd.Timestamp.now(tz="UTC") - pd.Timestamp(div["seen"]) <= pd.Timedelta(hours=GATE_DIV_HOURS):
        why.append("divergencia RSI diaria " + NAME[dirn].lower())
    if GATE_1H and not why:
        st["last"][k] = key + "|filtrado"
        return f"cruce {NAME[dirn]} sin confluencia (no se avisa)"
    ok = send(f"{label(sym)} · HORARIO\nCruce {NAME[dirn]} EMA{FAST}/EMA{SLOW}\nPrecio: {c.iloc[-1]:.5g}\n"
              f"Contexto: {', '.join(why) if why else 'sin confluencia'}\nVela: {ts:%Y-%m-%d %H:%M} UTC")
    if ok:
        st["last"][k] = key
    return f"cruce {NAME[dirn]} (alerta {'enviada' if ok else 'FALLÓ'})"


def run_symbol(sym, st, h1, d1):
    ctx = st["ctx"].setdefault(sym, {})
    px = float(h1.close.iloc[-1])
    obs = find_obs(to_weekly(d1))
    s_ob = check_ob(sym, px, obs, st, ctx)   # fija el contexto semanal
    s_div = check_div(sym, d1, st, ctx)      # fija el contexto diario
    s_x = check_cross(sym, h1, st, ctx)
    print(f"{label(sym)}: precio {px:.5g} | semanal OB: {s_ob} | diario RSI: {s_div} | horario cruce: {s_x}")
    for dirn, lo, hi, ts, mult in obs:
        print(f"      OB {NAME[dirn]} semana {ts:%Y-%m-%d} zona {lo:.5g}–{hi:.5g} ({mult:.1f} ATR)")


def scan(st):
    h1s = fetch_all(WATCHLIST, "1h")
    d1s = fetch_all(WATCHLIST, "1d")
    ok, nodata = 0, []
    for sym in WATCHLIST:
        h1, d1 = h1s.get(sym), d1s.get(sym)
        if h1 is None or d1 is None or len(h1) < 30 or len(d1) < 120:
            nodata.append(sym)
            continue
        try:
            run_symbol(sym, st, h1, d1)
            ok += 1
        except Exception as e:
            print(f"Error con {sym}: {e}")
            nodata.append(sym)
    print(f"\nRevisados {ok}/{len(WATCHLIST)} activos.")
    if nodata:
        print("Sin datos suficientes:", ", ".join(nodata))
    if NEAR:
        print(f"Cerca de un OB (<= {NEAR_PCT*100:.0f}%):",
              "; ".join(f"{label(s)} OB {NAME[d]} a {x*100:.1f}%" for s, d, x in sorted(NEAR, key=lambda t: t[2])))


def load_state():
    try:
        st = json.load(open(STATE_FILE))
    except Exception:
        st = {}
    for k in ("last", "ob_sent", "ctx"):
        if not isinstance(st.get(k), dict):
            st[k] = {}
    return st


def main():
    if not TOKEN or not CHAT_ID:
        print("ERROR: faltan los secretos TELEGRAM_TOKEN y/o TELEGRAM_CHAT_ID "
              "(Settings > Secrets and variables > Actions)")
        sys.exit(1)
    if "--test" in sys.argv:
        ok = send("✅ Bot de alertas funcionando. Si ves este mensaje, Telegram está bien configurado.")
        print("Mensaje de prueba enviado." if ok else "FALLÓ el mensaje de prueba.")
        sys.exit(0 if ok else 1)
    st = load_state()
    scan(st)
    json.dump(st, open(STATE_FILE, "w"))
    sys.exit(1 if SEND_FAILED else 0)


if __name__ == "__main__":
    main()
