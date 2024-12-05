import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# ====== CONFIGURACIÓN DE LA PÁGINA ====== #
st.set_page_config(page_title="Análisis Avanzado de Portafolios", layout="wide")
st.sidebar.title("Configuración de Análisis")

# ====== FUNCIONES PRINCIPALES ====== #

# Descargar datos históricos
def descargar_datos(etfs, start_date, end_date):
    try:
        data = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
        return data.ffill().dropna()
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        return pd.DataFrame()

# Calcular métricas por activo
def calcular_metricas(data):
    rendimientos = data.pct_change().dropna()
    media = rendimientos.mean() * 252
    volatilidad = rendimientos.std() * np.sqrt(252)
    sharpe = media / volatilidad
    sortino = media / (rendimientos[rendimientos < 0].std() * np.sqrt(252))
    drawdown = (data / data.cummax() - 1).min()
    return rendimientos, media, volatilidad, sharpe, sortino, drawdown

# VaR y CVaR
def calcular_var_cvar(rendimientos, alpha=0.95):
    var = rendimientos.quantile(1 - alpha)
    cvar = rendimientos[rendimientos <= var].mean()
    return var, cvar

# Optimización de portafolios
def optimizar_portafolio(rendimientos, objetivo, tasa_libre_riesgo=0.02):
    n = len(rendimientos.columns)
    mean_returns = rendimientos.mean() * 252
    cov_matrix = rendimientos.cov() * 252

    def sharpe_ratio(weights):
        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(port_return - tasa_libre_riesgo) / port_volatility

    def min_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    objetivo_func = sharpe_ratio if objetivo == "Sharpe" else min_volatility
    bounds = [(0, 1) for _ in range(n)]
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    result = minimize(objetivo_func, [1 / n] * n, bounds=bounds, constraints=constraints)
    return result.x, mean_returns, cov_matrix

# Backtesting con validaciones
def backtesting(port_weights, rendimientos, benchmark):
    if benchmark not in rendimientos.columns:
        st.error(f"El benchmark '{benchmark}' no se encuentra en los datos disponibles.")
        return pd.Series(dtype=float), pd.Series(dtype=float)
    portfolio_returns = (rendimientos * port_weights).sum(axis=1)
    benchmark_returns = rendimientos[benchmark]
    return portfolio_returns.cumsum(), benchmark_returns.cumsum()

# Black-Litterman Model
def black_litterman(mean_returns, cov_matrix, market_weights, views, confidence):
    tau = 0.05
    pi = np.dot(cov_matrix, market_weights)

    Q = np.array(views).reshape(-1, 1)
    P = np.eye(len(market_weights))
    omega = np.diag(np.diag(np.dot(P, np.dot(tau * cov_matrix, P.T))) / confidence)

    M_inverse = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(P.T, np.dot(np.linalg.inv(omega), P)))
    BL_returns = M_inverse @ (np.linalg.inv(tau * cov_matrix) @ pi + P.T @ np.linalg.inv(omega) @ Q)
    return BL_returns.flatten()

# ====== INTERFAZ ====== #

# Entradas desde el sidebar
st.sidebar.header("Parámetros de Entrada")
etfs_input = st.sidebar.text_input("Ingrese los ETFs separados por comas:", "AGG,EMB,VTI,EEM,GLD")
etfs = [etf.strip() for etf in etfs_input.split(',')]

benchmark = st.sidebar.text_input("Ingrese el Benchmark:", "^GSPC")

start_date = st.sidebar.date_input("Fecha de inicio:", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("Fecha de fin:", pd.to_datetime("2023-01-01"))

# Descargar datos
data = descargar_datos(etfs + [benchmark], start_date, end_date)

if not data.empty:
    # Calcular métricas
    rendimientos, media, volatilidad, sharpe, sortino, drawdown = calcular_metricas(data)

    # ====== ANÁLISIS DE ACTIVOS ====== #
    st.header("Análisis de Activos")
    st.subheader("Métricas de los ETFs")
    stats_table = pd.DataFrame({
        "Rendimiento Anualizado": media,
        "Volatilidad Anualizada": volatilidad,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Drawdown": drawdown
    }).T
    st.dataframe(stats_table)

    # ====== OPTIMIZACIÓN DE PORTAFOLIOS ====== #
    st.header("Optimización de Portafolios")
    objetivo = st.sidebar.selectbox("Seleccione el objetivo:", ["Sharpe", "Volatilidad Mínima"])
    port_weights, mean_returns, cov_matrix = optimizar_portafolio(rendimientos[etfs], objetivo)

    st.write(f"Pesos del Portafolio ({objetivo}):")
    st.dataframe(pd.DataFrame(port_weights, index=etfs, columns=["Pesos"]))

    # ====== BACKTESTING ====== #
    st.header("Backtesting")
    port_returns, benchmark_returns = backtesting(port_weights, rendimientos, benchmark)

    if not port_returns.empty and not benchmark_returns.empty:
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=port_returns.index, y=port_returns, name="Portafolio"))
        fig_bt.add_trace(go.Scatter(x=benchmark_returns.index, y=benchmark_returns, name="Benchmark"))
        fig_bt.update_layout(title="Backtesting: Portafolio vs Benchmark",
                             xaxis_title="Fecha", yaxis_title="Rendimientos Acumulados")
        st.plotly_chart(fig_bt)

    # ====== BLACK-LITTERMAN ====== #
    st.header("Modelo Black-Litterman")
    market_weights = np.array([1 / len(etfs)] * len(etfs))
    views_input = st.text_input("Ingrese vistas esperadas para cada activo:", "0.03,0.04,0.05,0.02,0.01")
    confidence_input = st.slider("Nivel de confianza en las vistas (0-100):", 0, 100, 50)

    views = [float(x.strip()) for x in views_input.split(',')]
    confidence = confidence_input / 100

    if len(views) == len(etfs):
        bl_returns = black_litterman(mean_returns[etfs], cov_matrix, market_weights, views, confidence)
        st.write("Rendimientos ajustados por Black-Litterman:")
        st.dataframe(pd.DataFrame(bl_returns, index=etfs, columns=["Rendimientos"]))

    # ====== GRÁFICOS ====== #
    st.header("Gráficos Adicionales")
    st.subheader("Rendimiento Acumulado por Activo")
    fig = go.Figure()
    for etf in etfs:
        fig.add_trace(go.Scatter(x=data.index, y=(1 + rendimientos[etf]).cumprod() - 1, mode="lines", name=etf))
    fig.update_layout(title="Rendimiento Acumulado por Activo", xaxis_title="Fecha", yaxis_title="Rendimiento Acumulado")
    st.plotly_chart(fig)

    st.subheader("Distribución de Retornos")
    for etf in etfs:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=rendimientos[etf], nbinsx=50))
        fig_hist.update_layout(title=f"Distribución de Retornos: {etf}", xaxis_title="Retorno", yaxis_title="Frecuencia")
        st.plotly_chart(fig_hist)

# Mensaje si no hay datos
else:
    st.error("No se pudieron descargar datos. Verifica los ETFs y el benchmark ingresados.")

