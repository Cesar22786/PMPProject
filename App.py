import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# Configuración de la página
st.set_page_config(page_title="Análisis Avanzado de Portafolios", layout="wide", initial_sidebar_state="expanded")

# ====== FUNCIONES PRINCIPALES ====== #

# Guardar datos en CSV
def guardar_datos_csv(data, filename):
    data.to_csv(filename)
    st.success(f"Datos guardados en {filename}.")

# Descargar datos de Yahoo Finance
def descargar_datos(etfs, start_date, end_date):
    with st.spinner(f"Descargando datos desde {start_date} hasta {end_date}..."):
        try:
            data = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
            if data.empty:
                st.error("No se pudieron descargar datos para los símbolos ingresados. Verifique las fechas o los símbolos.")
            return data.ffill().dropna()
        except Exception as e:
            st.error(f"Error al descargar datos: {e}")
            return pd.DataFrame()

# Calcular métricas básicas de los activos
def calcular_metricas(data):
    rendimientos = data.pct_change().dropna()
    media = rendimientos.mean() * 252
    volatilidad = rendimientos.std() * np.sqrt(252)
    sharpe = media / volatilidad
    sortino = media / (rendimientos[rendimientos < 0].std() * np.sqrt(252))
    drawdown = (data / data.cummax() - 1).min()
    return rendimientos, media, volatilidad, sharpe, sortino, drawdown

# Optimizar portafolios
def optimizar_portafolio(rendimientos, weights, tasa_libre_riesgo=0.02):
    n = len(rendimientos.columns)
    mean_returns = rendimientos.mean() * 252
    cov_matrix = rendimientos.cov() * 252

    def port_metrics(weights):
        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - tasa_libre_riesgo) / port_volatility
        return -sharpe

    bounds = [(0, 1) for _ in range(n)]
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    result = minimize(port_metrics, weights, bounds=bounds, constraints=constraints)
    return result.x, mean_returns, cov_matrix

# Modelo Black-Litterman
def black_litterman(mean_returns, cov_matrix, market_weights, views, confidence):
    try:
        tau = 0.05
        pi = np.dot(cov_matrix, market_weights)

        Q = np.array(views).reshape(-1, 1)
        P = np.eye(len(market_weights))
        omega = np.diag(np.diag(np.dot(P, np.dot(tau * cov_matrix, P.T))) / confidence)

        M_inverse = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(P.T, np.dot(np.linalg.inv(omega), P)))
        BL_returns = M_inverse @ (np.linalg.inv(tau * cov_matrix) @ pi + P.T @ np.linalg.inv(omega) @ Q)
        return BL_returns.flatten()
    except Exception as e:
        st.error(f"Error en el modelo Black-Litterman: {e}")
        return []

# ====== INTERFAZ ====== #

# Entrada de parámetros del usuario
st.sidebar.header("Parámetros del Portafolio")
etfs_input = st.sidebar.text_input("Ingrese los ETFs separados por comas (por ejemplo: AGG,EMB,VTI,EEM,GLD):", "AGG,EMB,VTI,EEM,GLD")
etfs = [etf.strip() for etf in etfs_input.split(',')]

benchmarks_opciones = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "MSCI World": "URTH",
}
benchmark = st.sidebar.selectbox("Seleccione el Benchmark:", options=benchmarks_opciones.keys())
benchmark_symbol = benchmarks_opciones[benchmark]

start_date = st.sidebar.date_input("Fecha de inicio:", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("Fecha de fin:", pd.to_datetime("2023-01-01"))

weights_input = st.sidebar.text_input("Pesos iniciales (opcional):", ",".join(["0.2"] * len(etfs)))
weights = [float(w.strip()) for w in weights_input.split(",")] if weights_input else [1 / len(etfs)] * len(etfs)

# Descarga de datos
st.sidebar.header("Opciones de Descarga")
guardar_csv = st.sidebar.checkbox("Guardar datos descargados en CSV")

data = descargar_datos(etfs + [benchmark_symbol], start_date, end_date)
if data.empty:
    st.error("No se pudieron descargar los datos. Verifique las fechas o los símbolos ingresados.")
else:
    if guardar_csv:
        guardar_datos_csv(data, "portafolio_datos.csv")

    rendimientos, media, volatilidad, sharpe, sortino, drawdown = calcular_metricas(data)

    # ====== ANÁLISIS ====== #
    st.header("Análisis del Portafolio")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimiento Promedio Anualizado", f"{media.mean():.2%}")
    col2.metric("Volatilidad Promedio Anualizada", f"{volatilidad.mean():.2%}")
    col3.metric("Sharpe Ratio Promedio", f"{sharpe.mean():.2f}")

    st.subheader("Estadísticas Detalladas")
    stats_table = pd.DataFrame({
        "Rendimiento Anualizado": media,
        "Volatilidad Anualizada": volatilidad,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Drawdown": drawdown
    }).T
    st.dataframe(stats_table)

    st.subheader("Distribución de Retornos")
    for etf in etfs:
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=rendimientos[etf], nbinsx=50, name=f"{etf}"))
        fig_hist.update_layout(
            title=f"Distribución de Retornos para {etf}",
            xaxis_title="Retorno",
            yaxis_title="Frecuencia"
        )
        st.plotly_chart(fig_hist)

    # ====== OPTIMIZACIÓN ====== #
    st.header("Optimización del Portafolio")
    opt_weights, mean_returns, cov_matrix = optimizar_portafolio(rendimientos[etfs], weights)

    st.subheader("Pesos Óptimos del Portafolio")
    st.write(pd.DataFrame(opt_weights, index=etfs, columns=["Pesos Óptimos"]))

    # ====== BLACK-LITTERMAN ====== #
    st.header("Modelo Black-Litterman")
    market_weights = np.array([1 / len(etfs)] * len(etfs))
    views_input = st.text_input("Ingrese las vistas (rendimientos esperados por activo) separados por comas:", "0.03,0.04,0.05,0.02,0.01")
    confidence_input = st.slider("Nivel de Confianza en las Vistas (0-100):", 0, 100, 50)

    views = [float(v.strip()) for v in views_input.split(",")]

    if len(views) != len(etfs):
        st.error(f"El número de vistas ingresadas ({len(views)}) no coincide con el número de activos seleccionados ({len(etfs)}).")
    else:
        try:
            confidence = confidence_input / 100
            bl_returns = black_litterman(mean_returns[etfs], cov_matrix, market_weights, views, confidence)
            st.write("Rendimientos Ajustados por Black-Litterman:")
            st.dataframe(pd.DataFrame(bl_returns, index=etfs, columns=["Rendimientos"]))
        except Exception as e:
            st.error(f"Error en el modelo Black-Litterman: {e}")

    # ====== BACKTESTING ====== #
    st.header("Backtesting")
    port_returns = (rendimientos[etfs] * opt_weights).sum(axis=1).cumsum()
    benchmark_returns = rendimientos[benchmark_symbol].cumsum()

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=port_returns.index, y=port_returns, name="Portafolio"))
    fig_bt.add_trace(go.Scatter(x=benchmark_returns.index, y=benchmark_returns, name="Benchmark"))
    fig_bt.update_layout(title="Backtesting: Portafolio vs Benchmark", xaxis_title="Fecha", yaxis_title="Rendimientos Acumulados")
    st.plotly_chart(fig_bt)

 
   
