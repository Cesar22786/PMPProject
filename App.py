import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# Configuración de la página
st.set_page_config(
    page_title="Análisis Avanzado de Portafolios",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== ESTILOS ====== #
def agregar_fondo():
    fondo_html = """
    <style>
    body {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background: linear-gradient(to bottom, #0f2027, #2c5364);
    }
    .stSidebar {
        background: linear-gradient(to bottom, #203a43, #2c5364);
    }
    .metric-container .metric {
        background: #2c5364;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
    }
    .stMarkdown h1, h2, h3 {
        color: #FFD700;
    }
    </style>
    """
    st.markdown(fondo_html, unsafe_allow_html=True)

agregar_fondo()

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
st.sidebar.header("Parámetros del Portafolio")
etfs_input = st.sidebar.text_input("Ingrese los ETFs separados por comas:", "AGG,EMB,VTI,EEM,GLD")
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

# Pesos iniciales
weights_input = st.sidebar.text_input("Pesos iniciales (opcional):", ",".join(["0.2"] * len(etfs)))
weights = [float(w.strip()) for w in weights_input.split(",")] if weights_input else [1 / len(etfs)] * len(etfs)
guardar_csv = st.sidebar.checkbox("Guardar datos descargados en CSV")
data = descargar_datos(etfs + [benchmark_symbol], start_date, end_date)

if data.empty:
    st.error("No se pudieron descargar los datos. Verifique las fechas o los símbolos ingresados.")
else:
    if guardar_csv:
        guardar_datos_csv(data, "portafolio_datos.csv")

    rendimientos, media, volatilidad, sharpe, sortino, drawdown = calcular_metricas(data)

    st.title("📊 Análisis del Portafolio")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimiento Promedio", f"{media.mean():.2%}")
    col2.metric("Volatilidad Promedio", f"{volatilidad.mean():.2%}")
    col3.metric("Sharpe Ratio", f"{sharpe.mean():.2f}")

    st.subheader("Estadísticas Detalladas")
    stats_table = pd.DataFrame({
        "Rendimiento Anualizado": media,
        "Volatilidad Anualizada": volatilidad,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Drawdown": drawdown
    }).T
    st.dataframe(stats_table.style.highlight_max(axis=1, color="lightgreen"))

    st.header("📊 Distribución de Retornos")
    for etf in etfs:
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=rendimientos[etf],
                nbinsx=50,
                marker_color="#FFD700",
                opacity=0.8
            )
        )
        fig_hist.update_layout(
            title=f"Distribución de Retornos para {etf}",
            xaxis_title="Retorno",
            yaxis_title="Frecuencia",
            template="plotly_dark"
        )
        st.plotly_chart(fig_hist)

    st.header("🚀 Optimización del Portafolio")
    opt_weights, mean_returns, cov_matrix = optimizar_portafolio(rendimientos[etfs], weights)
    st.bar_chart(pd.DataFrame(opt_weights, index=etfs, columns=["Pesos Óptimos"]))

    st.header("🔮 Modelo Black-Litterman")
    market_weights = np.array([1 / len(etfs)] * len(etfs))
    views_input = st.text_input("Ingrese las vistas (rendimientos esperados por activo):", "0.03,0.04,0.05,0.02,0.01")
    confidence_input = st.slider("Confianza en las vistas (0-100):", 0, 100, 50)

    try:
        views = [float(v.strip()) for v in views_input.split(",")]
        confidence = confidence_input / 100
        bl_returns = black_litterman(mean_returns[etfs], cov_matrix, market_weights, views, confidence)
        st.write("Retornos ajustados por Black-Litterman:")
        st.dataframe(pd.DataFrame(bl_returns, index=etfs, columns=["Rendimientos"]))
    except Exception as e:
        st.error(f"Error: {e}")

    st.header("📈 Backtesting")
    port_returns = (rendimientos[etfs] * opt_weights).sum(axis=1).cumsum()
    benchmark_returns = rendimientos[benchmark_symbol].cumsum()

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=port_returns.index, y=port_returns, name="Portafolio", line=dict(color="cyan")))
    fig_bt.add_trace(go.Scatter(x=benchmark_returns.index, y=benchmark_returns, name="Benchmark", line=dict(color="orange")))
    fig_bt.update_layout(template="plotly_dark", title="Backtesting", xaxis_title="Fecha", yaxis_title="Rendimiento Acumulado")
    st.plotly_chart(fig_bt)

