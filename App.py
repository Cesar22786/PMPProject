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

# ====== ESTILOS Y GIF ====== #
def agregar_fondo_y_gif():
    # Fondo y estilos
    fondo_html = """
    <style>
    body {
        background: linear-gradient(to right, #000000, #800000);
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background: linear-gradient(to bottom, #000000, #800000);
    }
    .stSidebar {
        background: linear-gradient(to bottom, #400000, #800000);
    }
    .metric-container .metric {
        background: #400000;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
    }
    .stMarkdown h1, h2, h3 {
        color: #FFD700;
    }
    .css-1q8dd3e p {
        color: #ffcccc !important;
    }
    .dataframe {
        color: #ffffff !important;
        background-color: #333333;
    }
    .gif-container {
        position: relative;
        margin-left: 20px;
        width: 150px;
    }
    </style>
    """
    st.markdown(fondo_html, unsafe_allow_html=True)

# Llamar a la función para agregar el fondo y el GIF
agregar_fondo_y_gif()

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

        if len(views) != len(market_weights):
            raise ValueError("El número de vistas no coincide con el número de activos seleccionados.")

        Q = np.array(views).reshape(-1, 1)
        P = np.eye(len(market_weights))

        omega = np.diag(np.diag(np.dot(P, np.dot(tau * cov_matrix, P.T))) / confidence)
        M_inverse = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(P.T, np.dot(np.linalg.inv(omega), P)))
        BL_returns = M_inverse @ (np.linalg.inv(tau * cov_matrix) @ pi + P.T @ np.linalg.inv(omega) @ Q)

        if BL_returns.shape[0] != len(market_weights):
            raise ValueError("El cálculo de Black-Litterman devolvió un tamaño inesperado.")
        return BL_returns.flatten()

    except Exception as e:
        st.error(f"Error en el modelo Black-Litterman: {e}")
        return []

# ====== Visualización de Métricas ====== #
st.title("📊 Análisis del Portafolio")
col1, col2 = st.columns([3, 1])

with col1:
    stats_table = pd.DataFrame({
        "Rendimiento Anualizado": media,
        "Volatilidad Anualizada": volatilidad,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Drawdown": drawdown
    }).T
    st.dataframe(stats_table.style.highlight_max(axis=1, color="lightgreen"))

with col2:
    st.markdown(
        """
        <div class="gif-container">
            <img src="https://i.gifer.com/1z4b.gif" alt="GIF" style="width:100%;border-radius:10px;">
        </div>
        """,
        unsafe_allow_html=True,
    )

# ====== Distribución de Retornos ====== #
st.subheader("Distribución de Retornos")
for etf in etfs:
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=rendimientos[etf], nbinsx=50, marker_color="blue", opacity=0.75))
    fig_hist.update_layout(
        title=f"Distribución de Retornos para {etf}",
        xaxis_title="Retorno",
        yaxis_title="Frecuencia",
        template="plotly_dark"
    )
    st.plotly_chart(fig_hist)

# ====== Optimización del Portafolio ====== #
st.header("🚀 Optimización del Portafolio")
opt_weights, mean_returns, cov_matrix = optimizar_portafolio(rendimientos[etfs], weights)
st.subheader("Pesos Óptimos del Portafolio")
st.bar_chart(pd.DataFrame(opt_weights, index=etfs, columns=["Pesos Óptimos"]))

# ====== Modelo Black-Litterman ====== #
st.header("🔮 Modelo Black-Litterman")
market_weights = np.array([1 / len(etfs)] * len(etfs))
views_input = st.text_input("Ingrese las vistas (rendimientos esperados por activo):", "0.03,0.04,0.05,0.02,0.01")
confidence_input = st.slider("Confianza en las vistas (0-100):", 0, 100, 50)

try:
    views = [float(v.strip()) for v in views_input.split(",")]
    confidence = confidence_input / 100

    if len(views) != len(etfs):
        st.warning("El número de vistas no coincide con el número de activos. Verifica las entradas.")
    else:
        bl_returns = black_litterman(mean_returns[etfs], cov_matrix, market_weights, views, confidence)
        if bl_returns:
            st.write("Retornos ajustados por Black-Litterman:")
            st.dataframe(pd.DataFrame(bl_returns, index=etfs, columns=["Rendimientos"]))
        else:
            st.warning("No se pudo calcular Black-Litterman correctamente.")
except Exception as e:
    st.error(f"Error: {e}")

# ====== Backtesting ====== #
st.header("📈 Backtesting")
port_returns = (rendimientos[etfs] * opt_weights).sum(axis=1).cumsum()
benchmark_returns = rendimientos[benchmark_symbol].cumsum()

fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(x=port_returns.index, y=port_returns, name="Portafolio", line=dict(color="cyan")))
fig_bt.add_trace(go.Scatter(x=benchmark_returns.index, y=benchmark_returns, name="Benchmark", line=dict(color="orange")))
fig_bt.update_layout(
    title="Backtesting: Portafolio vs Benchmark",
    xaxis_title="Fecha",
    yaxis_title="Rendimiento Acumulado",
    template="plotly_dark"
)
st.plotly_chart(fig_bt)

# ====== Leyenda ====== #
with st.expander("📘 Leyenda: Explicación de métricas y visualizaciones"):
    st.write("""
    ### Métricas y Visualizaciones
    - **Rendimiento Promedio Anualizado:** Representa el rendimiento promedio que un activo o portafolio podría generar en un año.
    - **Volatilidad Promedio Anualizada:** Mide el nivel de riesgo o variabilidad en los retornos anuales del activo o portafolio.
    - **Sharpe Ratio:** Indica el rendimiento ajustado al riesgo, comparando el rendimiento con la volatilidad. Un valor más alto es mejor.
    - **Sortino Ratio:** Similar al Sharpe Ratio, pero solo considera la volatilidad negativa (pérdidas).
    - **Drawdown:** La caída máxima desde un pico hasta un valle en el valor del portafolio.
    - **Distribución de Retornos:** Histograma que muestra la frecuencia de los retornos observados para cada activo.
    - **Optimización del Portafolio:** Cálculo de los pesos óptimos de los activos para maximizar el Sharpe Ratio o minimizar la volatilidad.
    - **Modelo Black-Litterman:** Ajusta los retornos esperados del mercado incorporando las opiniones de los inversores (vistas) y su nivel de confianza.
    - **Backtesting:** Compara el rendimiento acumulado del portafolio optimizado contra un benchmark seleccionado, mostrando resultados históricos.
    """)



