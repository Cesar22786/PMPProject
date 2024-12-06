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
        position: absolute;
        top: 10px;
        right: 10px;
        width: 100px; /* Ajustamos el tamaño a un ancho más pequeño */
        z-index: 10;
    }
    </style>
    <div class="gif-container">
        <img src="https://art.pixilart.com/8ebf216d8b6c2f3.gif" alt="GIF" style="width:100%;border-radius:10px;box-shadow:0px 4px 10px rgba(0,0,0,0.5);">
    </div>
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
        tau = 0.05  # Parámetro de escala
        pi = np.dot(cov_matrix, market_weights)  # Retornos implícitos del mercado

        # Validación de vistas
        if len(views) != len(market_weights):
            raise ValueError("El número de vistas no coincide con el número de activos seleccionados.")

        Q = np.array(views).reshape(-1, 1)  # Vistas expresadas como matriz columna
        P = np.eye(len(market_weights))  # Matriz identidad (1 vista por activo)

        omega = np.diag(np.diag(np.dot(P, np.dot(tau * cov_matrix, P.T))) / confidence)  # Matriz de incertidumbre

        M_inverse = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(P.T, np.dot(np.linalg.inv(omega), P)))
        BL_returns = M_inverse @ (np.linalg.inv(tau * cov_matrix) @ pi + P.T @ np.linalg.inv(omega) @ Q)
        return BL_returns.flatten()

    except Exception as e:
        st.error(f"Error en el modelo Black-Litterman: {e}")
        return []

# Agregar la tabla con VaR y CVaR
    st.subheader("📉 Análisis de Riesgo: VaR y CVaR")
    alpha = 0.95
    var_cvar = {}
    for etf in etfs:
        var = rendimientos[etf].quantile(1 - alpha)
        cvar = rendimientos[etf][rendimientos[etf] <= var].mean()
        var_cvar[etf] = {"VaR (95%)": var, "CVaR (95%)": cvar}

    # Mostrar la tabla de VaR y CVaR
    var_cvar_df = pd.DataFrame(var_cvar).T
    st.dataframe(var_cvar_df.style.format({"VaR (95%)": "{:.2%}", "CVaR (95%)": "{:.2%}"}))

# ====== Entrada de Parámetros del Usuario ====== #
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

weights_input = st.sidebar.text_input("Pesos iniciales (opcional):", ",".join(["0.2"] * len(etfs)))
weights = [float(w.strip()) for w in weights_input.split(",")] if weights_input else [1 / len(etfs)] * len(etfs)

guardar_csv = st.sidebar.checkbox("Guardar datos descargados en CSV")

# ====== Descarga de Datos ====== #
data = descargar_datos(etfs + [benchmark_symbol], start_date, end_date)
if data.empty:
    st.error("No se pudieron descargar los datos. Verifique las fechas o los símbolos ingresados.")
else:
    if guardar_csv:
        guardar_datos_csv(data, "portafolio_datos.csv")

    rendimientos, media, volatilidad, sharpe, sortino, drawdown = calcular_metricas(data)

 # ====== Métricas del Portafolio ====== #
st.title("📊 Análisis del Portafolio")
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
st.dataframe(stats_table.style.highlight_max(axis=1, color="lightgreen"))

st.subheader("📉 Análisis de Riesgo: VaR y CVaR")
alpha = 0.95
var_cvar = {}
for etf in etfs:
    var = rendimientos[etf].quantile(1 - alpha)
    cvar = rendimientos[etf][rendimientos[etf] <= var].mean()
    var_cvar[etf] = {"VaR (95%)": var, "CVaR (95%)": cvar}

var_cvar_df = pd.DataFrame(var_cvar).T
st.dataframe(var_cvar_df.style.format({"VaR (95%)": "{:.2%}", "CVaR (95%)": "{:.2%}"}))

# ====== Distribución de Retornos ====== #
st.subheader("Distribución de Retornos")

# Crear un selector para elegir un ETF específico
selected_etf = st.selectbox("Seleccione un ETF para visualizar su distribución de retornos:", etfs)

# Gráfica de distribución de retornos para el ETF seleccionado
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=rendimientos[selected_etf],
    nbinsx=50,
    marker_color="goldenrod",  # Cambiamos el color a un dorado que combina con el diseño
    opacity=0.85
))
fig_hist.update_layout(
    title=f"Distribución de Retornos para {selected_etf}",
    xaxis_title="Retorno",
    yaxis_title="Frecuencia",
    template="plotly_dark",
    title_font=dict(size=20, color="gold"),  # Ajustamos el color del título a dorado
    font=dict(size=14, color="white")  # Cambiamos el color del texto para mejor visibilidad
)

# Mostrar la gráfica seleccionada
st.plotly_chart(fig_hist, use_container_width=True)

# ====== Optimización del Portafolio ====== #
st.header("🚀 Optimización del Portafolio")
opt_weights, mean_returns, cov_matrix = optimizar_portafolio(rendimientos[etfs], weights)
st.subheader("Pesos Óptimos del Portafolio")

# Crear el gráfico con el color "goldenrod"
fig_opt_weights = go.Figure()
fig_opt_weights.add_trace(go.Bar(
    x=etfs,
    y=opt_weights,
    marker_color="goldenrod",  # Color dorado uniforme con el gráfico de distribución de retornos
    opacity=0.85
))
fig_opt_weights.update_layout(
    title="Pesos Óptimos del Portafolio",
    xaxis_title="Activos",
    yaxis_title="Peso",
    template="plotly_dark",
    title_font=dict(size=20, color="gold"),  # Ajustamos el color del título a dorado
    font=dict(size=14, color="white")  # Cambiamos el color del texto para mayor consistencia
)

# Mostrar el gráfico
st.plotly_chart(fig_opt_weights, use_container_width=True)

# ====== Modelo Black-Litterman ====== #
st.header("🔮 Modelo Black-Litterman")
market_weights = np.array([1 / len(etfs)] * len(etfs))
views_input = st.text_input("Ingrese las vistas (rendimientos esperados por activo):", "0.03,0.04,0.05,0.02,0.01")
confidence_input = st.slider("Confianza en las vistas (0-100):", 0, 100, 50)

try:
    views = np.array([float(v.strip()) for v in views_input.split(",")])
    confidence = confidence_input / 100

    if len(views) != len(etfs):
        st.warning("El número de vistas no coincide con el número de activos. Verifica las entradas.")
    else:
        tau = 0.05
        P = np.eye(len(etfs))
        omega = np.diag(np.diag(np.dot(P, np.dot(tau * cov_matrix, P.T))) / confidence)

        pi = np.dot(cov_matrix, market_weights)

        # Cálculo de Black-Litterman
        M_inverse = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + np.dot(P.T, np.dot(np.linalg.inv(omega), P)))
        BL_returns = M_inverse @ (np.linalg.inv(tau * cov_matrix) @ pi + P.T @ np.linalg.inv(omega) @ views)

        # Mostrar resultados ajustados
        st.subheader("📈 Retornos Ajustados por Black-Litterman")
        adjusted_df = pd.DataFrame(BL_returns, index=etfs, columns=["Retorno Ajustado"])
        st.dataframe(adjusted_df.style.format("{:.2%}").background_gradient(cmap="Greens"))
except Exception as e:
    st.error(f"Error en Black-Litterman: {e}")

# ====== Backtesting ====== #
st.header("📈 Backtesting")

# Calcular rendimientos acumulados del portafolio y el benchmark
try:
    port_returns = (rendimientos[etfs] * opt_weights).sum(axis=1)
    port_cumulative_returns = (1 + port_returns).cumprod() - 1

    benchmark_returns = rendimientos[benchmark_symbol]
    benchmark_cumulative_returns = (1 + benchmark_returns).cumprod() - 1

    # Mostrar métricas clave del backtesting
    st.subheader("📊 Métricas del Backtesting")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimiento Total del Portafolio", f"{port_cumulative_returns.iloc[-1]:.2%}")
    col2.metric("Rendimiento Total del Benchmark", f"{benchmark_cumulative_returns.iloc[-1]:.2%}")
    col3.metric("Diferencia de Rendimiento", f"{(port_cumulative_returns.iloc[-1] - benchmark_cumulative_returns.iloc[-1]):.2%}")

    # Visualización del rendimiento acumulado
    st.subheader("📈 Rendimientos Acumulados: Portafolio vs Benchmark")
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=port_cumulative_returns.index,
        y=port_cumulative_returns,
        name="Portafolio",
        line=dict(color="cyan", width=3)
    ))
    fig_bt.add_trace(go.Scatter(
        x=benchmark_cumulative_returns.index,
        y=benchmark_cumulative_returns,
        name=f"Benchmark ({benchmark})",
        line=dict(color="orange", width=3)
    ))
    fig_bt.update_layout(
        title="Rendimientos Acumulados: Portafolio vs Benchmark",
        xaxis_title="Fecha",
        yaxis_title="Rendimiento Acumulado",
        template="plotly_dark",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_bt, use_container_width=True)

except Exception as e:
    st.error(f"Error al calcular o mostrar el backtesting: {e}")

# ====== Leyenda ====== #
with st.expander("📘 Leyenda: Explicación de métricas y visualizaciones"):
    st.write("""
    ### Métricas y Visualizaciones
        - *Rendimiento Promedio Anualizado:* Representa el rendimiento promedio que un activo o portafolio podría generar en un año.
        - *Volatilidad Promedio Anualizada:* Mide el nivel de riesgo o variabilidad en los retornos anuales del activo o portafolio.
        - *Sharpe Ratio:* Indica el rendimiento ajustado al riesgo, comparando el rendimiento con la volatilidad. Un valor más alto es mejor.
        - *Sortino Ratio:* Similar al Sharpe Ratio, pero solo considera la volatilidad negativa (pérdidas).
        - *Drawdown:* La caída máxima desde un pico hasta un valle en el valor del portafolio.
        - *VaR (Valor en Riesgo):* Pérdida máxima esperada con un nivel de confianza del 95%.
        - *CVaR (Valor en Riesgo Condicional):* Pérdida promedio esperada en el peor 5% de los casos.
        - *Distribución de Retornos:* Histograma que muestra la frecuencia de los retornos observados para cada activo.
        - *Optimización del Portafolio:* Cálculo de los pesos óptimos de los activos para maximizar el Sharpe Ratio o minimizar la volatilidad.
        - *Modelo Black-Litterman:* Ajusta los retornos esperados del mercado incorporando las opiniones de los inversores (vistas) y su nivel de confianza.
        - *Backtesting:* Compara el rendimiento acumulado del portafolio optimizado contra un benchmark seleccionado, mostrando resultados históricos.
        """)












