import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# Configuración de la página
st.set_page_config(page_title="Análisis Avanzado de Portafolios", layout="wide")
st.sidebar.title("Configuración de Análisis")

# ====== FUNCIONES PRINCIPALES ====== #

# Función para descargar datos históricos
def descargar_datos(etfs, start_date, end_date):
    data = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
    return data.ffill().dropna()

# Calcular métricas básicas de los activos
def calcular_metricas(data):
    rendimientos = data.pct_change().dropna()
    media = rendimientos.mean() * 252
    volatilidad = rendimientos.std() * np.sqrt(252)
    sharpe = media / volatilidad
    sortino = media / (rendimientos[rendimientos < 0].std() * np.sqrt(252))
    drawdown = (data / data.cummax() - 1).min()
    return rendimientos, media, volatilidad, sharpe, sortino, drawdown

# Cálculo de VaR y CVaR
def calcular_var_cvar(rendimientos, alpha=0.95):
    var = rendimientos.quantile(1 - alpha)
    cvar = rendimientos[rendimientos <= var].mean()
    return var, cvar

# Función para optimizar portafolios
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

    if objetivo == "Sharpe":
        objetivo_func = sharpe_ratio
    else:
        objetivo_func = min_volatility

    bounds = [(0, 1) for _ in range(n)]
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    result = minimize(objetivo_func, [1 / n] * n, bounds=bounds, constraints=constraints)
    return result.x, mean_returns, cov_matrix

# Backtesting de portafolios
def backtesting(port_weights, rendimientos, benchmark):
    if benchmark not in rendimientos.columns:
        raise KeyError(f"El benchmark '{benchmark}' no está en los datos.")
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

# Entrada de parámetros del usuario
st.sidebar.header("Parámetros de los ETFs")
etfs_input = st.sidebar.text_input("Ingrese los ETFs separados por comas (por ejemplo: AGG,EMB,VTI,EEM,GLD):", "AGG,EMB,VTI,EEM,GLD")
etfs = [etf.strip() for etf in etfs_input.split(',')]

# Opciones de benchmarks
benchmarks_opciones = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "MSCI World": "URTH",
}
benchmark = st.sidebar.selectbox("Seleccione el Benchmark:", options=benchmarks_opciones.keys())
benchmark_symbol = benchmarks_opciones[benchmark]

# Fechas de análisis
start_date = st.sidebar.date_input("Fecha de inicio:", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("Fecha de fin:", pd.to_datetime("2023-01-01"))

# Descargar datos
data = descargar_datos(etfs + [benchmark_symbol], start_date, end_date)
rendimientos, media, volatilidad, sharpe, sortino, drawdown = calcular_metricas(data)

# ====== ANÁLISIS DE ACTIVOS ====== #

st.header("Análisis de Activos")
st.subheader("Estadísticas de los ETFs")
stats_table = pd.DataFrame({
    "Rendimiento Anualizado": media,
    "Volatilidad Anualizada": volatilidad,
    "Sharpe Ratio": sharpe,
    "Sortino Ratio": sortino,
    "Drawdown": drawdown
}).T
st.dataframe(stats_table)

# Gráfico de rendimiento acumulado por activo
st.subheader("Rendimiento Acumulado por Activo")
fig = go.Figure()
for etf in etfs:
    fig.add_trace(go.Scatter(x=data.index, y=(1 + rendimientos[etf]).cumprod() - 1, mode="lines", name=etf))
fig.update_layout(title="Rendimiento Acumulado por Activo", xaxis_title="Fecha", yaxis_title="Rendimiento Acumulado")
st.plotly_chart(fig)

# Correlación entre activos
st.subheader("Correlación entre Activos")
correlation_matrix = rendimientos.corr()
st.write("Matriz de Correlación")
st.dataframe(correlation_matrix)

# Heatmap de correlaciones
st.subheader("Heatmap de Correlación")
fig_corr = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    colorscale="Viridis"))
fig_corr.update_layout(title="Mapa de Calor de Correlaciones", xaxis_title="Activos", yaxis_title="Activos")
st.plotly_chart(fig_corr)

# ====== OPTIMIZACIÓN DE PORTAFOLIOS ====== #

st.header("Optimización de Portafolios")
objetivo = st.sidebar.selectbox("Seleccione el objetivo:", ["Sharpe", "Volatilidad Mínima"])
port_weights, mean_returns, cov_matrix = optimizar_portafolio(rendimientos[etfs], objetivo)

st.write(f"Pesos del Portafolio ({objetivo}):", pd.DataFrame(port_weights, index=etfs, columns=["Pesos"]))

# ====== BACKTESTING ====== #

st.header("Backtesting")

# Validar si el benchmark está en los datos
if benchmark_symbol not in rendimientos.columns:
    st.error(f"El benchmark '{benchmark}' no se encuentra en los datos descargados.")
else:
    try:
        # Alinear los pesos con los ETFs, excluyendo el benchmark
        rendimientos_sin_benchmark = rendimientos[etfs]
        port_returns, benchmark_returns = backtesting(port_weights, rendimientos_sin_benchmark, benchmark_symbol)

        # Verificar si los datos tienen suficientes registros
        if port_returns.empty or benchmark_returns.empty:
            st.warning("No hay suficientes datos para realizar el Backtesting.")
        else:
            # Crear gráfico de Backtesting
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=port_returns.index, y=port_returns, name="Portafolio"))
            fig_bt.add_trace(go.Scatter(x=benchmark_returns.index, y=benchmark_returns, name="Benchmark"))
            fig_bt.update_layout(
                title="Backtesting: Portafolio vs Benchmark",
                xaxis_title="Fecha",
                yaxis_title="Rendimientos Acumulados"
            )
            st.plotly_chart(fig_bt)
    except Exception as e:
        st.error(f"Ocurrió un error durante el Backtesting: {e}")

# ====== BLACK-LITTERMAN ====== #

st.header("Modelo Black-Litterman")
market_weights = np.array([1 / len(etfs)] * len(etfs))
views_input = st.text_input("Ingrese las vistas (rendimientos esperados por activo) separados por comas:", "0.03,0.04,0.05,0.02,0.01")
confidence_input = st.slider("Nivel de Confianza en las Vistas (0-100):", 0, 100, 50)

# Validar longitud de las vistas
views = [float(x.strip()) for x in views_input.split(',')]
if len(views) != len(etfs):
    st.error(f"El número de vistas ({len(views)}) no coincide con el número de activos seleccionados ({len(etfs)}).")
else:
    confidence = confidence_input / 100

    try:
        # Calcular retornos ajustados por Black-Litterman
        bl_returns = black_litterman(mean_returns[etfs], cov_matrix, market_weights, views, confidence)
        st.write("Rendimientos Ajustados por Black-Litterman:")
        st.dataframe(pd.DataFrame(bl_returns, index=etfs, columns=["Rendimientos"]))
    except Exception as e:
        st.error(f"Ocurrió un error durante el cálculo del modelo Black-Litterman: {e}")

st.header("Backtesting")

# Validar si el benchmark está en los datos
if benchmark_symbol not in rendimientos.columns:
    st.error(f"El benchmark '{benchmark}' no se encuentra en los datos descargados.")
else:
    try:
        # Ejecutar Backtesting
        port_returns, benchmark_returns = backtesting(port_weights, rendimientos, benchmark_symbol)

        # Verificar si los datos tienen suficientes registros
        if port_returns.empty or benchmark_returns.empty:
            st.warning("No hay suficientes datos para realizar el Backtesting.")
        else:
            # Crear gráfico de Backtesting
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=port_returns.index, y=port_returns, name="Portafolio"))
            fig_bt.add_trace(go.Scatter(x=benchmark_returns.index, y=benchmark_returns, name="Benchmark"))
            fig_bt.update_layout(
                title="Backtesting: Portafolio vs Benchmark",
                xaxis_title="Fecha",
                yaxis_title="Rendimientos Acumulados"
            )
            st.plotly_chart(fig_bt)
    except Exception as e:
        st.error(f"Ocurrió un error durante el Backtesting: {e}")

# ====== BLACK-LITTERMAN ====== #

st.header("Modelo Black-Litterman")
market_weights = np.array([1 / len(etfs)] * len(etfs))
views_input = st.text_input("Ingrese las vistas (rendimientos esperados por activo) separados por comas:", "0.03,0.04,0.05,0.02,0.01")
confidence_input = st.slider("Nivel de Confianza en las Vistas (0-100):", 0, 100, 50)

views = [float(x.strip()) for x in views_input.split(',')]
confidence = confidence_input / 100

bl_returns = black_litterman(mean_returns[etfs], cov_matrix, market_weights, views, confidence)
st.write("Rendimientos Ajustados por Black-Litterman:", pd.DataFrame(bl_returns, index=etfs, columns=["Rendimientos"]))

# ====== GRÁFICOS DE DISTRIBUCIONES ====== #

st.header("Distribuciones de Retornos")
for etf in etfs:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rendimientos[etf], nbinsx=50, name=f"{etf}"))
    fig.update_layout(title=f"Distribución de Retornos para {etf}", xaxis_title="Retorno", yaxis_title="Frecuencia")
    st.plotly_chart(fig)

# ====== CONCLUSIONES ====== #

st.header("Conclusión")
st.write("Con base en los análisis realizados, puedes evaluar cuál portafolio o enfoque (Black-Litterman, optimización tradicional) maximiza tus objetivos financieros.")
