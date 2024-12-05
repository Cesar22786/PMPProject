import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# Configuración de la página
st.set_page_config(page_title="Portafolios Óptimos y Black-Litterman", layout="wide")
st.sidebar.title("Configuración de Análisis")

# Función para descargar datos
def descargar_datos(etfs, start_date, end_date):
    data = yf.download(etfs, start=start_date, end=end_date)['Adj Close']
    return data.ffill().dropna()

# Función para calcular métricas
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

    if objetivo == "Sharpe":
        objetivo_func = sharpe_ratio
    else:
        objetivo_func = min_volatility

    bounds = [(0, 1) for _ in range(n)]
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    result = minimize(objetivo_func, [1 / n] * n, bounds=bounds, constraints=constraints)
    return result.x, mean_returns, cov_matrix

# Backtesting
def backtesting(port_weights, rendimientos, benchmark):
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

# Entradas
etfs_input = st.sidebar.text_input("Ingrese los ETFs separados por comas (por ejemplo: AGG,EMB,VTI,EEM,GLD):", "AGG,EMB,VTI,EEM,GLD")
etfs = [etf.strip() for etf in etfs_input.split(',')]

benchmark = st.sidebar.text_input("Ingrese el Benchmark:", "^GSPC")

start_date = st.sidebar.date_input("Fecha de inicio:", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("Fecha de fin:", pd.to_datetime("2023-01-01"))

# Descargar datos
data = descargar_datos(etfs + [benchmark], start_date, end_date)
rendimientos, media, volatilidad, sharpe, sortino, drawdown = calcular_metricas(data)

# Visualización de métricas
st.header("Estadísticas de Activos")
st.dataframe(pd.DataFrame({
    "Rendimiento Anualizado": media,
    "Volatilidad Anualizada": volatilidad,
    "Sharpe Ratio": sharpe,
    "Sortino Ratio": sortino,
    "Drawdown": drawdown
}).T)

# Optimización de portafolios
st.header("Portafolios Óptimos")
objetivo = st.sidebar.selectbox("Seleccione el objetivo:", ["Sharpe", "Volatilidad Mínima"])
port_weights, mean_returns, cov_matrix = optimizar_portafolio(rendimientos[etfs], objetivo)

st.write(f"Pesos del Portafolio ({objetivo}):", pd.DataFrame(port_weights, index=etfs, columns=["Pesos"]))

# Backtesting
st.header("Backtesting")
port_returns, benchmark_returns = backtesting(port_weights, rendimientos[etfs], benchmark)

fig = go.Figure()
fig.add_trace(go.Scatter(x=port_returns.index, y=port_returns, name="Portafolio"))
fig.add_trace(go.Scatter(x=benchmark_returns.index, y=benchmark_returns, name="Benchmark"))
fig.update_layout(title="Backtesting: Portafolio vs Benchmark", xaxis_title="Fecha", yaxis_title="Rendimientos Acumulados")
st.plotly_chart(fig)

# Black-Litterman Model
st.header("Modelo Black-Litterman")
market_weights = np.array([1 / len(etfs)] * len(etfs))
views_input = st.text_input("Ingrese las vistas (rendimientos esperados por activo) separados por comas:", "0.03,0.04,0.05,0.02,0.01")
confidence_input = st.slider("Nivel de Confianza en las Vistas (0-100):", 0, 100, 50)

views = [float(x.strip()) for x in views_input.split(',')]
confidence = confidence_input / 100

bl_returns = black_litterman(mean_returns[etfs], cov_matrix, market_weights, views, confidence)
st.write("Rendimientos Ajustados por Black-Litterman:", pd.DataFrame(bl_returns, index=etfs, columns=["Rendimientos"]))

# Gráficos adicionales
st.header("Distribuciones de Retornos")
for etf in etfs:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=rendimientos[etf], nbinsx=50, name=f"{etf}"))
    fig.update_layout(title=f"Distribución de Retornos para {etf}", xaxis_title="Retorno", yaxis_title="Frecuencia")
    st.plotly_chart(fig)

# Comparativa de métricas por ventana de tiempo
st.header("Análisis por Ventana de Tiempo")
windows = [30, 90, 180, 252, 756]
resultados = {}

for window in windows:
    resultados[window] = {
        "Rendimiento Promedio": rendimientos.tail(window).mean() * 252,
        "Volatilidad": rendimientos.tail(window).std() * np.sqrt(252),
        "VaR": calcular_var_cvar(rendimientos.tail(window))[0],
        "CVaR": calcular_var_cvar(rendimientos.tail(window))[1]
    }

st.write(pd.DataFrame(resultados).T)

# Conclusión final
st.header("Conclusión")
st.write("Con base en los análisis realizados, puedes evaluar cuál portafolio o enfoque (Black-Litterman, optimización tradicional) maximiza tus objetivos financieros.")
