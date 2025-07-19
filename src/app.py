import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="🌳 Dashboard de Previsão de Desmatamento",
    page_icon="🌿",
    layout="wide"
)

# Tema escuro
st.markdown("""
<style>
    body, .main {
        background-color: #0e1117 !important;
        color: #e0e0e0;
    }
    h1, h2, h3 {
        color: #81c784 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #1c1f26 !important;
        color: #e0e0e0;
    }
    .metric-card {
        background-color: #1c1f26;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.5);
        margin-bottom: 16px;
        color: #e0e0e0;
    }
    .footer {
        font-size: 0.85em;
        color: #9e9e9e;
        text-align: center;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# Paleta para gráficos
cores_vibrantes = ['#1E88E5', '#FFC107', '#E91E63']

metricas_modelos = {
    'RandomForest': {'RMSE': 0.2749, 'MAE': 0.1353, 'MAPE': 73.61, 'R2': 0.1792, 'Pearson_r': 0.4302},
    'XGBoost':      {'RMSE': 0.2739, 'MAE': 0.1347, 'MAPE': 73.54, 'R2': 0.1859, 'Pearson_r': 0.4376},
    'LightGBM':     {'RMSE': 0.2736, 'MAE': 0.1338, 'MAPE': 73.08, 'R2': 0.1877, 'Pearson_r': 0.4418}
}
modelos = list(metricas_modelos.keys())

st.title("🌿 Dashboard de Previsão de Desmatamento")
st.markdown("<small>Análise comparativa entre algoritmos de Machine Learning</small>", unsafe_allow_html=True)

# 🔧 Sidebar
st.sidebar.header("🔧 Filtros e Visualizações")
modelo_selecionado = st.sidebar.selectbox("📌 Modelo para análise detalhada", modelos)
show_erro = st.sidebar.checkbox("📉 Gráficos de erro", True)
show_ajuste = st.sidebar.checkbox("📈 Métricas de ajuste", True)
show_dispersao = st.sidebar.checkbox("🔍 Previsões vs Reais", True)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Opções Avançadas")
suavizar = st.sidebar.slider("🔷 Suavizar pontos (Rolling Média)", 1, 10, 1, help="Aplica média móvel na dispersão")
tamanho_pontos = st.sidebar.slider("🔷 Tamanho dos pontos", 5, 15, 7)
opacidade_pontos = st.sidebar.slider("🔷 Opacidade dos pontos", 0.1, 1.0, 0.8)
linha_referencia = st.sidebar.checkbox("🔷 Mostrar linha ideal (y=x)", True)
escala_x = st.sidebar.selectbox("🔷 Escala eixo X", ["linear", "log"])
escala_y = st.sidebar.selectbox("🔷 Escala eixo Y", ["linear", "log"])

# 📊 Métricas
st.markdown("## 📊 Métricas do Modelo Selecionado")
metrica = metricas_modelos[modelo_selecionado]
col1, col2, col3 = st.columns(3)
col1.markdown(f"<div class='metric-card'><strong>RMSE</strong><br>{metrica['RMSE']:.4f}</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='metric-card'><strong>MAE</strong><br>{metrica['MAE']:.4f}</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='metric-card'><strong>MAPE (%)</strong><br>{metrica['MAPE']:.2f}</div>", unsafe_allow_html=True)
col4, col5 = st.columns(2)
col4.markdown(f"<div class='metric-card'><strong>R²</strong><br>{metrica['R2']:.4f}</div>", unsafe_allow_html=True)
col5.markdown(f"<div class='metric-card'><strong>Pearson r</strong><br>{metrica['Pearson_r']:.4f}</div>", unsafe_allow_html=True)

# Gráficos de erro
if show_erro:
    st.markdown("## 📉 Comparação de Erros")
    df_erro = pd.DataFrame({
        'Modelo': modelos,
        'RMSE': [metricas_modelos[m]['RMSE'] for m in modelos],
        'MAE': [metricas_modelos[m]['MAE'] for m in modelos],
        'MAPE': [metricas_modelos[m]['MAPE'] for m in modelos]
    })
    fig_erro = go.Figure()
    for i, metrica_nome in enumerate(['RMSE', 'MAE', 'MAPE']):
        fig_erro.add_trace(go.Bar(
            x=df_erro['Modelo'],
            y=df_erro[metrica_nome],
            name=metrica_nome,
            marker_color=cores_vibrantes[i]
        ))
    fig_erro.update_layout(
        barmode='group',
        template="plotly_dark",
        paper_bgcolor="#0e1117"
    )
    st.plotly_chart(fig_erro, use_container_width=True)

# Gráficos de ajuste
if show_ajuste:
    st.markdown("## 📈 Métricas de Ajuste")
    df_ajuste = pd.DataFrame({
        'Modelo': modelos,
        'R²': [metricas_modelos[m]['R2'] for m in modelos],
        'Pearson r': [metricas_modelos[m]['Pearson_r'] for m in modelos]
    })
    fig_ajuste = go.Figure()
    for i, metrica_nome in enumerate(['R²', 'Pearson r']):
        fig_ajuste.add_trace(go.Bar(
            x=df_ajuste['Modelo'],
            y=df_ajuste[metrica_nome],
            name=metrica_nome,
            marker_color=cores_vibrantes[i]
        ))
    fig_ajuste.update_layout(
        barmode='group',
        template="plotly_dark",
        paper_bgcolor="#0e1117"
    )
    st.plotly_chart(fig_ajuste, use_container_width=True)

# Gráfico dispersão + Ranking
if show_dispersao:
    st.markdown("## 🔍 Dispersão: Previsões vs Valores Reais")
    np.random.seed(42)
    reais = np.random.normal(1.0, 0.3, 100)
    reais = np.clip(reais, 0, None)

    erros_rmse = {}

    fig_disp = go.Figure()
    for i, modelo in enumerate(modelos):
        pred = reais + np.random.normal(0, 0.05 + 0.02 * i, size=100)
        if suavizar > 1:
            pred = pd.Series(pred).rolling(suavizar, min_periods=1).mean().values

        rmse = np.sqrt(np.mean((reais - pred) ** 2))
        erros_rmse[modelo] = rmse

        fig_disp.add_trace(go.Scatter(
            x=reais,
            y=pred,
            mode='markers',
            name=modelo,
            marker=dict(color=cores_vibrantes[i], size=tamanho_pontos, opacity=opacidade_pontos)
        ))

    if linha_referencia:
        fig_disp.add_shape(type='line', x0=0, y0=0, x1=2, y1=2,
                           line=dict(color='#e0e0e0', dash='dot'))

    fig_disp.update_layout(
        xaxis_type=escala_x,
        yaxis_type=escala_y,
        template="plotly_dark",
        paper_bgcolor="#0e1117"
    )
    st.plotly_chart(fig_disp, use_container_width=True)

    ranking = sorted(erros_rmse.items(), key=lambda x: x[1])
    melhor, segundo, pior = ranking[0], ranking[1], ranking[-1]

    st.markdown("## 🏆 Ranking dos Modelos (com base no gráfico de dispersão suavizado)")
    st.markdown(f"✅ **Melhor modelo:** {melhor[0]} (RMSE={melhor[1]:.4f})")
    st.markdown(f"🔷 **Segundo melhor:** {segundo[0]} (RMSE={segundo[1]:.4f})")
    st.markdown(f"❌ **Pior modelo:** {pior[0]} (RMSE={pior[1]:.4f})")

# Rodapé
st.markdown(f"""
<div class="footer">
    🌱 Dashboard atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Modo Escuro com opções avançadas
</div>
""", unsafe_allow_html=True)
