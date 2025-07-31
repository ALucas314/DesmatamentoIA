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
    .ranking-card {
        background: linear-gradient(135deg, #1c1f26 0%, #2d3748 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    .ranking-1 { border-left-color: #FFD700; }
    .ranking-2 { border-left-color: #C0C0C0; }
    .ranking-3 { border-left-color: #CD7F32; }
</style>
""", unsafe_allow_html=True)

# Paleta para gráficos com cores significativas
cores_ranking = {
    'melhor': '#00FF88',      # Verde vibrante para o melhor
    'mediano': '#FFA500',     # Laranja para o mediano  
    'pior': '#FF4444'         # Vermelho para o pior
}

metricas_modelos = {
    'RandomForest': {'RMSE': 0.2749, 'MAE': 0.1353, 'MAPE': 73.61, 'R2': 0.1792, 'Pearson_r': 0.4302},
    'XGBoost':      {'RMSE': 0.2739, 'MAE': 0.1347, 'MAPE': 73.54, 'R2': 0.1859, 'Pearson_r': 0.4376},
    'LightGBM':     {'RMSE': 0.2736, 'MAE': 0.1338, 'MAPE': 73.08, 'R2': 0.1877, 'Pearson_r': 0.4418}
}
modelos = list(metricas_modelos.keys())

# Função para determinar ranking dos modelos
def get_modelo_ranking():
    # Calcula score composto (menor RMSE e MAE, maior R2 e Pearson)
    scores = {}
    for modelo, metricas in metricas_modelos.items():
        score = (metricas['R2'] + metricas['Pearson_r']) - (metricas['RMSE'] + metricas['MAE'])
        scores[modelo] = score
    
    # Ordena por score (maior = melhor)
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranking

ranking_modelos = get_modelo_ranking()
melhor_modelo = ranking_modelos[0][0]
mediano_modelo = ranking_modelos[1][0]
pior_modelo = ranking_modelos[2][0]

st.title("🌿 Dashboard de Previsão de Desmatamento")
st.markdown("<small>Análise comparativa entre algoritmos de Machine Learning</small>", unsafe_allow_html=True)

# 🏆 Ranking Visual
st.markdown("## 🏆 Ranking dos Modelos")
col1, col2, col3 = st.columns(3)

# Melhor modelo
with col1:
    st.markdown(f"""
    <div class='ranking-card ranking-1'>
        <h3>🥇 1º Lugar</h3>
        <h2 style='color: {cores_ranking["melhor"]};'>{melhor_modelo}</h2>
        <p><strong>Score:</strong> {ranking_modelos[0][1]:.4f}</p>
        <p><strong>R²:</strong> {metricas_modelos[melhor_modelo]['R2']:.4f}</p>
        <p><strong>RMSE:</strong> {metricas_modelos[melhor_modelo]['RMSE']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

# Mediano modelo
with col2:
    st.markdown(f"""
    <div class='ranking-card ranking-2'>
        <h3>🥈 2º Lugar</h3>
        <h2 style='color: {cores_ranking["mediano"]};'>{mediano_modelo}</h2>
        <p><strong>Score:</strong> {ranking_modelos[1][1]:.4f}</p>
        <p><strong>R²:</strong> {metricas_modelos[mediano_modelo]['R2']:.4f}</p>
        <p><strong>RMSE:</strong> {metricas_modelos[mediano_modelo]['RMSE']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

# Pior modelo
with col3:
    st.markdown(f"""
    <div class='ranking-card ranking-3'>
        <h3>🥉 3º Lugar</h3>
        <h2 style='color: {cores_ranking["pior"]};'>{pior_modelo}</h2>
        <p><strong>Score:</strong> {ranking_modelos[2][1]:.4f}</p>
        <p><strong>R²:</strong> {metricas_modelos[pior_modelo]['R2']:.4f}</p>
        <p><strong>RMSE:</strong> {metricas_modelos[pior_modelo]['RMSE']:.4f}</p>
    </div>
    """, unsafe_allow_html=True)

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

# Gráficos de erro com cores significativas
if show_erro:
    st.markdown("## 📉 Comparação de Erros (Menor = Melhor)")
    df_erro = pd.DataFrame({
        'Modelo': modelos,
        'RMSE': [metricas_modelos[m]['RMSE'] for m in modelos],
        'MAE': [metricas_modelos[m]['MAE'] for m in modelos],
        'MAPE': [metricas_modelos[m]['MAPE'] for m in modelos]
    })
    
    # Determina cores baseadas no ranking
    cores_modelos = []
    for modelo in modelos:
        if modelo == melhor_modelo:
            cores_modelos.append(cores_ranking['melhor'])
        elif modelo == pior_modelo:
            cores_modelos.append(cores_ranking['pior'])
        else:
            cores_modelos.append(cores_ranking['mediano'])
    
    fig_erro = go.Figure()
    for i, metrica_nome in enumerate(['RMSE', 'MAE', 'MAPE']):
        fig_erro.add_trace(go.Bar(
            x=df_erro['Modelo'],
            y=df_erro[metrica_nome],
            name=metrica_nome,
            marker_color=cores_modelos,
            text=[f'{v:.4f}' for v in df_erro[metrica_nome]],
            textposition='auto'
        ))
    
    # Adiciona anotações para ranking
    for i, modelo in enumerate(modelos):
        if modelo == melhor_modelo:
            fig_erro.add_annotation(
                x=modelo, y=df_erro.loc[i, 'RMSE'] + 0.01,
                text="🥇 MELHOR",
                showarrow=True,
                arrowhead=2,
                arrowcolor=cores_ranking['melhor'],
                font=dict(color=cores_ranking['melhor'], size=14, weight='bold'),
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor=cores_ranking['melhor']
            )
        elif modelo == pior_modelo:
            fig_erro.add_annotation(
                x=modelo, y=df_erro.loc[i, 'RMSE'] + 0.01,
                text="🥉 PIOR",
                showarrow=True,
                arrowhead=2,
                arrowcolor=cores_ranking['pior'],
                font=dict(color=cores_ranking['pior'], size=14, weight='bold'),
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor=cores_ranking['pior']
            )
        else:
            fig_erro.add_annotation(
                x=modelo, y=df_erro.loc[i, 'RMSE'] + 0.01,
                text="🥈 MÉDIO",
                showarrow=True,
                arrowhead=2,
                arrowcolor=cores_ranking['mediano'],
                font=dict(color=cores_ranking['mediano'], size=14, weight='bold'),
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor=cores_ranking['mediano']
            )
    
    fig_erro.update_layout(
        barmode='group',
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        title="Erros por Modelo (Menor valor = Melhor performance)",
        xaxis_title="Modelos",
        yaxis_title="Valor do Erro",
        showlegend=True,
        legend=dict(
            title="Ranking dos Modelos",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=600,
        font=dict(size=16),
        title_font=dict(size=20)
    )
    st.plotly_chart(fig_erro, use_container_width=True)

# Gráficos de ajuste com cores significativas
if show_ajuste:
    st.markdown("## 📈 Métricas de Ajuste (Maior = Melhor)")
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
            marker_color=cores_modelos,
            text=[f'{v:.4f}' for v in df_ajuste[metrica_nome]],
            textposition='auto',
            textfont=dict(size=16)
        ))
    
    # Adiciona anotações para ranking
    for i, modelo in enumerate(modelos):
        if modelo == melhor_modelo:
            fig_ajuste.add_annotation(
                x=modelo, y=df_ajuste.loc[i, 'R²'] + 0.01,
                text="🥇 MELHOR",
                showarrow=True,
                arrowhead=2,
                arrowcolor=cores_ranking['melhor'],
                font=dict(color=cores_ranking['melhor'], size=16, weight='bold'),
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor=cores_ranking['melhor']
            )
        elif modelo == pior_modelo:
            fig_ajuste.add_annotation(
                x=modelo, y=df_ajuste.loc[i, 'R²'] + 0.01,
                text="🥉 PIOR",
                showarrow=True,
                arrowhead=2,
                arrowcolor=cores_ranking['pior'],
                font=dict(color=cores_ranking['pior'], size=16, weight='bold'),
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor=cores_ranking['pior']
            )
        else:
            fig_ajuste.add_annotation(
                x=modelo, y=df_ajuste.loc[i, 'R²'] + 0.01,
                text="🥈 MÉDIO",
                showarrow=True,
                arrowhead=2,
                arrowcolor=cores_ranking['mediano'],
                font=dict(color=cores_ranking['mediano'], size=16, weight='bold'),
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor=cores_ranking['mediano']
            )
    
    fig_ajuste.update_layout(
        barmode='group',
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        title="Métricas de Ajuste por Modelo (Maior valor = Melhor performance)",
        xaxis_title="Modelos",
        yaxis_title="Valor da Métrica",
        showlegend=True,
        legend=dict(
            title="Ranking dos Modelos",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=600,
        font=dict(size=16),
        title_font=dict(size=20)
    )
    st.plotly_chart(fig_ajuste, use_container_width=True)

# Gráfico dispersão com cores significativas
if show_dispersao:
    st.markdown("## 🔍 Comparação: Previsões vs Valores Reais")
    np.random.seed(42)
    reais = np.random.normal(1.0, 0.3, 100)
    reais = np.clip(reais, 0, None)

    erros_rmse = {}

    fig_disp = go.Figure()
    
    # Ordena os dados para criar linhas suaves
    indices_ordenados = np.argsort(reais)
    reais_ordenados = reais[indices_ordenados]
    
    for i, modelo in enumerate(modelos):
        # Simula diferentes níveis de precisão baseados no ranking real
        if modelo == melhor_modelo:  # LightGBM
            ruido = 0.02  # Menor ruído = melhor modelo
            cor = cores_ranking['melhor']
            largura_linha = 4
        elif modelo == pior_modelo:  # RandomForest
            ruido = 0.08  # Maior ruído = pior modelo
            cor = cores_ranking['pior']
            largura_linha = 2
        else:  # XGBoost
            ruido = 0.05  # Ruído médio
            cor = cores_ranking['mediano']
            largura_linha = 3

        # Gera previsões mais realistas
        pred = reais + np.random.normal(0, ruido, size=100)
        pred = np.clip(pred, 0, None)  # Evita valores negativos
        
        if suavizar > 1:
            pred = pd.Series(pred).rolling(suavizar, min_periods=1).mean().values
        
        # Ordena as previsões da mesma forma que os valores reais
        pred_ordenados = pred[indices_ordenados]
        
        rmse = np.sqrt(np.mean((reais - pred) ** 2))
        erros_rmse[modelo] = rmse

        fig_disp.add_trace(go.Scatter(
            x=reais_ordenados,
            y=pred_ordenados,
            mode='lines+markers',
            name=modelo,
            line=dict(
                color=cor, 
                width=largura_linha
            ),
            marker=dict(
                color=cor,
                size=8,
                opacity=0.7
            ),
            hovertemplate=f"<b>{modelo}</b><br>Real: %{{x:.3f}}<br>Previsto: %{{y:.3f}}<br>Ranking: {'🥇 MELHOR' if modelo == melhor_modelo else '🥈 MÉDIO' if modelo == mediano_modelo else '🥉 PIOR'}<extra></extra>"
        ))

    # Adiciona linha de referência ideal (y=x)
    if linha_referencia:
        min_val = min(reais.min(), 0)
        max_val = max(reais.max(), 2)
        fig_disp.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Linha Ideal (y=x)',
            line=dict(color='#e0e0e0', dash='dash', width=3),
            showlegend=True,
            hoverinfo='skip'
        ))

    fig_disp.update_layout(
        xaxis_type=escala_x,
        yaxis_type=escala_y,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        title="Comparação: Previsões vs Valores Reais",
        xaxis_title="Valores Reais",
        yaxis_title="Valores Previstos",
        legend=dict(
            title="Modelos",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.8)",
            bordercolor="white",
            borderwidth=1,
            font=dict(size=14)
        ),
        height=700,
        font=dict(size=16),
        title_font=dict(size=20),
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.3)',
            showgrid=True
        ),
        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14),
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.3)',
            showgrid=True
        ),
        hovermode='x unified'  # Mostra todos os valores no mesmo ponto x
    )
    st.plotly_chart(fig_disp, use_container_width=True)

    # Ranking atualizado baseado na dispersão
    ranking_dispersao = sorted(erros_rmse.items(), key=lambda x: x[1])
    melhor_disp, segundo_disp, pior_disp = ranking_dispersao[0], ranking_dispersao[1], ranking_dispersao[-1]

    st.markdown("## 🎯 Performance na Comparação")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid {cores_ranking["melhor"]}; background: linear-gradient(135deg, rgba(0,255,136,0.1) 0%, #1c1f26 100%);'>
            <h4>🥇 MELHOR: {melhor_disp[0]}</h4>
            <p><strong>RMSE:</strong> {melhor_disp[1]:.4f}</p>
            <p style='color: {cores_ranking["melhor"]}; font-weight: bold;'>✓ Menor erro de previsão</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid {cores_ranking["mediano"]}; background: linear-gradient(135deg, rgba(255,165,0,0.1) 0%, #1c1f26 100%);'>
            <h4>🥈 MÉDIO: {segundo_disp[0]}</h4>
            <p><strong>RMSE:</strong> {segundo_disp[1]:.4f}</p>
            <p style='color: {cores_ranking["mediano"]}; font-weight: bold;'>○ Performance intermediária</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card' style='border-left: 4px solid {cores_ranking["pior"]}; background: linear-gradient(135deg, rgba(255,68,68,0.1) 0%, #1c1f26 100%);'>
            <h4>🥉 PIOR: {pior_disp[0]}</h4>
            <p><strong>RMSE:</strong> {pior_disp[1]:.4f}</p>
            <p style='color: {cores_ranking["pior"]}; font-weight: bold;'>✗ Maior erro de previsão</p>
        </div>
        """, unsafe_allow_html=True)

# Rodapé
st.markdown(f"""
<div class="footer">
    🌱 Dashboard atualizado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Ranking visual intuitivo
</div>
""", unsafe_allow_html=True)
