# 🌳 Sistema Avançado de Previsão de Desmatamento

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ALucas314/DesmatamentoIA/blob/AlgoritimoComDashboard/LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.11.0-FF4B4B.svg)](https://streamlit.io/)

🔗 **Repositório:** [https://github.com/ALucas314/DesmatamentoIA/tree/AlgoritimoComDashboard](https://github.com/ALucas314/DesmatamentoIA/tree/AlgoritimoComDashboard)

## 📌 Visão Geral

Sistema preditivo para identificação de áreas de risco de desmatamento utilizando três algoritmos de machine learning:

- ✅ **Random Forest**
- ✅ **XGBoost**  
- ✅ **LightGBM** (melhor desempenho)

## 📊 Métricas Comparativas

| Modelo        | RMSE   | MAE    | MAPE   | R²     | Pearson R |
|---------------|--------|--------|--------|--------|-----------|
| Random Forest | 0.2749 | 0.1353 | 73.61% | 0.1792 | 0.4302    |
| XGBoost       | 0.2739 | 0.1347 | 73.54% | 0.1859 | 0.4376    |
| **LightGBM**  | **0.2736** | **0.1338** | **73.07%** | **0.1877** | **0.4418** |

## 🏗️ Estrutura do Projeto

DesmatamentoIA/
├── AlgoritimoComDashboard/
│ ├── data/ dados_desmatamento.csv
│ ├── models/
│ │ ├── AlgoritimosTreinamentoRegressao.py
│ │ ├── LightGBM_modelo.pkl
│ │ ├── RandomForest_modelo.pkl
│ │ ├── XGBoost_modelo.pkl
│ ├── src/
│ │ ├── app.py



## 🚀 Como Executar

### Pré-requisitos
- Python 3.8+
- Git

### Instalação
```bash
git clone https://github.com/ALucas314/DesmatamentoIA.git
cd DesmatamentoIA/AlgoritimoComDashboard
pip install -r requirements.txt

# Executar análise preditiva
python src/preditorde_desmatamento.py

# Iniciar dashboard (http://localhost:8501)
streamlit run src/dashboard.py

🛠️ Funcionalidades
Pré-processamento
Tratamento automático de datas

Winsorização de outliers

Transformação log da variável alvo

Criação de features temporais

Modelagem
Validação cruzada (5 folds)

Otimização de hiperparâmetros

Métricas robustas de avaliação

Visualização
Gráficos comparativos

Análise de dispersão

Dashboard interativo

📚 Dependências
python
numpy==1.21.5
pandas==1.3.5
scikit-learn==1.0.2
xgboost==1.5.1
lightgbm==3.3.2
matplotlib==3.5.1
seaborn==0.11.2
plotly==5.6.0
streamlit==1.11.0
📝 Licença
Este projeto está licenciado sob a MIT License.

✉️ Contato
Autor: ALucas314
Contribuições: Aberto para issues e pull requests

text
New chat
