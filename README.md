# 🌳 Sistema Avançado de Previsão de Desmatamento

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ALucas314/DesmatamentoIA/blob/AlgoritimoComDashboard/LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.11.0-FF4B4B.svg)](https://streamlit.io/)

🔗 **Repositório:** [https://github.com/ALucas314/DesmatamentoIA/tree/AlgoritimoComDashboard](https://github.com/ALucas314/DesmatamentoIA/tree/AlgoritimoComDashboard)

---

## 📌 Visão Geral

Sistema preditivo para identificação de áreas de risco de desmatamento utilizando três algoritmos de machine learning:

* ✅ **Random Forest**
* ✅ **XGBoost**
* ✅ **LightGBM** (melhor desempenho)

---

## 📊 Métricas Comparativas

| Modelo        | RMSE       | MAE        | MAPE       | R²         | Pearson R  |
| ------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| Random Forest | 0.2749     | 0.1353     | 73.61%     | 0.1792     | 0.4302     |
| XGBoost       | 0.2739     | 0.1347     | 73.54%     | 0.1859     | 0.4376     |
| **LightGBM**  | **0.2736** | **0.1338** | **73.07%** | **0.1877** | **0.4418** |

---

## 🏗️ Estrutura do Projeto

```
DesmatamentoIA/
├── AlgoritimoComDashboard/
│   ├── data/
│   │   └── dados_desmatamento.csv
│   ├── models/
│   │   ├── AlgoritimosTreinamentoRegressao.py
│   │   ├── LightGBM_modelo.pkl
│   │   ├── RandomForest_modelo.pkl
│   │   └── XGBoost_modelo.pkl
│   ├── src/
│   │   ├── app.py
│   │   ├── preditorde_desmatamento.py
│   │   └── dashboard.py
```

---

## 🚀 Como Executar

### Pré-requisitos

* Python 3.8+
* Git

### Instalação e execução

No terminal:

```bash
git clone https://github.com/ALucas314/DesmatamentoIA.git
cd DesmatamentoIA/AlgoritimoComDashboard

pip install -r requirements.txt

# Executar análise preditiva
python src/preditorde_desmatamento.py

# Iniciar dashboard interativo (abre em http://localhost:8501)
streamlit run src/dashboard.py
```

---

## 🛠️ Funcionalidades

### Pré-processamento

* Tratamento automático de datas
* Winsorização de outliers
* Transformação logarítmica da variável alvo
* Criação de features temporais

### Modelagem

* Validação cruzada (5 folds)
* Otimização de hiperparâmetros
* Métricas robustas de avaliação

### Visualização

* Gráficos comparativos das métricas
* Análise de dispersão das previsões
* Dashboard interativo via Streamlit

---

## 📚 Dependências

```
numpy==1.21.5
pandas==1.3.5
scikit-learn==1.0.2
xgboost==1.5.1
lightgbm==3.3.2
matplotlib==3.5.1
seaborn==0.11.2
plotly==5.6.0
streamlit==1.11.0
```

---

## 📝 Licença

Este projeto está licenciado sob a **MIT License**. Veja o arquivo [LICENSE](https://github.com/ALucas314/DesmatamentoIA/blob/AlgoritimoComDashboard/LICENSE) para mais detalhes.

---

## ✉️ Contato

* **Autor:** ALucas314
* **Contribuições:** Aberto para issues e pull requests

---

Aqui está a seção completa e organizada para **instalar todas as bibliotecas necessárias** de forma explícita, que você pode incluir no README, junto com os imports e descrições. Vou fazer um trecho focado na instalação das bibliotecas, usando o pip sem ambiente virtual (globalmente), conforme seu pedido.

---

## 📚 Instalação das Bibliotecas Necessárias

Para rodar o projeto e o dashboard, você precisa instalar todas as dependências listadas abaixo.

### Comando único para instalação (sem ambiente virtual)

Abra o terminal (cmd, PowerShell, bash, ou terminal do VSCode) e rode o comando:

```bash
pip install streamlit pandas numpy plotly matplotlib seaborn scikit-learn xgboost lightgbm
```

> **Obs:** Se seu Python usa `python3` e `pip3`, substitua o comando por:

```bash
pip3 install streamlit pandas numpy plotly matplotlib seaborn scikit-learn xgboost lightgbm
```

---

## 📄 Bibliotecas usadas no projeto

No código, são usados os seguintes imports principais:

```python
# Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# Algoritmo de Treinamento e Análise
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mstats, pearsonr
import warnings
import pickle
import os
```

---

## 🖥️ Dicas para rodar no VSCode

1. Abra o terminal integrado (\`Ctrl + \`\`)
2. Execute o comando de instalação acima para garantir todas as libs
3. Para rodar o script principal (treinamento e análise):

```bash
python src/app.py
```

4. Para iniciar o dashboard interativo:

```bash
streamlit run src/app.py
```

---


