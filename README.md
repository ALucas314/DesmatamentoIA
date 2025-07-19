```markdown
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
│   │   └── dados\_desmatamento.csv
│   ├── models/
│   │   ├── AlgoritimosTreinamentoRegressao.py
│   │   ├── LightGBM\_modelo.pkl
│   │   ├── RandomForest\_modelo.pkl
│   │   └── XGBoost\_modelo.pkl
│   ├── src/
│   │   ├── app.py
│   │   ├── preditorde\_desmatamento.py
│   │   └── dashboard.py

````

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
````

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

Liste as bibliotecas usadas neste projeto e suas versões compatíveis:

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

## 📚 Instalação das Bibliotecas Necessárias (SEM usar ambiente virtual)

Abra o terminal do sistema ou o terminal integrado do VSCode (Ctrl + \`) e execute:

```bash
pip install streamlit pandas numpy plotly matplotlib seaborn scikit-learn xgboost lightgbm
```

> Se seu sistema usa `python3` e `pip3`, use:

```bash
pip3 install streamlit pandas numpy plotly matplotlib seaborn scikit-learn xgboost lightgbm
```

---

## 📄 Bibliotecas usadas no projeto

No código, são utilizados os seguintes imports principais:

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
2. Execute o comando para instalar as bibliotecas (se ainda não instalou)
3. Para rodar o script principal (treinamento e análise):

```bash
python src/preditorde_desmatamento.py
```

4. Para iniciar o dashboard interativo:

```bash
streamlit run src/dashboard.py
```

---

## 📝 Licença

Este projeto está licenciado sob a **MIT License**. Veja o arquivo [LICENSE](https://github.com/ALucas314/DesmatamentoIA/blob/AlgoritimoComDashboard/LICENSE) para mais detalhes.

---

## ✉️ Contato

* **Autor:** ALucas314
* **Contribuições:** Aberto para issues e pull requests

---

```
```
