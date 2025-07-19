import pandas as pd
import numpy as np
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
warnings.filterwarnings("ignore")

sns.set(style="whitegrid", palette="pastel", font_scale=1.2)

class PreditordeDesmatamentoAvancado:
    def __init__(self):
        self.modelos = {
            "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
            "XGBoost": XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
            "LightGBM": LGBMRegressor(random_state=42, n_jobs=-1)
        }

    def carregar_dados(self, arquivo_csv):
        for sep in ['\t', ',', ';']:
            try:
                df = pd.read_csv(arquivo_csv, sep=sep)
                if 'viewDate' in df.columns:
                    print(f"✅ Dados carregados com {df.shape[0]} linhas e {df.shape[1]} colunas (delimitador '{sep}')")
                    return df
            except Exception:
                continue
        raise ValueError("Não foi possível detectar o delimitador do arquivo ou coluna 'viewDate' não encontrada.")

    def preparar_dados(self, df):
        df['viewDate'] = pd.to_datetime(df['viewDate'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['viewDate'])
        df['year'] = df['viewDate'].dt.year
        df['month'] = df['viewDate'].dt.month
        df['day'] = df['viewDate'].dt.day
        df['weekday'] = df['viewDate'].dt.weekday
        df['quarter'] = df['viewDate'].dt.quarter

        df['areaMunKm_winsor'] = mstats.winsorize(df['areaMunKm'], limits=[0.01, 0.01])
        df['target_log'] = np.log1p(df['areaMunKm_winsor'])

        X = df[['areaUcKm', 'year', 'month', 'day', 'weekday', 'quarter', 'uf', 'className']].copy()
        X['areaUcKm'] = X['areaUcKm'].fillna(0)
        X['areaUcKm_year'] = X['areaUcKm'] * X['year']

        for col in ['uf', 'className']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        y = df['target_log']

        return X, y, df

    def avaliar_modelo(self, modelo, X_train, y_train, X_test, y_test):
        modelo.fit(X_train, y_train)
        preds_log = modelo.predict(X_test)
        preds = np.expm1(preds_log)
        y_true = np.expm1(y_test)

        rmse = np.sqrt(mean_squared_error(y_true, preds))
        mae = mean_absolute_error(y_true, preds)
        medae = np.median(np.abs(y_true - preds))
        mape = np.mean(np.abs((y_true - preds) / (y_true + 1e-9))) * 100
        r2 = r2_score(y_true, preds)
        pearson_corr, _ = pearsonr(y_true, preds)

        return {
            'rmse': rmse,
            'mae': mae,
            'medae': medae,
            'mape': mape,
            'r2': r2,
            'pearson_r': pearson_corr
        }

    def cross_validate_modelos(self, X, y, n_splits=5):
        resultados = {nome: {'rmse': [], 'mae': [], 'medae': [], 'mape': [], 'r2': [], 'pearson_r': []} for nome in self.modelos.keys()}
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"➡️ Fold {fold +1} de {n_splits}")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            for nome, modelo in self.modelos.items():
                res = self.avaliar_modelo(modelo, X_train, y_train, X_test, y_test)
                for metric in resultados[nome].keys():
                    resultados[nome][metric].append(res[metric])

        resumo = {}
        for nome in resultados:
            resumo[nome] = {metric: np.mean(vals) for metric, vals in resultados[nome].items()}
        return resumo

    def treinar_final_e_gerar_previsoes(self, X, y):
        modelos_treinados = {}
        preds_finais = {}

        for nome, modelo in self.modelos.items():
            modelo.fit(X, y)
            pred_log = modelo.predict(X)
            pred = np.expm1(pred_log)
            modelos_treinados[nome] = modelo
            preds_finais[nome] = pred

        return modelos_treinados, preds_finais

    def plot_metricas_barras(self, resumo):
        modelos = list(resumo.keys())
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        erros = ['rmse', 'mae', 'mape']
        titles = ['RMSE', 'MAE', 'MAPE (%)']

        for ax, erro, title in zip(axs, erros, titles):
            valores = [resumo[m][erro] for m in modelos]
            ax.bar(modelos, valores, color='cornflowerblue')
            ax.set_title(f'Comparação de {title} entre Modelos', fontsize=14, weight='bold')
            ax.set_ylabel(title)
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            for i, v in enumerate(valores):
                ax.text(i, v * 1.01, f"{v:.3f}", ha='center', fontsize=11)

        plt.tight_layout()
        plt.show()
        print("\n📖 Interpretação: gráficos de barras mostram os erros RMSE, MAE e MAPE para cada modelo. Menores valores indicam melhores performances.")

    def plot_metricas_linhas(self, resumo):
        modelos = list(resumo.keys())
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        metrics = ['r2', 'pearson_r']
        titles = ['R² (Coeficiente de Determinação)', 'Coeficiente de Correlação de Pearson']

        for ax, metric, title in zip(axs, metrics, titles):
            valores = [resumo[m][metric] for m in modelos]
            ax.plot(modelos, valores, marker='o', linestyle='-', color='mediumseagreen')
            ax.set_title(title, fontsize=14, weight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle='--', alpha=0.6)
            for i, v in enumerate(valores):
                ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=11)

        plt.tight_layout()
        plt.show()
        print("\n📖 Interpretação: gráficos de linha mostram R² e Pearson r para os modelos. Valores próximos de 1 indicam melhor ajuste e correlação.")

    def plot_previsoes_todos_modelos(self, df, preds_finais):
        modelos = list(preds_finais.keys())
        n = len(modelos)
        plt.figure(figsize=(16, 5 * n))
        y_true = df['areaMunKm'].values

        # Limites para zoom entre os percentis 1% e 99%
        lower_lim = np.percentile(y_true, 1)
        upper_lim = np.percentile(y_true, 99)

        for i, modelo in enumerate(modelos, 1):
            plt.subplot(n, 1, i)
            y_pred = preds_finais[modelo]

            # Filtra dados para o intervalo dos percentis
            mask = (y_true >= lower_lim) & (y_true <= upper_lim) & (y_pred >= lower_lim) & (y_pred <= upper_lim)
            x_plot = y_true[mask]
            y_plot = y_pred[mask]

            plt.scatter(x_plot, y_plot, alpha=0.5, color='tab:blue', edgecolor='k', linewidth=0.3)
            plt.plot([lower_lim, upper_lim], [lower_lim, upper_lim], 'r--', lw=2)
            plt.xlim(lower_lim, upper_lim)
            plt.ylim(lower_lim, upper_lim)

            plt.xlabel('Valores Reais (Área Municipal Km²)', fontsize=12, weight='bold')
            plt.ylabel('Previsões (Área Municipal Km²)', fontsize=12, weight='bold')
            plt.title(f'Previsões vs Valores Reais - Modelo: {modelo} (zoom nos 98% centrais)', fontsize=14, weight='bold')
            plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        print("\n📖 Interpretação: pontos muito extremos foram removidos para melhor visualização, focando na faixa principal dos dados.")

    def pipeline_completo(self, arquivo_csv):
        df = self.carregar_dados(arquivo_csv)
        X, y, df = self.preparar_dados(df)

        print("\n🔍 Validando modelos com K-Fold Cross Validation (5 folds):")
        resumo_cv = self.cross_validate_modelos(X, y)

        print("\n📊 Resultados detalhados por modelo:")
        for modelo, metricas in resumo_cv.items():
            print(f"➡️ {modelo}: ", end="")
            print(", ".join([f"{k.upper()}={v:.4f}" for k, v in metricas.items()]))

        print("\n📈 Gerando gráficos simples de comparação de métricas de erro...")
        self.plot_metricas_barras(resumo_cv)

        print("\n📈 Gerando gráficos simples de comparação de métricas de ajuste e correlação...")
        self.plot_metricas_linhas(resumo_cv)

        print("\n🔧 Treinando modelo final com o dataset completo para previsões...")
        modelos_treinados, preds_finais = self.treinar_final_e_gerar_previsoes(X, y)

        print("\n📉 Gerando gráficos simples de previsões vs reais para todos os modelos...")
        self.plot_previsoes_todos_modelos(df, preds_finais)

        melhor_modelo = min(resumo_cv.keys(), key=lambda m: resumo_cv[m]['rmse'])
        print(f"\n🏆 Melhor modelo: {melhor_modelo} com RMSE médio de {resumo_cv[melhor_modelo]['rmse']:.4f}")

        return resumo_cv, modelos_treinados, preds_finais

if __name__ == "__main__":
    predictor = PreditordeDesmatamentoAvancado()
    resumo_cv, modelos_treinados, preds_finais = predictor.pipeline_completo("dados_desmatamento.csv")
