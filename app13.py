import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import io
import requests
import yfinance as yf
import base64

warnings.filterwarnings('ignore')

# ---------- CONFIGURAÇÃO DA PÁGINA STREAMLIT ----------
st.set_page_config(
    page_title="Projeção de Câmbio - ML & Ensemble",
    page_icon="💱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- ESTILOS GLOBAIS ----------
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)
plt.rcParams['font.size'] = 12

# ---------- FUNÇÕES ORIGINAIS (ADAPTADAS) ----------

def selecionar_tipo_analise():
    """Interface no sidebar para escolha do tipo de análise"""
    opcao = st.sidebar.radio(
        "⏰ Tipo de Análise",
        options=["MENSAL", "DIÁRIA", "HORÁRIA"],
        index=0,
        help="Mensal: projeção para 12 meses | Diária: 12 dias | Horária: 24 horas"
    )
    if opcao == "DIÁRIA":
        return "DIARIA", 12, "dias", "D"
    elif opcao == "HORÁRIA":
        return "HORARIA", 24, "horas", "H"
    else:
        return "MENSAL", 12, "meses", "ME"

def selecionar_par_moedas():
    """Interface no sidebar para escolha do par"""
    pares = {
        "USD/BRL": "Dólar Americano vs Real Brasileiro",
        "EUR/BRL": "Euro vs Real Brasileiro",
        "GBP/BRL": "Libra Esterlina vs Real Brasileiro",
        "JPY/BRL": "Iene Japonês vs Real Brasileiro"
    }
    par = st.sidebar.selectbox(
        "🌍 Par de Moedas",
        options=list(pares.keys()),
        format_func=lambda x: f"{x} - {pares[x]}"
    )
    return par

def obter_parametros(tipo_analise, par_moedas):
    """Retorna parâmetros específicos para cada tipo de análise e par de moedas - DADOS REAIS OUTUBRO 2025"""
    parametros_base = {
        'MENSAL': {
            'USD/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 4.25,
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.80,
                'volatilidade': 2.1, 'commodity_weight': 0.25,
                'arima_order': (2,1,2), 'periodos_projecao': 12,
                'n_simulacoes': 10000, 'moeda_base': 'USD',
                'moeda_contraria': 'BRL', 'nome_completo': 'Dólar Americano vs Real Brasileiro'
            },
            'EUR/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 2.15,
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.20,
                'volatilidade': 1.8, 'commodity_weight': 0.18,
                'arima_order': (2,1,2), 'periodos_projecao': 12,
                'n_simulacoes': 10000, 'moeda_base': 'EUR',
                'moeda_contraria': 'BRL', 'nome_completo': 'Euro vs Real Brasileiro'
            },
            'GBP/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 4.00,
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.50,
                'volatilidade': 1.9, 'commodity_weight': 0.15,
                'arima_order': (2,1,2), 'periodos_projecao': 12,
                'n_simulacoes': 10000, 'moeda_base': 'GBP',
                'moeda_contraria': 'BRL', 'nome_completo': 'Libra Esterlina vs Real Brasileiro'
            },
            'JPY/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 0.50,
                'inflacao_brasil': 4.80, 'inflacao_exterior': 1.50,
                'volatilidade': 1.5, 'commodity_weight': 0.08,
                'arima_order': (2,1,2), 'periodos_projecao': 12,
                'n_simulacoes': 10000, 'moeda_base': 'JPY',
                'moeda_contraria': 'BRL', 'nome_completo': 'Iene Japonês vs Real Brasileiro'
            }
        },
        'DIARIA': {
            'USD/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 4.25,
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.80,
                'volatilidade_diaria': 0.015, 'commodity_weight': 0.25,
                'arima_order': (1,1,1), 'periodos_projecao': 12,
                'n_simulacoes': 5000, 'moeda_base': 'USD',
                'moeda_contraria': 'BRL', 'nome_completo': 'Dólar Americano vs Real Brasileiro'
            },
            'EUR/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 2.15,
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.20,
                'volatilidade_diaria': 0.012, 'commodity_weight': 0.18,
                'arima_order': (1,1,1), 'periodos_projecao': 12,
                'n_simulacoes': 5000, 'moeda_base': 'EUR',
                'moeda_contraria': 'BRL', 'nome_completo': 'Euro vs Real Brasileiro'
            },
            'GBP/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 4.00,
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.50,
                'volatilidade_diaria': 0.013, 'commodity_weight': 0.15,
                'arima_order': (1,1,1), 'periodos_projecao': 12,
                'n_simulacoes': 5000, 'moeda_base': 'GBP',
                'moeda_contraria': 'BRL', 'nome_completo': 'Libra Esterlina vs Real Brasileiro'
            },
            'JPY/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 0.50,
                'inflacao_brasil': 4.80, 'inflacao_exterior': 1.50,
                'volatilidade_diaria': 0.010, 'commodity_weight': 0.08,
                'arima_order': (1,1,1), 'periodos_projecao': 12,
                'n_simulacoes': 5000, 'moeda_base': 'JPY',
                'moeda_contraria': 'BRL', 'nome_completo': 'Iene Japonês vs Real Brasileiro'
            }
        },
        'HORARIA': {
            'USD/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 4.25,
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.80,
                'volatilidade_horaria': 0.002, 'commodity_weight': 0.25,
                'arima_order': (1,1,0), 'periodos_projecao': 24,
                'n_simulacoes': 2000, 'moeda_base': 'USD',
                'moeda_contraria': 'BRL', 'nome_completo': 'Dólar Americano vs Real Brasileiro'
            },
            'EUR/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 2.15,
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.20,
                'volatilidade_horaria': 0.0015, 'commodity_weight': 0.18,
                'arima_order': (1,1,0), 'periodos_projecao': 24,
                'n_simulacoes': 2000, 'moeda_base': 'EUR',
                'moeda_contraria': 'BRL', 'nome_completo': 'Euro vs Real Brasileiro'
            },
            'GBP/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 4.00,
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.50,
                'volatilidade_horaria': 0.0018, 'commodity_weight': 0.15,
                'arima_order': (1,1,0), 'periodos_projecao': 24,
                'n_simulacoes': 2000, 'moeda_base': 'GBP',
                'moeda_contraria': 'BRL', 'nome_completo': 'Libra Esterlina vs Real Brasileiro'
            },
            'JPY/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 0.50,
                'inflacao_brasil': 4.80, 'inflacao_exterior': 1.50,
                'volatilidade_horaria': 0.0010, 'commodity_weight': 0.08,
                'arima_order': (1,1,0), 'periodos_projecao': 24,
                'n_simulacoes': 2000, 'moeda_base': 'JPY',
                'moeda_contraria': 'BRL', 'nome_completo': 'Iene Japonês vs Real Brasileiro'
            }
        }
    }
    return parametros_base[tipo_analise].get(par_moedas, parametros_base[tipo_analise]['USD/BRL'])

def criar_dados_historicos(tipo_analise, par_moedas, cotacao_atual=None):
    """Cria dados históricos de acordo com o tipo de análise (sintéticos)"""
    if cotacao_atual is None:
        cotacoes_padrao = {
            'USD/BRL': 5.2333,
            'EUR/BRL': 6.2200,
            'GBP/BRL': 7.8500,
            'JPY/BRL': 0.0330
        }
        cotacao_atual = cotacoes_padrao.get(par_moedas, 5.2333)

    np.random.seed(42)
    if tipo_analise == 'DIARIA':
        dates = pd.date_range(start='2022-01-01', end=datetime.now(), freq='D')
        dates = dates[dates.dayofweek < 5]
        n_periods = len(dates)
        base_value = cotacao_atual * 0.95
        valores = [base_value]
        for i in range(1, n_periods):
            variacao = np.random.normal(0, 0.008)
            tendencia = 0.0001
            dia_semana = dates[i].weekday()
            if dia_semana == 0:
                sazonalidade = 0.001
            elif dia_semana == 4:
                sazonalidade = -0.001
            else:
                sazonalidade = 0
            novo_valor = valores[-1] * (1 + variacao + tendencia + sazonalidade)
            valores.append(novo_valor)
        valores[-1] = cotacao_atual
        serie = pd.Series(valores, index=dates, name='Cotação')
    elif tipo_analise == 'HORARIA':
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='H')
        dates = dates[dates.dayofweek < 5]
        n_periods = len(dates)
        base_value = cotacao_atual * 0.98
        valores = [base_value]
        for i in range(1, n_periods):
            variacao = np.random.normal(0, 0.0008)
            tendencia = 0.00002
            hora = dates[i].hour
            if hora in [0,1,2]:
                sazonalidade = 0.0001
            elif hora in [8,9,10]:
                sazonalidade = 0.0003
            elif hora in [12,13,14]:
                sazonalidade = 0.0004
            elif hora in [17,18,19]:
                sazonalidade = -0.0002
            else:
                sazonalidade = 0
            novo_valor = valores[-1] * (1 + variacao + tendencia + sazonalidade)
            valores.append(novo_valor)
        valores[-1] = cotacao_atual
        serie = pd.Series(valores, index=dates, name='Cotação')
    else:  # MENSAL
        dates = pd.date_range(start='2015-01-01', end=datetime.now(), freq='ME')
        n_periods = len(dates)
        base_value = cotacao_atual * 0.7
        valores = [base_value]
        for i in range(1, n_periods):
            variacao = np.random.normal(0, 0.03)
            tendencia = 0.002
            mes = dates[i].month
            if mes in [7,8]:
                sazonalidade = 0.005
            elif mes in [1,2]:
                sazonalidade = -0.003
            else:
                sazonalidade = 0
            novo_valor = valores[-1] * (1 + variacao + tendencia + sazonalidade)
            valores.append(novo_valor)
        valores[-1] = cotacao_atual
        serie = pd.Series(valores, index=dates, name='Cotação')
    return serie

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ---------- FUNÇÕES DE COTAÇÃO ----------
def buscar_cotacao_twelve(par_moedas, api_key):
    # Se o usuário não forneceu chave, tenta pegar dos secrets do Streamlit
    if not api_key:
        try:
            api_key = st.secrets["TWELVE_API_KEY"]
        except:
            st.error("❌ Chave da API Twelve Data não configurada. Configure nos secrets do Streamlit ou digite uma chave.")
            return None
    
    try:
        simbolos = {'USD/BRL': 'USD/BRL', 'EUR/BRL': 'EUR/BRL', 'GBP/BRL': 'GBP/BRL', 'JPY/BRL': 'JPY/BRL'}
        simbolo = simbolos.get(par_moedas)
        if not simbolo:
            return None
        url = f"https://api.twelvedata.com/price?symbol={simbolo}&apikey={api_key}"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        dados = response.json()
        if 'price' not in dados:
            return None
        cotacao = float(dados['price'])
        ranges_validos = {'USD/BRL': (3.0,7.0), 'EUR/BRL': (4.0,8.0), 'GBP/BRL': (5.0,10.0), 'JPY/BRL': (0.02,0.05)}
        min_val, max_val = ranges_validos.get(par_moedas, (0,100))
        if not (min_val < cotacao < max_val):
            return None
        return cotacao
    except:
        return None

def buscar_cotacao_yahoo(par_moedas):
    try:
        simbolos = {'USD/BRL': 'USDBRL=X', 'EUR/BRL': 'EURBRL=X', 'GBP/BRL': 'GBPBRL=X', 'JPY/BRL': 'JPYBRL=X'}
        simbolo = simbolos.get(par_moedas)
        if not simbolo:
            return None
        ticker = yf.Ticker(simbolo)
        dados = ticker.history(period="1d", interval="1h")
        if dados.empty:
            return None
        cotacao = dados['Close'].iloc[-1]
        ranges_validos = {'USD/BRL': (3.0,7.0), 'EUR/BRL': (4.0,8.0), 'GBP/BRL': (5.0,10.0), 'JPY/BRL': (0.02,0.05)}
        min_val, max_val = ranges_validos.get(par_moedas, (0,100))
        if not (min_val < cotacao < max_val):
            return None
        return cotacao
    except:
        return None

# ---------- MACHINE LEARNING ----------
def criar_features_ml(dados_historicos, parametros, tipo_analise):
    df = dados_historicos.copy()
    if len(df) < 10:
        df['lag_1'] = df['Cotação'].shift(1)
        df['retorno_1'] = df['Cotação'].pct_change(1)
        df['MM5'] = df['Cotação'].rolling(5, min_periods=1).mean()
        df = df.fillna(method='bfill').fillna(method='ffill')
        return df

    try:
        if tipo_analise == 'DIARIA':
            for lag in [1,2,3,5]:
                df[f'lag_{lag}'] = df['Cotação'].shift(lag)
            df['retorno_1d'] = df['Cotação'].pct_change(1)
            df['retorno_5d'] = df['Cotação'].pct_change(5)
            df['volatilidade_5d'] = df['retorno_1d'].rolling(5, min_periods=1).std()
            df['MM5'] = df['Cotação'].rolling(5, min_periods=1).mean()
            df['MM20'] = df['Cotação'].rolling(20, min_periods=1).mean()
            df['MM5_ratio'] = df['Cotação'] / df['MM5']
            df['MM20_ratio'] = df['Cotação'] / df['MM20']
            df['RSI'] = calculate_rsi(df['Cotação'], window=10)
            df['diferencial_juros'] = (parametros['juros_brasil'] - parametros['juros_exterior']) / 100 / 252
            df['dia_semana'] = df.index.dayofweek
        elif tipo_analise == 'HORARIA':
            for lag in [1,2,4,8]:
                df[f'lag_{lag}'] = df['Cotação'].shift(lag)
            df['retorno_1h'] = df['Cotação'].pct_change(1)
            df['retorno_4h'] = df['Cotação'].pct_change(4)
            df['volatilidade_4h'] = df['retorno_1h'].rolling(4, min_periods=1).std()
            df['MM4'] = df['Cotação'].rolling(4, min_periods=1).mean()
            df['MM12'] = df['Cotação'].rolling(12, min_periods=1).mean()
            df['MM4_ratio'] = df['Cotação'] / df['MM4']
            df['MM12_ratio'] = df['Cotação'] / df['MM12']
            df['RSI'] = calculate_rsi(df['Cotação'], window=12)
            df['diferencial_juros'] = (parametros['juros_brasil'] - parametros['juros_exterior']) / 100 / (365*24)
            df['hora_dia'] = df.index.hour
            df['dia_semana'] = df.index.dayofweek
        else:  # MENSAL
            for lag in [1,2,3]:
                df[f'lag_{lag}'] = df['Cotação'].shift(lag)
            df['retorno_1m'] = df['Cotação'].pct_change(1)
            df['retorno_3m'] = df['Cotação'].pct_change(3)
            df['volatilidade_3m'] = df['retorno_1m'].rolling(3, min_periods=1).std()
            df['MM6'] = df['Cotação'].rolling(6, min_periods=1).mean()
            df['MM12'] = df['Cotação'].rolling(12, min_periods=1).mean()
            df['MM6_ratio'] = df['Cotação'] / df['MM6']
            df['RSI'] = calculate_rsi(df['Cotação'])
            df['diferencial_juros'] = (parametros['juros_brasil'] - parametros['juros_exterior']) / 100
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        return df
    except Exception as e:
        df_fallback = dados_historicos.copy()
        df_fallback['lag_1'] = df_fallback['Cotação'].shift(1)
        df_fallback['retorno_1'] = df_fallback['Cotação'].pct_change(1)
        return df_fallback.fillna(method='bfill').fillna(method='ffill')

def treinar_modelo_ml(dados_com_features, parametros, tipo_analise):
    if dados_com_features.empty:
        return None, None, None
    colunas_excluir = ['Cotação']
    if tipo_analise == 'DIARIA':
        colunas_excluir.extend(['MM5','MM20'])
    elif tipo_analise == 'HORARIA':
        colunas_excluir.extend(['MM4','MM12'])
    else:
        colunas_excluir.extend(['MM6','MM12'])
    feature_columns = [col for col in dados_com_features.columns if col not in colunas_excluir]
    if len(feature_columns) == 0:
        return None, None, None
    X = dados_com_features[feature_columns]
    y = dados_com_features['Cotação']
    if len(X) < 10:
        return None, None, None
    if tipo_analise == 'HORARIA':
        split_point = max(20, int(len(X)*0.8))
    elif tipo_analise == 'MENSAL':
        split_point = max(6, int(len(X)*0.7))
    else:
        split_point = int(len(X)*0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    if len(X_train) < 5:
        return None, None, None
    if tipo_analise == 'HORARIA':
        model = RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_split=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    elif tipo_analise == 'MENSAL':
        model = RandomForestRegressor(n_estimators=30, max_depth=5, min_samples_split=8, min_samples_leaf=4, random_state=42, n_jobs=-1)
    else:
        model = RandomForestRegressor(n_estimators=30, max_depth=6, min_samples_split=15, min_samples_leaf=8, random_state=42, n_jobs=-1)
    try:
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test) if len(X_test) > 0 else train_score
        y_pred = model.predict(X_test) if len(X_test) > 0 else model.predict(X_train)
        y_actual = y_test if len(y_test) > 0 else y_train
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
        metricas = {'train_score': train_score, 'test_score': test_score, 'rmse': rmse, 'mape': mape}
        return model, feature_columns, metricas
    except:
        return None, None, None

def prever_com_ml(model, feature_columns, dados_historicos, parametros, periodos_projecao, tipo_analise):
    if model is None:
        ultimo_valor = dados_historicos['Cotação'].iloc[-1]
        if tipo_analise == 'DIARIA':
            ultima_data = dados_historicos.index[-1]
            datas_projecao = []
            current_date = ultima_data
            for i in range(periodos_projecao):
                current_date += timedelta(days=1)
                while current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                datas_projecao.append(current_date)
        elif tipo_analise == 'HORARIA':
            ultima_data = dados_historicos.index[-1]
            datas_projecao = []
            current_date = ultima_data
            horas_geradas = 0
            while horas_geradas < periodos_projecao:
                current_date += timedelta(hours=1)
                if current_date.weekday() < 5:
                    datas_projecao.append(current_date)
                    horas_geradas += 1
        else:
            datas_projecao = [dados_historicos.index[-1] + relativedelta(months=i) for i in range(1, periodos_projecao+1)]
        return [ultimo_valor] * len(datas_projecao), datas_projecao

    if tipo_analise == 'DIARIA':
        ultima_data = dados_historicos.index[-1]
        datas_projecao = []
        current_date = ultima_data
        for i in range(periodos_projecao):
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
            datas_projecao.append(current_date)
    elif tipo_analise == 'HORARIA':
        ultima_data = dados_historicos.index[-1]
        datas_projecao = []
        current_date = ultima_data
        horas_geradas = 0
        while horas_geradas < periodos_projecao:
            current_date += timedelta(hours=1)
            if current_date.weekday() < 5:
                datas_projecao.append(current_date)
                horas_geradas += 1
    else:
        datas_projecao = [dados_historicos.index[-1] + relativedelta(months=i) for i in range(1, periodos_projecao+1)]

    projecao_ml = []
    if tipo_analise == 'MENSAL':
        try:
            valor_atual = dados_historicos['Cotação'].iloc[-1]
            if len(dados_historicos) >= 12:
                dados_recentes = dados_historicos['Cotação'].tail(12)
                retorno_medio = dados_recentes.pct_change().mean()
                if np.isnan(retorno_medio):
                    retorno_medio = 0
                tendencia = np.clip(retorno_medio, -0.015, 0.015)
            else:
                tendencia = 0
            dados_com_features = criar_features_ml(dados_historicos, parametros, tipo_analise)
            previsao_base = valor_atual
            if not dados_com_features.empty:
                for col in feature_columns:
                    if col not in dados_com_features.columns:
                        dados_com_features[col] = 0
                ultimas_features = dados_com_features[feature_columns].iloc[-1:]
                previsao_tentativa = model.predict(ultimas_features)[0]
                variacao_tentativa = abs(previsao_tentativa - valor_atual) / valor_atual
                if variacao_tentativa <= 0.08:
                    previsao_base = previsao_tentativa
            for periodo in range(len(datas_projecao)):
                if periodo == 0:
                    previsao = previsao_base
                else:
                    previsao = projecao_ml[-1] * (1 + tendencia * 0.6)
                variacao_total = abs(previsao - valor_atual) / valor_atual
                if variacao_total > 0.15:
                    previsao = valor_atual * (1 + np.sign(tendencia) * 0.15)
                projecao_ml.append(previsao)
        except:
            projecao_ml = [dados_historicos['Cotação'].iloc[-1]] * len(datas_projecao)
    elif tipo_analise == 'HORARIA':
        dados_atual = dados_historicos.copy()
        for periodo in range(len(datas_projecao)):
            try:
                dados_com_features = criar_features_ml(dados_atual, parametros, tipo_analise)
                if not dados_com_features.empty:
                    for col in feature_columns:
                        if col not in dados_com_features.columns:
                            dados_com_features[col] = 0
                    ultimas_features = dados_com_features[feature_columns].iloc[-1:]
                    previsao = model.predict(ultimas_features)[0]
                    if np.isnan(previsao) or previsao <= 0:
                        previsao = dados_atual['Cotação'].iloc[-1]
                else:
                    previsao = dados_atual['Cotação'].iloc[-1]
            except:
                previsao = dados_atual['Cotação'].iloc[-1]
            projecao_ml.append(previsao)
            nova_data = datas_projecao[periodo]
            nova_linha = pd.DataFrame({'Cotação': [previsao]}, index=[nova_data])
            dados_atual = pd.concat([dados_atual, nova_linha])
            dados_atual['MM4'] = dados_atual['Cotação'].rolling(4, min_periods=1).mean()
            dados_atual['MM12'] = dados_atual['Cotação'].rolling(12, min_periods=1).mean()
            dados_atual = dados_atual.fillna(method='ffill')
    else:  # DIARIA
        dados_atual = dados_historicos.copy()
        for periodo in range(len(datas_projecao)):
            dados_com_features = criar_features_ml(dados_atual, parametros, tipo_analise)
            ultimas_features = dados_com_features.iloc[-1:][feature_columns]
            previsao = model.predict(ultimas_features)[0]
            projecao_ml.append(previsao)
            nova_data = datas_projecao[periodo]
            nova_linha = pd.DataFrame({'Cotação': [previsao], 'MM5': [np.nan], 'MM20': [np.nan]}, index=[nova_data])
            dados_atual = pd.concat([dados_atual, nova_linha])
            dados_atual['MM5'] = dados_atual['Cotação'].rolling(5, min_periods=1).mean()
            dados_atual['MM20'] = dados_atual['Cotação'].rolling(20, min_periods=1).mean()
    return projecao_ml, datas_projecao

# ---------- PROJEÇÃO ECONÔMICA ----------
def calcular_projecao_economica(dados_historicos, parametros, periodos_projecao, tipo_analise):
    projecao = []
    valor_atual = dados_historicos['Cotação'].iloc[-1]
    media_historica = dados_historicos['Cotação'].mean()
    desvio_historico = dados_historicos['Cotação'].std()
    valor_maximo_historico = dados_historicos['Cotação'].max()
    for periodo in range(periodos_projecao):
        if tipo_analise == 'DIARIA':
            diferencial_juros = ((parametros['juros_brasil'] - parametros['juros_exterior']) / 100) / 252
            desvio_da_media = (media_historica - valor_atual) / desvio_historico
            forca_reversao = desvio_da_media * 0.02
            resistencia = -0.001 if valor_atual > valor_maximo_historico * 0.98 else 0
            choque_volatilidade = np.random.normal(0, parametros['volatilidade_diaria'])
            dia_semana = (dados_historicos.index[-1].weekday() + periodo) % 7
            sazonalidade = 0.001 if dia_semana == 0 else (-0.001 if dia_semana == 4 else 0)
            fator_total = diferencial_juros + forca_reversao + resistencia + choque_volatilidade + sazonalidade
            variacao_maxima = 0.02
        elif tipo_analise == 'HORARIA':
            diferencial_juros = ((parametros['juros_brasil'] - parametros['juros_exterior']) / 100) / (365*24)
            desvio_da_media = (media_historica - valor_atual) / desvio_historico
            forca_reversao = desvio_da_media * 0.001
            resistencia = -0.0001 if valor_atual > valor_maximo_historico * 0.99 else 0
            choque_volatilidade = np.random.normal(0, parametros['volatilidade_horaria'])
            hora_atual = (dados_historicos.index[-1].hour + periodo) % 24
            sazonalidade = 0.0003 if hora_atual in [8,9] else (-0.0002 if hora_atual in [16,17] else 0)
            fator_total = diferencial_juros + forca_reversao + resistencia + choque_volatilidade + sazonalidade
            variacao_maxima = 0.005
        else:
            diferencial_juros = ((parametros['juros_brasil'] - parametros['juros_exterior']) / 100) / 12
            desvio_da_media = (media_historica - valor_atual) / desvio_historico
            forca_reversao = desvio_da_media * 0.005
            resistencia = -0.0005 if valor_atual > valor_maximo_historico * 0.98 else 0
            choque_volatilidade = np.random.normal(0, parametros['volatilidade']/300)
            mes_atual = (dados_historicos.index[-1].month + periodo) % 12
            sazonalidade = 0.001 if mes_atual in [5,6,7] else (-0.001 if mes_atual in [10,11] else 0)
            fator_total = (diferencial_juros * 0.7) + (forca_reversao * 0.3) + resistencia + choque_volatilidade + sazonalidade
            variacao_maxima = 0.02
        fator_total = np.clip(fator_total, -variacao_maxima, variacao_maxima)
        valor_atual = valor_atual * (1 + fator_total)
        if tipo_analise == 'HORARIA':
            limite_queda = -0.03
        else:
            limite_queda = -0.20
        variacao_total = (valor_atual - dados_historicos['Cotação'].iloc[-1]) / dados_historicos['Cotação'].iloc[-1]
        if variacao_total < limite_queda:
            valor_atual = dados_historicos['Cotação'].iloc[-1] * (1 + limite_queda)
        projecao.append(valor_atual)
    return projecao

# ---------- MONTE CARLO ----------
def simulacao_monte_carlo(dados_historicos, parametros, periodos_projecao, tipo_analise):
    np.random.seed(42)
    returns = dados_historicos['Cotação'].pct_change().dropna()
    simulacoes = []
    for _ in range(parametros['n_simulacoes']):
        previsao = [dados_historicos['Cotação'].iloc[-1]]
        for _ in range(periodos_projecao):
            retorno = np.random.normal(returns.mean(), returns.std())
            previsao.append(previsao[-1] * (1 + retorno))
        simulacoes.append(previsao[1:])
    simulacoes = np.array(simulacoes)
    projecao_mc = pd.Series(simulacoes.mean(axis=0))
    return projecao_mc, simulacoes

# ---------- ENSEMBLE ----------
def criar_ensemble_previsoes(projecao_economica, projecao_arima, projecao_mc, projecao_ml, metricas_ml=None, tipo_analise='HORARIA'):
    projecoes = {}
    modelos_ativos = []
    if len(projecao_economica) > 0:
        projecoes['Economico'] = projecao_economica
        modelos_ativos.append('Economico')
    if not projecao_arima.empty and len(projecao_arima) > 0:
        projecoes['ARIMA'] = projecao_arima
        modelos_ativos.append('ARIMA')
    if not projecao_mc.empty and len(projecao_mc) > 0:
        projecoes['Monte_Carlo'] = projecao_mc
        modelos_ativos.append('Monte_Carlo')
    ml_valido = False
    if projecao_ml and len(projecao_ml) > 0 and not all(np.isnan(x) for x in projecao_ml):
        projecoes['ML'] = projecao_ml
        modelos_ativos.append('ML')
        ml_valido = True
    if tipo_analise == 'HORARIA':
        pesos_base = {'Economico': 0.35, 'Monte_Carlo': 0.30, 'ARIMA': 0.25, 'ML': 0.10}
    elif tipo_analise == 'DIARIA':
        pesos_base = {'Economico': 0.30, 'Monte_Carlo': 0.25, 'ARIMA': 0.25, 'ML': 0.20}
    else:
        pesos_base = {'Economico': 0.40, 'Monte_Carlo': 0.40, 'ARIMA': 0.29, 'ML': 0.01}
    if not ml_valido:
        peso_redistribuir = pesos_base['ML']
        for modelo in ['Economico','Monte_Carlo','ARIMA']:
            if modelo in modelos_ativos:
                pesos_base[modelo] += peso_redistribuir / 3
    pesos_finais = {}
    for modelo in modelos_ativos:
        if modelo == 'ML' and ml_valido:
            pesos_finais[modelo] = 0.25 if tipo_analise == 'MENSAL' else (0.15 if tipo_analise == 'DIARIA' else 0.10)
        else:
            pesos_finais[modelo] = pesos_base.get(modelo, 0.25)
    total = sum(pesos_finais.values())
    pesos_finais = {m: p/total for m,p in pesos_finais.items()}
    ensemble = []
    n_periodos = len(projecao_economica)
    for i in range(n_periodos):
        valor = 0
        for modelo, serie in projecoes.items():
            if isinstance(serie, pd.Series):
                valor += serie.iloc[i] * pesos_finais[modelo]
            else:
                valor += serie[i] * pesos_finais[modelo]
        ensemble.append(valor)
    return ensemble, pesos_finais

# ---------- MÉTRICAS ----------
def calcular_metricas(dados_historicos, projecao_ensemble, tipo_analise):
    returns = dados_historicos['Cotação'].pct_change().dropna()
    if tipo_analise == 'DIARIA':
        metricas = {
            'valor_atual': dados_historicos['Cotação'].iloc[-1],
            'volatilidade': returns.std() * np.sqrt(252) * 100,
            'retorno_periodo': (dados_historicos['Cotação'].iloc[-1] / dados_historicos['Cotação'].iloc[-21] - 1) * 100 if len(dados_historicos) > 21 else 0,
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'volatilidade_label': 'Volatilidade Anual',
            'periodo_analise': 'Retorno 20 dias'
        }
    elif tipo_analise == 'HORARIA':
        metricas = {
            'valor_atual': dados_historicos['Cotação'].iloc[-1],
            'volatilidade': returns.std() * np.sqrt(24*365) * 100,
            'retorno_periodo': (dados_historicos['Cotação'].iloc[-1] / dados_historicos['Cotação'].iloc[-24] - 1) * 100 if len(dados_historicos) > 24 else 0,
            'sharpe_ratio': (returns.mean() * (24*365)) / (returns.std() * np.sqrt(24*365)) if returns.std() > 0 else 0,
            'volatilidade_label': 'Volatilidade Anual',
            'periodo_analise': 'Retorno 24 horas'
        }
    else:
        metricas = {
            'valor_atual': dados_historicos['Cotação'].iloc[-1],
            'volatilidade': returns.std() * np.sqrt(12) * 100,
            'retorno_periodo': (dados_historicos['Cotação'].iloc[-1] / dados_historicos['Cotação'].iloc[-13] - 1) * 100 if len(dados_historicos) > 12 else 0,
            'sharpe_ratio': (returns.mean() * 12) / (returns.std() * np.sqrt(12)) if returns.std() > 0 else 0,
            'volatilidade_label': 'Volatilidade Anual',
            'periodo_analise': 'Retorno 12 meses'
        }
    if projecao_ensemble:
        metricas['variacao_projetada'] = ((projecao_ensemble[-1] / dados_historicos['Cotação'].iloc[-1]) - 1) * 100
    return metricas

# ---------- VISUALIZAÇÃO ----------
def gerar_grafico(dados_historicos, projecao_arima, projecao_economica, projecao_mc, projecao_ml, projecao_ensemble, datas_projecao, parametros, par_moedas, tipo_analise, simulacoes):
    fig = plt.figure(constrained_layout=True, figsize=(20, 16))
    gs = fig.add_gridspec(4, 3)

    # Gráfico principal
    ax_principal = fig.add_subplot(gs[:2, :])
    if tipo_analise == 'DIARIA':
        dados_recentes = dados_historicos.tail(60)
        dados_recentes['Cotação'].plot(ax=ax_principal, color='navy', label='Histórico (60 dias)', linewidth=2)
        if 'MM20' in dados_recentes.columns:
            dados_recentes['MM20'].plot(ax=ax_principal, color='green', label='MM 20 Dias', linestyle='--', linewidth=1.5)
    elif tipo_analise == 'HORARIA':
        dados_recentes = dados_historicos.tail(48)
        dados_recentes['Cotação'].plot(ax=ax_principal, color='navy', label='Histórico (48 horas)', linewidth=2)
        if 'MM12' in dados_recentes.columns:
            dados_recentes['MM12'].plot(ax=ax_principal, color='green', label='MM 12 Horas', linestyle='--', linewidth=1.5)
    else:
        dados_historicos['Cotação'].plot(ax=ax_principal, color='navy', label='Histórico', linewidth=2.5)
        if 'MM12' in dados_historicos.columns:
            dados_historicos['MM12'].plot(ax=ax_principal, color='green', label='MM 12M', linestyle='--', linewidth=2)

    # Projeções
    if not projecao_arima.empty:
        projecao_arima_series = pd.Series(projecao_arima, index=datas_projecao)
        projecao_arima_series.plot(ax=ax_principal, color='orange', marker='s', label='ARIMA', linestyle=':', markersize=4)
    projecao_economica_series = pd.Series(projecao_economica, index=datas_projecao)
    projecao_economica_series.plot(ax=ax_principal, color='red', marker='o', label='Econômica', linestyle='--', markersize=4)
    projecao_mc_series = pd.Series(projecao_mc, index=datas_projecao)
    projecao_mc_series.plot(ax=ax_principal, color='purple', marker='^', label='Monte Carlo', linestyle='-.', markersize=4)
    if projecao_ml:
        projecao_ml_series = pd.Series(projecao_ml, index=datas_projecao)
        projecao_ml_series.plot(ax=ax_principal, color='brown', marker='D', label='Machine Learning', linestyle='-', markersize=4)
    if projecao_ensemble:
        projecao_ensemble_series = pd.Series(projecao_ensemble, index=datas_projecao)
        projecao_ensemble_series.plot(ax=ax_principal, color='gold', marker='*', label='ENSEMBLE', linestyle='-', linewidth=3, markersize=6)

    periodo_texto = "DIAS" if tipo_analise == "DIARIA" else ("HORAS" if tipo_analise == "HORARIA" else "MESES")
    ax_principal.set_title(f'ANÁLISE {tipo_analise} {par_moedas} - PROJEÇÕES PARA {len(datas_projecao)} {periodo_texto}',
                           pad=20, fontsize=16, fontweight='bold')
    ax_principal.set_ylabel(f'R$/{parametros["moeda_base"]}', fontsize=12)
    ax_principal.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_principal.grid(True, alpha=0.3)

    # RSI
    ax_rsi = fig.add_subplot(gs[2, 0])
    if 'RSI' in dados_historicos.columns:
        dados_historicos['RSI'].dropna().plot(ax=ax_rsi, color='brown', linewidth=2)
        ax_rsi.axhline(70, color='red', linestyle='--', alpha=0.7, label='Sobrevendido')
        ax_rsi.axhline(30, color='green', linestyle='--', alpha=0.7, label='Sobrecomprado')
        ax_rsi.set_title('RSI', fontsize=12)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.legend()
        ax_rsi.grid(True, alpha=0.3)

    # Distribuição de retornos
    ax_returns = fig.add_subplot(gs[2, 1])
    returns = dados_historicos['Cotação'].pct_change().dropna()
    returns.tail(min(100, len(returns))).hist(ax=ax_returns, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax_returns.set_title('Distribuição dos Retornos', fontsize=12)
    ax_returns.set_xlabel('Retorno')
    ax_returns.set_ylabel('Frequência')
    ax_returns.grid(True, alpha=0.3)

    # Monte Carlo simulações
    ax_mc = fig.add_subplot(gs[2, 2])
    for i in range(min(30, simulacoes.shape[0])):
        ax_mc.plot(datas_projecao, simulacoes[i], alpha=0.05, color='blue')
    ax_mc.plot(datas_projecao, projecao_mc, color='red', linewidth=2, label='Média')
    ax_mc.set_title('Simulações Monte Carlo', fontsize=12)
    ax_mc.legend()
    ax_mc.grid(True, alpha=0.3)

    # Tabela de projeções
    ax_tabela = fig.add_subplot(gs[3, :])
    ax_tabela.axis('off')
    colunas = ['Data', 'Econômica', 'ARIMA', 'Monte Carlo', 'ML', 'ENSEMBLE']
    dados_tabela = []
    for i, data in enumerate(datas_projecao):
        if tipo_analise == 'DIARIA':
            data_str = data.strftime('%d/%m')
        elif tipo_analise == 'HORARIA':
            data_str = data.strftime('%d/%m %H:%M')
        else:
            data_str = data.strftime('%m/%Y')
        linha = [
            data_str,
            f'R$ {projecao_economica[i]:.4f}',
            f'R$ {projecao_arima.iloc[i]:.4f}' if not projecao_arima.empty else 'N/A',
            f'R$ {projecao_mc.iloc[i]:.4f}',
            f'R$ {projecao_ml[i]:.4f}' if projecao_ml else 'N/A',
            f'R$ {projecao_ensemble[i]:.4f}' if projecao_ensemble else 'N/A'
        ]
        dados_tabela.append(linha)
    tabela = ax_tabela.table(cellText=dados_tabela, colLabels=colunas, colColours=['#f0f0f0']*6,
                             cellLoc='center', loc='center', bbox=[0,0,1,1])
    tabela.set_fontsize(8 if tipo_analise=='HORARIA' else 9)
    tabela.scale(1, 1.8 if tipo_analise=='HORARIA' else 1.5)

    plt.suptitle(f'ANÁLISE COMPLETA {par_moedas} - {tipo_analise} COM ML + ENSEMBLE',
                 fontsize=18, fontweight='bold', y=0.98)
    return fig

# ---------- PDF EM BYTES ----------
def gerar_recomendacoes(metricas, dados_historicos):
    recomendacoes = []
    rsi_atual = dados_historicos['RSI'].iloc[-1] if 'RSI' in dados_historicos.columns and not dados_historicos['RSI'].isna().all() else 50
    if rsi_atual > 70:
        recomendacoes.append("• RSI indica SOBRECOMPRA - Considerar redução de exposição")
    elif rsi_atual < 30:
        recomendacoes.append("• RSI indica SOBREVENDA - Oportunidade potencial de entrada")
    else:
        recomendacoes.append("• RSI em zona neutra - Mercado em equilíbrio")
    if metricas.get('volatilidade', 0) > 15:
        recomendacoes.append("• ALTA VOLATILIDADE - Recomendado uso de estratégias de hedge")
    else:
        recomendacoes.append("• Volatilidade moderada - Ambiente de risco controlado")
    variacao_projetada = metricas.get('variacao_projetada', 0)
    if variacao_projetada > 5:
        recomendacoes.append("• TENDÊNCIA DE ALTA - Favorável para posições longas em Real")
    elif variacao_projetada < -5:
        recomendacoes.append("• TENDÊNCIA DE BAIXA - Favorável para posições longas em Dólar/Euro")
    else:
        recomendacoes.append("• TENDÊNCIA LATERAL - Estratégias de range trading podem ser eficazes")
    return '\n'.join(recomendacoes)

def gerar_relatorio_pdf_bytes(fig, dados_historicos, metricas, parametros,
                              projecao_economica, projecao_arima, projecao_mc,
                              projecao_ml, projecao_ensemble, pesos_ensemble,
                              datas_projecao, par_moedas, tipo_analise, metricas_ml=None):
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        # Página 1 - gráfico
        fig.savefig(pdf, format='pdf', bbox_inches='tight', dpi=300)
        # Página 2 - relatório
        plt.figure(figsize=(11.69, 8.27))
        plt.axis('off')
        if tipo_analise == 'DIARIA':
            periodo_texto = "12 DIAS"
        elif tipo_analise == 'HORARIA':
            periodo_texto = "24 HORAS"
        else:
            periodo_texto = "12 MESES"

        valor_arima = f"R$ {projecao_arima.iloc[-1]:.4f}" if not projecao_arima.empty else 'N/A'
        valor_mc = f"R$ {projecao_mc.iloc[-1]:.4f}" if not projecao_mc.empty else 'N/A'
        valor_ml = f"R$ {projecao_ml[-1]:.4f}" if projecao_ml else 'N/A'
        valor_ensemble = f"R$ {projecao_ensemble[-1]:.4f}" if projecao_ensemble else 'N/A'
        variacao_economica = ((projecao_economica[-1] / dados_historicos['Cotação'].iloc[-1]) - 1) * 100
        variacao_ensemble = ((projecao_ensemble[-1] / dados_historicos['Cotação'].iloc[-1]) - 1) * 100 if projecao_ensemble else 0

        texto_ml = ""
        if metricas_ml:
            texto_ml = f"""
MACHINE LEARNING (Random Forest):
• R² Treino: {metricas_ml['train_score']:.4f}
• R² Teste: {metricas_ml['test_score']:.4f}
• RMSE: {metricas_ml['rmse']:.4f}
• MAPE: {metricas_ml['mape']:.2f}%
"""
        texto_ensemble = ""
        if pesos_ensemble:
            texto_ensemble = f"""
ENSEMBLE LEARNING (Média Ponderada):
• Peso Econômico: {pesos_ensemble.get('Economico', 0):.1%}
• Peso ARIMA: {pesos_ensemble.get('ARIMA', 0):.1%}
• Peso Monte Carlo: {pesos_ensemble.get('Monte_Carlo', 0):.1%}
• Peso ML: {pesos_ensemble.get('ML', 0):.1%}
"""
        relatorio_texto = f"""
RELATÓRIO DE ANÁLISE {par_moedas} - {parametros['nome_completo']}
Tipo de Análise: {tipo_analise} - Projeções para {periodo_texto}
Data de geração: {datetime.now().strftime('%d/%m/%Y %H:%M')}

RESUMO EXECUTIVO:
• Período analisado: {dados_historicos.index[0].strftime('%d/%m/%Y')} a {dados_historicos.index[-1].strftime('%d/%m/%Y')}
• Cotação atual: R$ {metricas['valor_atual']:.4f}
• {metricas.get('periodo_analise', 'Retorno 12 meses')}: {metricas.get('retorno_periodo', 0):.2f}%
• {metricas.get('volatilidade_label', 'Volatilidade anual')}: {metricas.get('volatilidade', 0):.2f}%
• Sharpe Ratio: {metricas.get('sharpe_ratio', 0):.2f}

PARÂMETROS UTILIZADOS:
• Juros Brasil (Selic): {parametros['juros_brasil']}%
• Juros {parametros['moeda_base']}: {parametros['juros_exterior']}%
• Inflação Brasil: {parametros['inflacao_brasil']}%
• Inflação {parametros['moeda_base']}: {parametros['inflacao_exterior']}%
• Volatilidade: {parametros.get('volatilidade', parametros.get('volatilidade_diaria', parametros.get('volatilidade_horaria', 0)))}

PROJEÇÕES PARA OS PRÓXIMOS {periodo_texto}:
• Modelo Econômico: R$ {projecao_economica[-1]:.4f} ({variacao_economica:+.2f}%)
• Modelo ARIMA: {valor_arima}
• Monte Carlo: {valor_mc}
• Machine Learning: {valor_ml}
• ENSEMBLE FINAL: {valor_ensemble} ({variacao_ensemble:+.2f}%)
{texto_ml}
{texto_ensemble}
RECOMENDAÇÕES:
{gerar_recomendacoes(metricas, dados_historicos)}

---
Relatório gerado automaticamente - Análise Técnica {par_moedas} {tipo_analise} com ML + Ensemble
"""
        plt.text(0.05, 0.95, relatorio_texto, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace', linespacing=1.5)
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Página 3 - tabela detalhada (opcional, mas já temos no gráfico)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# ---------- FUNÇÃO PRINCIPAL DE ANÁLISE ----------
def run_analysis(tipo_analise, par_moedas, cotacao_atual=None):
    parametros = obter_parametros(tipo_analise, par_moedas)
    if tipo_analise == 'DIARIA':
        periodos_projecao = 12
    elif tipo_analise == 'HORARIA':
        periodos_projecao = 24
    else:
        periodos_projecao = 12

    dados_historicos = criar_dados_historicos(tipo_analise, par_moedas, cotacao_atual)
    dados_historicos = dados_historicos.dropna().to_frame('Cotação')

    # Indicadores
    if tipo_analise == 'DIARIA':
        dados_historicos['MM5'] = dados_historicos['Cotação'].rolling(5).mean()
        dados_historicos['MM20'] = dados_historicos['Cotação'].rolling(20).mean()
        dados_historicos['RSI'] = calculate_rsi(dados_historicos['Cotação'], window=10)
    elif tipo_analise == 'HORARIA':
        dados_historicos['MM4'] = dados_historicos['Cotação'].rolling(4).mean()
        dados_historicos['MM12'] = dados_historicos['Cotação'].rolling(12).mean()
        dados_historicos['RSI'] = calculate_rsi(dados_historicos['Cotação'], window=12)
    else:
        dados_historicos['MM6'] = dados_historicos['Cotação'].rolling(6).mean()
        dados_historicos['MM12'] = dados_historicos['Cotação'].rolling(12).mean()
        dados_historicos['RSI'] = calculate_rsi(dados_historicos['Cotação'])

    # Machine Learning
    projecao_ml = []
    datas_projecao = []
    metricas_ml = None
    try:
        dados_com_features = criar_features_ml(dados_historicos, parametros, tipo_analise)
        modelo_ml, feature_columns, metricas_ml = treinar_modelo_ml(dados_com_features, parametros, tipo_analise)
        projecao_ml, datas_projecao = prever_com_ml(modelo_ml, feature_columns, dados_historicos, parametros, periodos_projecao, tipo_analise)
    except Exception as e:
        # Fallback: gerar datas
        if tipo_analise == 'DIARIA':
            ultima_data = dados_historicos.index[-1]
            datas_projecao = []
            current_date = ultima_data
            for i in range(periodos_projecao):
                current_date += timedelta(days=1)
                while current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                datas_projecao.append(current_date)
        elif tipo_analise == 'HORARIA':
            ultima_data = dados_historicos.index[-1]
            datas_projecao = []
            current_date = ultima_data
            horas_geradas = 0
            while horas_geradas < periodos_projecao:
                current_date += timedelta(hours=1)
                if current_date.weekday() < 5:
                    datas_projecao.append(current_date)
                    horas_geradas += 1
        else:
            datas_projecao = [dados_historicos.index[-1] + relativedelta(months=i) for i in range(1, periodos_projecao+1)]
        projecao_ml = [dados_historicos['Cotação'].iloc[-1]] * len(datas_projecao)

    periodos_reais = len(datas_projecao)

    # ARIMA
    projecao_arima = pd.Series(dtype=float)
    if len(dados_historicos) > 24:
        try:
            modelo_arima = ARIMA(dados_historicos['Cotação'].dropna(), order=parametros['arima_order'])
            modelo_ajustado = modelo_arima.fit()
            previsao_arima = modelo_ajustado.get_forecast(steps=periodos_reais)
            projecao_arima = previsao_arima.predicted_mean
        except:
            pass

    # Monte Carlo
    projecao_mc, simulacoes = simulacao_monte_carlo(dados_historicos, parametros, periodos_reais, tipo_analise)

    # Projeção Econômica
    projecao_economica = calcular_projecao_economica(dados_historicos, parametros, periodos_reais, tipo_analise)

    # Ensemble
    projecao_ensemble, pesos_ensemble = criar_ensemble_previsoes(
        projecao_economica, projecao_arima, projecao_mc, projecao_ml, metricas_ml, tipo_analise
    )

    # Métricas
    metricas = calcular_metricas(dados_historicos, projecao_ensemble, tipo_analise)

    # Gráfico
    fig = gerar_grafico(dados_historicos, projecao_arima, projecao_economica, projecao_mc,
                        projecao_ml, projecao_ensemble, datas_projecao, parametros,
                        par_moedas, tipo_analise, simulacoes)

    return {
        'fig': fig,
        'dados_historicos': dados_historicos,
        'metricas': metricas,
        'parametros': parametros,
        'projecao_economica': projecao_economica,
        'projecao_arima': projecao_arima,
        'projecao_mc': projecao_mc,
        'projecao_ml': projecao_ml,
        'projecao_ensemble': projecao_ensemble,
        'pesos_ensemble': pesos_ensemble,
        'datas_projecao': datas_projecao,
        'par_moedas': par_moedas,
        'tipo_analise': tipo_analise,
        'metricas_ml': metricas_ml
    }

# ---------- STREAMLIT UI ----------
def main():
    st.title("💱 Projeção de Pares de Moedas com Machine Learning e Ensemble")
    st.markdown("---")

    # Sidebar de configuração
    st.sidebar.header("⚙️ Configurações da Análise")
    tipo_analise, periodos, periodo_texto, freq = selecionar_tipo_analise()
    par_moedas = selecionar_par_moedas()

    st.sidebar.markdown("---")
    st.sidebar.header("💰 Cotação Atual")
    
    # Busca automática de cotação - sem chave hardcoded
    api_key = st.sidebar.text_input("API Key Twelve Data", type="password", placeholder="Deixe em branco para usar a chave padrão")
    buscar_auto = st.sidebar.checkbox("Buscar cotação automaticamente", value=True)
    
    cotacao_inicial = None
    if buscar_auto:
        with st.sidebar.status("Buscando cotação..."):
            cotacao_twelve = buscar_cotacao_twelve(par_moedas, api_key)
            if cotacao_twelve:
                cotacao_inicial = cotacao_twelve
                st.sidebar.success(f"✅ Twelve Data: R$ {cotacao_twelve:.4f}")
            else:
                cotacao_yahoo = buscar_cotacao_yahoo(par_moedas)
                if cotacao_yahoo:
                    cotacao_inicial = cotacao_yahoo
                    st.sidebar.success(f"✅ Yahoo Finance: R$ {cotacao_yahoo:.4f}")
                else:
                    st.sidebar.warning("⚠️ Não foi possível obter cotação automática. Usando valor padrão.")
    
    # Opção de entrada manual
    usar_manual = st.sidebar.checkbox("Inserir cotação manualmente", value=not buscar_auto)
    if usar_manual:
        valor_manual = st.sidebar.number_input(
            f"Digite a cotação {par_moedas}",
            min_value=0.0,
            max_value=100.0,
            value=5.2333 if par_moedas=="USD/BRL" else (6.22 if par_moedas=="EUR/BRL" else (7.85 if par_moedas=="GBP/BRL" else 0.033)),
            step=0.0001,
            format="%.4f"
        )
        cotacao_inicial = valor_manual

    st.sidebar.markdown("---")
    executar = st.sidebar.button("🚀 Executar Análise", type="primary", use_container_width=True)

    # Área principal
    if executar:
        with st.spinner("🔄 Processando análise... Isso pode levar alguns segundos."):
            try:
                resultado = run_analysis(tipo_analise, par_moedas, cotacao_inicial)
                
                # Armazenar no session_state para não perder ao interagir
                st.session_state['resultado'] = resultado
                st.session_state['analise_executada'] = True
            except Exception as e:
                st.error(f"❌ Erro durante a análise: {e}")
                st.exception(e)
                st.stop()

    if 'analise_executada' in st.session_state and st.session_state['analise_executada']:
        res = st.session_state['resultado']
        
        # Métricas em colunas
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Cotação Atual", f"R$ {res['metricas']['valor_atual']:.4f}")
        with col2:
            st.metric("📈 Projeção Ensemble", f"R$ {res['projecao_ensemble'][-1]:.4f}",
                     delta=f"{res['metricas'].get('variacao_projetada', 0):+.2f}%")
        with col3:
            st.metric("📊 Volatilidade", f"{res['metricas']['volatilidade']:.2f}%")
        with col4:
            st.metric("📋 Sharpe Ratio", f"{res['metricas']['sharpe_ratio']:.2f}")

        # Gráfico
        st.subheader("📈 Projeções dos Modelos")
        st.pyplot(res['fig'])

        # Tabela de projeções
        st.subheader("📅 Tabela de Projeções Detalhada")
        df_tabela = pd.DataFrame({
            'Data': [d.strftime('%d/%m/%Y %H:%M') if res['tipo_analise']=='HORARIA' else d.strftime('%d/%m/%Y') if res['tipo_analise']=='DIARIA' else d.strftime('%b/%Y') for d in res['datas_projecao']],
            'Econômica (R$)': [f"{v:.4f}" for v in res['projecao_economica']],
            'ARIMA (R$)': [f"{v:.4f}" if not pd.isna(v) else 'N/A' for v in res['projecao_arima']] if not res['projecao_arima'].empty else ['N/A']*len(res['datas_projecao']),
            'Monte Carlo (R$)': [f"{v:.4f}" for v in res['projecao_mc']],
            'Machine Learning (R$)': [f"{v:.4f}" for v in res['projecao_ml']] if res['projecao_ml'] else ['N/A']*len(res['datas_projecao']),
            'Ensemble (R$)': [f"{v:.4f}" for v in res['projecao_ensemble']],
            'Variação %': [f"{((res['projecao_ensemble'][i]/res['metricas']['valor_atual'])-1)*100:+.2f}%" for i in range(len(res['datas_projecao']))]
        })
        st.dataframe(df_tabela, use_container_width=True)

        # Métricas do ML
        if res['metricas_ml']:
            with st.expander("🤖 Métricas do Modelo Machine Learning"):
                col_ml1, col_ml2, col_ml3, col_ml4 = st.columns(4)
                col_ml1.metric("R² Treino", f"{res['metricas_ml']['train_score']:.4f}")
                col_ml2.metric("R² Teste", f"{res['metricas_ml']['test_score']:.4f}")
                col_ml3.metric("RMSE", f"{res['metricas_ml']['rmse']:.4f}")
                col_ml4.metric("MAPE", f"{res['metricas_ml']['mape']:.2f}%")

        # Pesos do Ensemble
        if res['pesos_ensemble']:
            with st.expander("⚖️ Pesos do Ensemble"):
                for modelo, peso in res['pesos_ensemble'].items():
                    st.write(f"- **{modelo}**: {peso:.1%}")

        # Botão de download do PDF
        st.subheader("📄 Relatório em PDF")
        pdf_bytes = gerar_relatorio_pdf_bytes(
            res['fig'], res['dados_historicos'], res['metricas'], res['parametros'],
            res['projecao_economica'], res['projecao_arima'], res['projecao_mc'],
            res['projecao_ml'], res['projecao_ensemble'], res['pesos_ensemble'],
            res['datas_projecao'], res['par_moedas'], res['tipo_analise'], res['metricas_ml']
        )
        st.download_button(
            label="📥 Baixar Relatório PDF",
            data=pdf_bytes,
            file_name=f"Relatorio_{res['par_moedas'].replace('/', '_')}_{res['tipo_analise']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    else:
        st.info("👈 Configure os parâmetros na barra lateral e clique em **Executar Análise** para começar.")

if __name__ == "__main__":
    main()
