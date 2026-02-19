# ========== BLOCO 1: IMPORTAÇÕES ==========
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import os
import subprocess
import sys
import requests  # ✅ ADICIONADO para o Twelve Data
import time 
import random


# GARANTIR que yfinance está instalado e importado CORRETAMENTE
try:
    import yfinance as yf
    print("✅ yfinance importado com sucesso")
except ImportError:
    print("📦 yfinance não encontrado. Instalando...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
        import yfinance as yf
        print("✅ yfinance instalado e importado com sucesso")
    except Exception as e:
        print(f"❌ Falha na instalação do yfinance: {e}")
        print("⚠️  O sistema usará apenas entrada manual")
        # Criar um objeto dummy para evitar erros
        class YFDummy:
            def Ticker(self, symbol):
                return self
            def history(self, **kwargs):
                class HistoryDummy:
                    @property
                    def empty(self):
                        return True
                return HistoryDummy()
        yf = YFDummy()

# Verificar se requests está instalado (normalmente já vem com Python)
try:
    import requests
    print("✅ requests importado com sucesso")
except ImportError:
    print("📦 requests não encontrado. Instalando...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests
        print("✅ requests instalado e importado com sucesso")
    except Exception as e:
        print(f"❌ Falha na instalação do requests: {e}")
        print("⚠️  Twelve Data não estará disponível")

warnings.filterwarnings('ignore')

# ========== BLOCO 2: CONFIGURAÇÕES GERAIS ==========
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ========== BLOCO 3: SELEÇÃO DO TIPO DE ANÁLISE ==========
def selecionar_tipo_analise():
    """Permite ao usuário escolher entre análise mensal, diária ou horária"""
    print("⏰ SELECIONE O TIPO DE ANÁLISE:")
    print("═" * 50)
    print("1. MENSAL - Projeções para 12 MESES (Longo Prazo)")
    print("2. DIÁRIA - Projeções para 12 DIAS (Curto Prazo)")
    print("3. HORÁRIA - Projeções para 24 HORAS (Intraday)")
    print("═" * 50)
    
    opcao = input("🎯 Digite o número da opção desejada (1-3): ").strip()
    
    if opcao == '2':
        return 'DIARIA', 12, 'dias', 'D'
    elif opcao == '3':
        return 'HORARIA', 24, 'horas', 'H'
    else:
        return 'MENSAL', 12, 'meses', 'ME'

def selecionar_par_moedas():
    """Permite ao usuário escolher qual par analisar"""
    print("\n🌍 SELECIONE O PAR DE MOEDAS PARA ANÁLISE:")
    print("═" * 50)
    print("1. USD/BRL (Dólar Americano vs Real Brasileiro)")
    print("2. EUR/BRL (Euro vs Real Brasileiro)") 
    print("3. GBP/BRL (Libra Esterlina vs Real)")
    print("4. JPY/BRL (Iene Japonês vs Real Brasileiro)")
    print("═" * 50)
    
    opcao = input("🎯 Digite o número da opção desejada (1-4): ").strip()
    
    pares = {
        '1': 'USD/BRL',
        '2': 'EUR/BRL', 
        '3': 'GBP/BRL',
        '4': 'JPY/BRL'
    }
    
    par_selecionado = pares.get(opcao, 'USD/BRL')
    print(f"✅ Par selecionado: {par_selecionado}")
    return par_selecionado

# ========== BLOCO 4: PARÂMETROS ESPECÍFICOS ==========
def obter_parametros(tipo_analise, par_moedas):
    """Retorna parâmetros específicos para cada tipo de análise e par de moedas - DADOS REAIS OUTUBRO 2025"""
    parametros_base = {
        'MENSAL': {
            'USD/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 4.25,  # DADOS REAIS 2025
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.80,  # DADOS REAIS 2025
                'volatilidade': 2.1, 'commodity_weight': 0.25,
                'arima_order': (2,1,2), 'periodos_projecao': 12,
                'n_simulacoes': 10000, 'moeda_base': 'USD',
                'moeda_contraria': 'BRL', 'nome_completo': 'Dólar Americano vs Real Brasileiro'
            },
            'EUR/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 2.15,  # DADOS REAIS 2025
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.20,  # DADOS REAIS 2025
                'volatilidade': 1.8, 'commodity_weight': 0.18,
                'arima_order': (2,1,2), 'periodos_projecao': 12,
                'n_simulacoes': 10000, 'moeda_base': 'EUR',
                'moeda_contraria': 'BRL', 'nome_completo': 'Euro vs Real Brasileiro'
            },
            'GBP/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 4.00,  # DADOS REAIS 2025
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.50,  # DADOS REAIS 2025
                'volatilidade': 1.9, 'commodity_weight': 0.15,
                'arima_order': (2,1,2), 'periodos_projecao': 12,
                'n_simulacoes': 10000, 'moeda_base': 'GBP',
                'moeda_contraria': 'BRL', 'nome_completo': 'Libra Esterlina vs Real Brasileiro'
            },
            'JPY/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 0.50,  # DADOS REAIS 2025
                'inflacao_brasil': 4.80, 'inflacao_exterior': 1.50,  # DADOS REAIS 2025
                'volatilidade': 1.5, 'commodity_weight': 0.08,
                'arima_order': (2,1,2), 'periodos_projecao': 12,
                'n_simulacoes': 10000, 'moeda_base': 'JPY',
                'moeda_contraria': 'BRL', 'nome_completo': 'Iene Japonês vs Real Brasileiro'
            }
        },
        'DIARIA': {
            'USD/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 4.25,  # DADOS REAIS 2025
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.80,  # DADOS REAIS 2025
                'volatilidade_diaria': 0.015, 'commodity_weight': 0.25,
                'arima_order': (1,1,1), 'periodos_projecao': 12,
                'n_simulacoes': 5000, 'moeda_base': 'USD',
                'moeda_contraria': 'BRL', 'nome_completo': 'Dólar Americano vs Real Brasileiro'
            },
            'EUR/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 2.15,  # DADOS REAIS 2025
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.20,  # DADOS REAIS 2025
                'volatilidade_diaria': 0.012, 'commodity_weight': 0.18,
                'arima_order': (1,1,1), 'periodos_projecao': 12,
                'n_simulacoes': 5000, 'moeda_base': 'EUR',
                'moeda_contraria': 'BRL', 'nome_completo': 'Euro vs Real Brasileiro'
            },
            'GBP/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 4.00,  # DADOS REAIS 2025
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.50,  # DADOS REAIS 2025
                'volatilidade_diaria': 0.013, 'commodity_weight': 0.15,
                'arima_order': (1,1,1), 'periodos_projecao': 12,
                'n_simulacoes': 5000, 'moeda_base': 'GBP',
                'moeda_contraria': 'BRL', 'nome_completo': 'Libra Esterlina vs Real Brasileiro'
            },
            'JPY/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 0.50,  # DADOS REAIS 2025
                'inflacao_brasil': 4.80, 'inflacao_exterior': 1.50,  # DADOS REAIS 2025
                'volatilidade_diaria': 0.010, 'commodity_weight': 0.08,
                'arima_order': (1,1,1), 'periodos_projecao': 12,
                'n_simulacoes': 5000, 'moeda_base': 'JPY',
                'moeda_contraria': 'BRL', 'nome_completo': 'Iene Japonês vs Real Brasileiro'
            }
        },
        'HORARIA': {
            'USD/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 4.25,  # DADOS REAIS 2025
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.80,  # DADOS REAIS 2025
                'volatilidade_horaria': 0.002, 'commodity_weight': 0.25,
                'arima_order': (1,1,0), 'periodos_projecao': 24,
                'n_simulacoes': 2000, 'moeda_base': 'USD',
                'moeda_contraria': 'BRL', 'nome_completo': 'Dólar Americano vs Real Brasileiro'
            },
            'EUR/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 2.15,  # DADOS REAIS 2025
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.20,  # DADOS REAIS 2025
                'volatilidade_horaria': 0.0015, 'commodity_weight': 0.18,
                'arima_order': (1,1,0), 'periodos_projecao': 24,
                'n_simulacoes': 2000, 'moeda_base': 'EUR',
                'moeda_contraria': 'BRL', 'nome_completo': 'Euro vs Real Brasileiro'
            },
            'GBP/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 4.00,  # DADOS REAIS 2025
                'inflacao_brasil': 4.80, 'inflacao_exterior': 2.50,  # DADOS REAIS 2025
                'volatilidade_horaria': 0.0018, 'commodity_weight': 0.15,
                'arima_order': (1,1,0), 'periodos_projecao': 24,
                'n_simulacoes': 2000, 'moeda_base': 'GBP',
                'moeda_contraria': 'BRL', 'nome_completo': 'Libra Esterlina vs Real Brasileiro'
            },
            'JPY/BRL': {
                'juros_brasil': 15.00, 'juros_exterior': 0.50,  # DADOS REAIS 2025
                'inflacao_brasil': 4.80, 'inflacao_exterior': 1.50,  # DADOS REAIS 2025
                'volatilidade_horaria': 0.0010, 'commodity_weight': 0.08,
                'arima_order': (1,1,0), 'periodos_projecao': 24,
                'n_simulacoes': 2000, 'moeda_base': 'JPY',
                'moeda_contraria': 'BRL', 'nome_completo': 'Iene Japonês vs Real Brasileiro'
            }
        }
    }
    
    return parametros_base[tipo_analise].get(par_moedas, parametros_base[tipo_analise]['USD/BRL'])

# ========== BLOCO 5: DADOS HISTÓRICOS ==========
def criar_dados_historicos(tipo_analise, par_moedas, cotacao_atual=None):
    """Cria dados históricos de acordo com o tipo de análise"""
    
    # Se não foi passada cotação atual, usar valor padrão baseado no par de moedas
    if cotacao_atual is None:
        cotacoes_padrao = {
            'USD/BRL': 5.2333,
            'EUR/BRL': 6.2200, 
            'GBP/BRL': 7.8500,
            'JPY/BRL': 0.0330  # Valor padrão para JPY/BRL
        }
        cotacao_atual = cotacoes_padrao.get(par_moedas, 5.2333)
        print(f"🔧 Usando cotação padrão: R$ {cotacao_atual:.4f}")
    
    if tipo_analise == 'DIARIA':
        print(f"📊 Gerando dados históricos DIÁRIOS para {par_moedas}...")
        
        dates = pd.date_range(start='2022-01-01', end=datetime.now(), freq='D')
        dates = dates[dates.dayofweek < 5]  # Apenas dias úteis
        n_periods = len(dates)
        np.random.seed(42)
        
        # GERAR DADOS BASEADOS NA COTAÇÃO ATUAL
        base_value = cotacao_atual * 0.95  # Começar 5% abaixo da cotação atual
        valores = [base_value]
        
        for i in range(1, n_periods):
            # Variação diária típica
            variacao = np.random.normal(0, 0.008)
            # Tendência de longo prazo (levemente positiva)
            tendencia = 0.0001
            # Sazonalidade semanal
            dia_semana = dates[i].weekday()
            if dia_semana == 0:  # Segunda
                sazonalidade = 0.001
            elif dia_semana == 4:  # Sexta
                sazonalidade = -0.001
            else:
                sazonalidade = 0
            
            novo_valor = valores[-1] * (1 + variacao + tendencia + sazonalidade)
            valores.append(novo_valor)
        
        # Garantir que o último valor seja exatamente a cotação atual
        valores[-1] = cotacao_atual
        
        serie = pd.Series(valores, index=dates, name='Cotação')
        print(f"✅ Dados históricos DIÁRIOS gerados: {len(serie)} dias úteis")
        print(f"🎯 Último valor (cotação atual): R$ {serie.iloc[-1]:.4f}")
        
    elif tipo_analise == 'HORARIA':
        print(f"📊 Gerando dados históricos HORÁRIOS para {par_moedas}...")
        
        # Gerar últimos 30 dias em frequência horária (24 horas por dia, apenas dias úteis)
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                             end=datetime.now(), freq='H')
        
        # CORREÇÃO: Apenas dias úteis (segunda a sexta), 24 horas por dia - FOREX
        dates = dates[dates.dayofweek < 5]  # Dias úteis apenas
        
        n_periods = len(dates)
        np.random.seed(42)
        
        # Valor base baseado na cotação atual
        base_value = cotacao_atual * 0.98 if cotacao_atual else 5.2
        valores = [base_value]
        
        for i in range(1, n_periods):
            # Variação horária típica (menor que diária)
            variacao = np.random.normal(0, 0.0008)
            
            # Tendência de muito curto prazo
            tendencia = 0.00002
            
            # Sazonalidade intraday - padrões de trading globais do Forex
            hora = dates[i].hour
            if hora in [0, 1, 2]:  # Madrugada Asia
                sazonalidade = 0.0001
            elif hora in [8, 9, 10]:  # Abertura Europa
                sazonalidade = 0.0003
            elif hora in [12, 13, 14]:  # Sobreposição Europa/América
                sazonalidade = 0.0004
            elif hora in [17, 18, 19]:  # Fechamento América
                sazonalidade = -0.0002
            else:
                sazonalidade = 0
            
            novo_valor = valores[-1] * (1 + variacao + tendencia + sazonalidade)
            valores.append(novo_valor)
        
        # Garantir que o último valor seja exatamente a cotação atual
        valores[-1] = cotacao_atual
        
        serie = pd.Series(valores, index=dates, name='Cotação')
        print(f"✅ Dados históricos HORÁRIOS gerados: {len(serie)} horas (24h/dia útil)")
        print(f"🎯 Último valor (cotação atual): R$ {serie.iloc[-1]:.4f}")
        
    else:
        print(f"📊 Gerando dados históricos MENSAlS para {par_moedas}...")
        
        dates = pd.date_range(start='2015-01-01', end=datetime.now(), freq='ME')
        n_periods = len(dates)
        np.random.seed(42)
        
        # GERAR DADOS BASEADOS NA COTAÇÃO ATUAL
        base_value = cotacao_atual * 0.7  # Começar mais baixo para mostrar tendência de alta
        valores = [base_value]
        
        for i in range(1, n_periods):
            # Variação mensal típica
            variacao = np.random.normal(0, 0.03)
            # Tendência de longo prazo
            tendencia = 0.002
            # Sazonalidade anual
            mes = dates[i].month
            if mes in [7, 8]:  # Meio do ano
                sazonalidade = 0.005
            elif mes in [1, 2]:  # Começo do ano
                sazonalidade = -0.003
            else:
                sazonalidade = 0
            
            novo_valor = valores[-1] * (1 + variacao + tendencia + sazonalidade)
            valores.append(novo_valor)
        
        # Garantir que o último valor seja exatamente a cotação atual
        valores[-1] = cotacao_atual
        
        serie = pd.Series(valores, index=dates, name='Cotação')
        print(f"✅ Dados históricos MENSAlS gerados: {len(serie)} meses")
        print(f"🎯 Último valor (cotação atual): R$ {serie.iloc[-1]:.4f}")
    
    return serie

# ========== BLOCO 6: FUNÇÕES AUXILIARES ==========
def calculate_rsi(series, window=14):
    """Calcula o Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ========== BLOCO 7: ATUALIZAÇÃO DE COTAÇÃO ==========
def buscar_cotacao_twelve(par_moedas, api_key):
    """Busca a cotação atual do Twelve Data"""
    try:
        # Mapeamento dos pares para símbolos Twelve Data
        simbolos = {
            'USD/BRL': 'USD/BRL',
            'EUR/BRL': 'EUR/BRL', 
            'GBP/BRL': 'GBP/BRL',
            'JPY/BRL': 'JPY/BRL'
        }
        
        simbolo = simbolos.get(par_moedas)
        if not simbolo:
            print(f"❌ Par {par_moedas} não mapeado para Twelve Data")
            return None
        
        print(f"🔍 Buscando cotação {par_moedas} no Twelve Data...")
        
        # Construir a URL da API
        url = f"https://api.twelvedata.com/price?symbol={simbolo}&apikey={api_key}"
        
        # Fazer a requisição
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ Erro na API Twelve Data: Status {response.status_code}")
            return None
            
        dados = response.json()
        
        # Verificar se a resposta contém o preço
        if 'price' not in dados:
            print(f"❌ Resposta inesperada da API: {dados}")
            return None
            
        cotacao = float(dados['price'])
        
        # Validação de sanidade - verificar se está dentro de range razoável
        ranges_validos = {
            'USD/BRL': (3.0, 7.0),
            'EUR/BRL': (4.0, 8.0),
            'GBP/BRL': (5.0, 10.0),
            'JPY/BRL': (0.02, 0.05)
        }
        
        min_val, max_val = ranges_validos[par_moedas]
        if not (min_val < cotacao < max_val):
            print(f"⚠️  Cotação Twelve Data fora do range esperado: R$ {cotacao:.4f}")
            return None
        
        print(f"✅ Twelve Data: {par_moedas} = R$ {cotacao:.4f}")
        return cotacao
        
    except Exception as e:
        print(f"❌ Erro ao buscar cotação Twelve Data: {e}")
        return None

def buscar_cotacao_yahoo(par_moedas):
    """Busca a cotação atual do Yahoo Finance (agora como fallback)"""
    try:
        # Mapeamento dos pares para símbolos Yahoo Finance
        simbolos = {
            'USD/BRL': 'USDBRL=X',
            'EUR/BRL': 'EURBRL=X', 
            'GBP/BRL': 'GBPBRL=X',
            'JPY/BRL': 'JPYBRL=X'
        }
        
        simbolo = simbolos.get(par_moedas)
        if not simbolo:
            return None
        
        print(f"🔍 Buscando cotação {par_moedas} no Yahoo Finance...")
        
        # Buscar dados - último dia com intervalo horário
        ticker = yf.Ticker(simbolo)
        dados = ticker.history(period="1d", interval="1h")
        
        if dados.empty:
            return None
            
        # Pegar última cotação disponível
        cotacao = dados['Close'].iloc[-1]
        
        # Validação de sanidade
        ranges_validos = {
            'USD/BRL': (3.0, 7.0),
            'EUR/BRL': (4.0, 8.0),
            'GBP/BRL': (5.0, 10.0),
            'JPY/BRL': (0.02, 0.05)
        }
        
        min_val, max_val = ranges_validos[par_moedas]
        if not (min_val < cotacao < max_val):
            return None
        
        print(f"✅ Yahoo Finance: {par_moedas} = R$ {cotacao:.4f}")
        return cotacao
        
    except Exception as e:
        print(f"❌ Erro no Yahoo Finance: {e}")
        return None

def obter_cotacao_atual(par_moedas):
    """Obtém cotação atual - PRIMEIRO Twelve Data, depois Yahoo, depois manual"""
    
    print(f"\n🎯 ATUALIZAÇÃO DE COTAÇÃO ATUAL - {par_moedas}")
    print("═" * 50)
    
    # API Key do Twelve Data
    API_KEY_TWELVE = "e631d88e2c7348c48d13a061a73c21ab"
    
    # PRIMEIRO: Tentar Twelve Data (fonte principal)
    cotacao_twelve = buscar_cotacao_twelve(par_moedas, API_KEY_TWELVE)
    
    if cotacao_twelve is not None:
        print(f"💡 Cotação Twelve Data: R$ {cotacao_twelve:.4f}")
        usar_twelve = input("🎯 Usar esta cotação? (s/n): ").strip().lower()
        
        if usar_twelve in ['s', 'sim', 'y', 'yes']:
            print(f"✅ Cotação definida: R$ {cotacao_twelve:.4f}")
            return cotacao_twelve
        else:
            print("🔄 Usuário optou por outra fonte...")
    
    # SEGUNDO: Tentar Yahoo Finance (fallback)
    print("🔄 Tentando Yahoo Finance como fallback...")
    cotacao_yahoo = buscar_cotacao_yahoo(par_moedas)
    
    if cotacao_yahoo is not None:
        print(f"💡 Cotação Yahoo Finance: R$ {cotacao_yahoo:.4f}")
        usar_yahoo = input("🎯 Usar esta cotação? (s/n): ").strip().lower()
        
        if usar_yahoo in ['s', 'sim', 'y', 'yes']:
            print(f"✅ Cotação definida: R$ {cotacao_yahoo:.4f}")
            return cotacao_yahoo
        else:
            print("🔄 Usuário optou por entrada manual...")
    else:
        print("ℹ️  Fontes automáticas não disponíveis, usando entrada manual...")
    
    # TERCEIRO: Input manual (fallback final)
    cotacoes_referencia = {
        'USD/BRL': 5.20,
        'EUR/BRL': 6.22,
        'GBP/BRL': 7.85,
        'JPY/BRL': 0.033
    }
    
    print(f"💡 Cotação de referência: R$ {cotacoes_referencia[par_moedas]}")
    print("═" * 50)
    
    while True:
        try:
            cotacao_input = input(f"💵 Digite a cotação {par_moedas} atual: ").replace(',', '.')
            cotacao = float(cotacao_input)
            
            ranges = {
                'USD/BRL': (3.0, 7.0),
                'EUR/BRL': (4.0, 8.0),
                'GBP/BRL': (5.0, 10.0),
                'JPY/BRL': (0.02, 0.05)
            }
            min_val, max_val = ranges[par_moedas]
            
            if min_val < cotacao < max_val:
                print(f"✅ Cotação definida: R$ {cotacao:.4f}")
                return cotacao
            else:
                print(f"⚠️  Valor fora do range esperado ({min_val}-{max_val})")
                
        except ValueError:
            print("⚠️  Digite um valor numérico válido")
            
# ========== BLOCO 8: MACHINE LEARNING ==========
def criar_features_ml(dados_historicos, parametros, tipo_analise):
    """Cria features para o modelo de Machine Learning - VERSÃO ROBUSTA"""
    print(f"🛠️ Criando features para {tipo_analise}...")
    
    df = dados_historicos.copy()
    
    # Garantir que temos dados suficientes
    if len(df) < 10:
        print(f"⚠️  Poucos dados para features: {len(df)} períodos")
        # Retornar features mínimas
        df['lag_1'] = df['Cotação'].shift(1)
        df['retorno_1'] = df['Cotação'].pct_change(1)
        df['MM5'] = df['Cotação'].rolling(5, min_periods=1).mean()
        df = df.fillna(method='bfill').fillna(method='ffill')
        return df
    
    try:
        if tipo_analise == 'DIARIA':
            # Features diárias
            for lag in [1, 2, 3, 5]:
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
            # Features horárias
            for lag in [1, 2, 4, 8]:
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
            
        else:
            # Features mensais - SIMPLIFICADAS para evitar overfitting
            print("🛠️ Criando features MENSAlS simplificadas...")
            
            # Lags básicos
            for lag in [1, 2, 3]:
                df[f'lag_{lag}'] = df['Cotação'].shift(lag)
            
            # Retornos simples
            df['retorno_1m'] = df['Cotação'].pct_change(1)
            df['retorno_3m'] = df['Cotação'].pct_change(3)
            
            # Volatilidade suave
            df['volatilidade_3m'] = df['retorno_1m'].rolling(3, min_periods=1).std()
            
            # Médias móveis
            df['MM6'] = df['Cotação'].rolling(6, min_periods=1).mean()
            df['MM12'] = df['Cotação'].rolling(12, min_periods=1).mean()
            df['MM6_ratio'] = df['Cotação'] / df['MM6']
            
            # RSI
            df['RSI'] = calculate_rsi(df['Cotação'])
            
            # Fatores econômicos básicos
            df['diferencial_juros'] = (parametros['juros_brasil'] - parametros['juros_exterior']) / 100
        
        # Preencher NaN de forma robusta
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        print(f"✅ Features criadas: {len(df.columns)} colunas, {len(df)} linhas")
        return df
        
    except Exception as e:
        print(f"❌ Erro ao criar features: {e}")
        # Fallback: retornar dados básicos
        df_fallback = dados_historicos.copy()
        df_fallback['lag_1'] = df_fallback['Cotação'].shift(1)
        df_fallback['retorno_1'] = df_fallback['Cotação'].pct_change(1)
        return df_fallback.fillna(method='bfill').fillna(method='ffill')

def treinar_modelo_ml(dados_com_features, parametros, tipo_analise):
    """Treina modelo Random Forest para previsão - VERSÃO ROBUSTA"""
    print("🤖 Preparando dados para Machine Learning...")
    
    if dados_com_features.empty:
        print("❌ Dados de features vazios")
        return None, None, None
    
    # Selecionar features
    colunas_excluir = ['Cotação']
    if tipo_analise == 'DIARIA':
        colunas_excluir.extend(['MM5', 'MM20'])
    elif tipo_analise == 'HORARIA':
        colunas_excluir.extend(['MM4', 'MM12'])
    else:
        colunas_excluir.extend(['MM6', 'MM12'])
    
    feature_columns = [col for col in dados_com_features.columns if col not in colunas_excluir]
    
    if len(feature_columns) == 0:
        print("❌ Nenhuma feature disponível")
        return None, None, None
    
    X = dados_com_features[feature_columns]
    y = dados_com_features['Cotação']
    
    print(f"📊 Shape dos dados: X={X.shape}, y={y.shape}")
    
    # Split dos dados
    if len(X) < 10:
        print("❌ Dados insuficientes para treinamento")
        return None, None, None
    
    if tipo_analise == 'HORARIA':
        split_point = max(20, int(len(X) * 0.8))
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    elif tipo_analise == 'MENSAL':
        # Para mensal, usar mais dados para treino (série é menor)
        split_point = max(6, int(len(X) * 0.7))
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    else:
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    print(f"📊 Dados para treino: {len(X_train)} períodos")
    print(f"📊 Dados para teste: {len(X_test)} períodos")
    
    if len(X_train) < 5:
        print("❌ Dados de treino insuficientes")
        return None, None, None
    
    # Configurar modelo - PARÂMETROS ESPECÍFICOS POR TIPO
    if tipo_analise == 'HORARIA':
        model = RandomForestRegressor(
            n_estimators=50, 
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
    elif tipo_analise == 'MENSAL':
        # Modelo mais simples para evitar overfitting em séries mensais
        model = RandomForestRegressor(
            n_estimators=30,
            max_depth=5,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = RandomForestRegressor(
            n_estimators=30,
            max_depth=6,
            min_samples_split=15,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1
        )
    
    print("🌲 Treinando Random Forest...")
    try:
        model.fit(X_train, y_train)
        
        # Métricas
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test) if len(X_test) > 0 else train_score
        
        y_pred = model.predict(X_test) if len(X_test) > 0 else model.predict(X_train)
        y_actual = y_test if len(y_test) > 0 else y_train
        
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
        
        print(f"✅ Modelo treinado com sucesso!")
        print(f"   📈 R² Treino: {train_score:.4f}")
        print(f"   📊 R² Teste: {test_score:.4f}")
        print(f"   📉 RMSE: {rmse:.4f}")
        print(f"   📋 MAPE: {mape:.2f}%")
        
        return model, feature_columns, {
            'train_score': train_score, 
            'test_score': test_score, 
            'rmse': rmse, 
            'mape': mape
        }
        
    except Exception as e:
        print(f"❌ Erro ao treinar modelo: {e}")
        return None, None, None

def prever_com_ml(model, feature_columns, dados_historicos, parametros, periodos_projecao, tipo_analise):
    """Faz previsões usando o modelo ML treinado - VERSÃO COMPLETA CORRIGIDA"""
    print("🔮 Fazendo previsões com Machine Learning...")
    
    if model is None:
        print("❌ Modelo ML não disponível para previsão")
        # Fallback: retornar último valor para todos os períodos
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
            datas_projecao = [dados_historicos.index[-1] + relativedelta(months=i) 
                             for i in range(1, periodos_projecao + 1)]
        
        return [ultimo_valor] * len(datas_projecao), datas_projecao
    
    # Datas de projeção
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
        
        # Gerar 24 horas consecutivas, pulando finais de semana
        horas_geradas = 0
        while horas_geradas < periodos_projecao:
            current_date += timedelta(hours=1)
            if current_date.weekday() < 5:
                datas_projecao.append(current_date)
                horas_geradas += 1
                
    else:
        datas_projecao = [dados_historicos.index[-1] + relativedelta(months=i) 
                         for i in range(1, periodos_projecao + 1)]
    
    projecao_ml = []
    
    # 🎯 ESTRATÉGIA ESPECÍFICA PARA CADA TIPO DE ANÁLISE
    
    if tipo_analise == 'MENSAL':
        print("📊 Aplicando estratégia ML CONSERVADORA para mensal...")
        
        try:
            valor_atual = dados_historicos['Cotação'].iloc[-1]
            print(f"📊 Valor atual para referência: {valor_atual:.4f}")
            
            # 1. Calcular tendência histórica de longo prazo
            if len(dados_historicos) >= 12:
                dados_recentes = dados_historicos['Cotação'].tail(12)
                retorno_medio = dados_recentes.pct_change().mean()
                if np.isnan(retorno_medio):
                    retorno_medio = 0
                # Limitar tendência máxima (evitar extremos)
                tendencia = np.clip(retorno_medio, -0.015, 0.015)  # Máximo ±1.5% ao mês
            else:
                tendencia = 0
            
            print(f"📊 Tendência mensal detectada: {tendencia*100:+.2f}%")
            
            # 2. Fazer previsão base apenas do primeiro período
            dados_com_features = criar_features_ml(dados_historicos, parametros, tipo_analise)
            
            previsao_base = valor_atual  # Fallback inicial
            
            if not dados_com_features.empty:
                # Garantir que todas as features estão presentes
                for col in feature_columns:
                    if col not in dados_com_features.columns:
                        dados_com_features[col] = 0
                
                ultimas_features = dados_com_features[feature_columns].iloc[-1:]
                previsao_tentativa = model.predict(ultimas_features)[0]
                
                # VALIDAÇÃO CRÍTICA: não permitir previsões absurdas
                variacao_tentativa = abs(previsao_tentativa - valor_atual) / valor_atual
                print(f"🔍 Validação ML: Previsão tentativa = {previsao_tentativa:.4f} (variação: {variacao_tentativa:.1%})")
                
                if variacao_tentativa <= 0.08:  # Até 8% de variação = aceitável
                    previsao_base = previsao_tentativa
                    print("✅ Previsão ML aceita")
                else:
                    print(f"⚠️  Previsão ML rejeitada (variação muito alta: {variacao_tentativa:.1%})")
                    previsao_base = valor_atual * (1 + tendencia)
            else:
                previsao_base = valor_atual * (1 + tendencia)
            
            # 3. Projeção com tendência suavizada
            for periodo in range(len(datas_projecao)):
                if periodo == 0:
                    previsao = previsao_base
                else:
                    # Aplicar tendência gradualmente com fator de redução
                    previsao = projecao_ml[-1] * (1 + tendencia * 0.6)
                
                # Validação final de sanidade
                variacao_total = abs(previsao - valor_atual) / valor_atual
                if variacao_total > 0.15:  # Limite máximo de 15% de variação total
                    print(f"⚠️  Correção aplicada: variação total muito alta ({variacao_total:.1%})")
                    previsao = valor_atual * (1 + np.sign(tendencia) * 0.15)
                
                projecao_ml.append(previsao)
                
        except Exception as e:
            print(f"❌ Erro no ML mensal: {e}")
            # Fallback conservador
            valor_atual = dados_historicos['Cotação'].iloc[-1]
            projecao_ml = [valor_atual] * len(datas_projecao)
    
    elif tipo_analise == 'HORARIA':
        print("📊 Aplicando estratégia HORÁRIA iterativa...")
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
                    
                    # Validação básica para horário
                    if np.isnan(previsao) or previsao <= 0:
                        previsao = dados_atual['Cotação'].iloc[-1]
                else:
                    previsao = dados_atual['Cotação'].iloc[-1]
                    
            except Exception as e:
                print(f"⚠️  Erro na previsão horária período {periodo}: {e}")
                previsao = dados_atual['Cotação'].iloc[-1]
            
            projecao_ml.append(previsao)
            
            # Atualizar dados para próxima iteração
            nova_data = datas_projecao[periodo]
            nova_linha = pd.DataFrame({'Cotação': [previsao]}, index=[nova_data])
            dados_atual = pd.concat([dados_atual, nova_linha])
            dados_atual['MM4'] = dados_atual['Cotação'].rolling(4, min_periods=1).mean()
            dados_atual['MM12'] = dados_atual['Cotação'].rolling(12, min_periods=1).mean()
            dados_atual = dados_atual.fillna(method='ffill')
    
    else:
        print("📊 Aplicando estratégia DIÁRIA padrão...")
        dados_atual = dados_historicos.copy()
        
        for periodo in range(len(datas_projecao)):
            dados_com_features = criar_features_ml(dados_atual, parametros, tipo_analise)
            ultimas_features = dados_com_features.iloc[-1:][feature_columns]
            previsao = model.predict(ultimas_features)[0]
            projecao_ml.append(previsao)
            
            nova_data = datas_projecao[periodo]
            novo_valor = previsao
            
            nova_linha = pd.DataFrame({
                'Cotação': [novo_valor],
                'MM5': [np.nan],
                'MM20': [np.nan]
            }, index=[nova_data])
            
            dados_atual = pd.concat([dados_atual, nova_linha])
            dados_atual['MM5'] = dados_atual['Cotação'].rolling(5, min_periods=1).mean()
            dados_atual['MM20'] = dados_atual['Cotação'].rolling(20, min_periods=1).mean()
    
    print(f"📈 Projeção ML final: {projecao_ml[0]:.4f} → {projecao_ml[-1]:.4f}")
    return projecao_ml, datas_projecao
# ========== BLOCO 9: PROJEÇÃO ECONÔMICA ==========
def calcular_projecao_economica(dados_historicos, parametros, periodos_projecao, tipo_analise):
    """Calcula projeção econômica - VERSÃO CORRIGIDA E MAIS REALISTA"""
    print("📈 Calculando projeção econômica...")
    
    projecao_economica = []
    valor_atual = dados_historicos['Cotação'].iloc[-1]

    # Estatísticas para reversão à média mais suave
    media_historica = dados_historicos['Cotação'].mean()
    desvio_historico = dados_historicos['Cotação'].std()
    valor_maximo_historico = dados_historicos['Cotação'].max()
    valor_minimo_historico = dados_historicos['Cotação'].min()

    print(f"📊 Estatísticas históricas: Média R$ {media_historica:.4f}, Desvio R$ {desvio_historico:.4f}")

    for periodo in range(periodos_projecao):
        if tipo_analise == 'DIARIA':
            diferencial_juros = ((parametros['juros_brasil'] - parametros['juros_exterior']) / 100) / 252
            
            desvio_da_media = (media_historica - valor_atual) / desvio_historico
            forca_reversao = desvio_da_media * 0.02
            
            resistencia = 0
            if valor_atual > valor_maximo_historico * 0.98:
                resistencia = -0.001
            
            choque_volatilidade = np.random.normal(0, parametros['volatilidade_diaria'])
            
            dia_semana = (dados_historicos.index[-1].weekday() + periodo) % 7
            if dia_semana == 0:
                fator_sazonal = 0.001
            elif dia_semana == 4:
                fator_sazonal = -0.001
            else:
                fator_sazonal = 0
            
            fator_total = (diferencial_juros + forca_reversao + resistencia + 
                           choque_volatilidade + fator_sazonal)
            
            variacao_maxima = 0.02
            
        elif tipo_analise == 'HORARIA':
            # PARÂMETROS PARA ANÁLISE HORÁRIA
            diferencial_juros = ((parametros['juros_brasil'] - parametros['juros_exterior']) / 100) / (365*24)
            
            desvio_da_media = (media_historica - valor_atual) / desvio_historico
            forca_reversao = desvio_da_media * 0.001  # Muito suave para horário
            
            resistencia = 0
            if valor_atual > valor_maximo_historico * 0.99:
                resistencia = -0.0001
            
            choque_volatilidade = np.random.normal(0, parametros['volatilidade_horaria'])
            
            # Sazonalidade intraday
            hora_atual = (dados_historicos.index[-1].hour + periodo) % 24
            if hora_atual in [8, 9]:  # Abertura
                fator_sazonal = 0.0003
            elif hora_atual in [16, 17]:  # Fechamento
                fator_sazonal = -0.0002
            else:
                fator_sazonal = 0
            
            fator_total = (diferencial_juros + forca_reversao + resistencia + 
                           choque_volatilidade + fator_sazonal)
            
            variacao_maxima = 0.005  # 0.5% máximo por hora
            
        else:
            # ✅ VERSÃO FINAL - MAIS REALISTA
            # 1. Diferencial de juros (Brasil 15% vs Euro 2.15% = +12.85% a.a. a favor do Real)
            diferencial_juros = ((parametros['juros_brasil'] - parametros['juros_exterior']) / 100) / 12

            # 2. Reversão à média MUITO mais suave
            desvio_da_media = (media_historica - valor_atual) / desvio_historico
            forca_reversao = desvio_da_media * 0.005  # ✅ Apenas 0.5% de reversão (era 2%)

            # 3. Resistência mínima ou zero
            resistencia = 0
            if valor_atual > valor_maximo_historico * 0.98:  # Só resistência se muito perto do máximo
                resistencia = -0.0005  # ✅ Quase insignificante

            # 4. Choque de volatilidade muito pequeno
            choque_volatilidade = np.random.normal(0, parametros['volatilidade']/300)  # ✅ Volatilidade mínima

            # 5. Sazonalidade mantida
            mes_atual = (dados_historicos.index[-1].month + periodo) % 12
            if mes_atual in [5, 6, 7]:
                fator_sazonal = 0.001  # ✅ Reduzida
            elif mes_atual in [10, 11]:
                fator_sazonal = -0.001
            else:
                fator_sazonal = 0

            # 6. Combinar fatores - AGORA COM MAIS PESO NO DIFERENCIAL DE JUROS
            fator_total = (diferencial_juros * 0.7) + (forca_reversao * 0.3) + resistencia + choque_volatilidade + fator_sazonal

            # 7. Limitar variação máxima
            variacao_maxima = 0.02  # ✅ Máximo 2% ao mês
            fator_total = np.clip(fator_total, -variacao_maxima, variacao_maxima)

        # Aplicar variação
        fator_total = np.clip(fator_total, -variacao_maxima, variacao_maxima)
        valor_atual = valor_atual * (1 + fator_total)

        # ✅ VALIDAÇÃO FINAL: Não permitir quedas extremas
        variacao_total = (valor_atual - dados_historicos['Cotação'].iloc[-1]) / dados_historicos['Cotação'].iloc[-1]
        
        if tipo_analise == 'HORARIA':
            limite_queda = -0.03  # 3% máximo para horário
        else:
            limite_queda = -0.20  # 20% máximo para diário/mensal
            
        if variacao_total < limite_queda:
            print(f"⚠️  Correção econômica: limite de queda atingido ({variacao_total:.1%})")
            valor_atual = dados_historicos['Cotação'].iloc[-1] * (1 + limite_queda)

        projecao_economica.append(valor_atual)

    print(f"📊 Projeção econômica corrigida: {projecao_economica[0]:.4f} → {projecao_economica[-1]:.4f}")
    return projecao_economica

# ========== BLOCO 10: MONTE CARLO ==========
def simulacao_monte_carlo(dados_historicos, parametros, periodos_projecao, tipo_analise):
    """Executa simulação Monte Carlo"""
    print("🎲 Executando simulações Monte Carlo...")
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

# ========== BLOCO 11: ENSEMBLE LEARNING - PESOS OTIMIZADOS ==========
def criar_ensemble_previsoes(projecao_economica, projecao_arima, projecao_mc, projecao_ml, metricas_ml=None, tipo_analise='HORARIA'):
    """Combina as previsões de todos os modelos usando ensemble learning - PESOS POR TIPO DE ANÁLISE"""
    print(f"\n🤝 CRIANDO ENSEMBLE PARA ANÁLISE {tipo_analise}")
    print("═" * 50)
    
    projecoes = {}
    pesos_por_tipo = {}
    
    # VERIFICAÇÃO DE MODELOS DISPONÍVEIS
    modelos_ativos = []
    
    # 1. Modelo Econômico (SEMPRE disponível)
    if len(projecao_economica) > 0:
        projecoes['Economico'] = projecao_economica
        modelos_ativos.append('Economico')
        print(f"✅ Econômico: {len(projecao_economica)} projeções")

    # 2. Modelo ARIMA
    if not projecao_arima.empty and len(projecao_arima) > 0:
        projecoes['ARIMA'] = projecao_arima
        modelos_ativos.append('ARIMA')
        print(f"✅ ARIMA: {len(projecao_arima)} projeções")

    # 3. Modelo Monte Carlo
    if not projecao_mc.empty and len(projecao_mc) > 0:
        projecoes['Monte_Carlo'] = projecao_mc
        modelos_ativos.append('Monte_Carlo')
        print(f"✅ Monte Carlo: {len(projecao_mc)} projeções")

    # 4. Modelo Machine Learning - PESOS DIFERENCIADOS POR TIPO
    ml_valido = False
    if (projecao_ml and len(projecao_ml) > 0 and 
        not all(np.isnan(x) for x in projecao_ml)):
        
        # ANALISAR A TENDÊNCIA DO ML
        variacao_ml = (projecao_ml[-1] - projecao_ml[0]) / projecao_ml[0] * 100
        print(f"📊 ML: Variação total de {variacao_ml:+.2f}%")
        
        # PESOS ESPECÍFICOS POR TIPO DE ANÁLISE
        if tipo_analise == 'HORARIA':
            # Horário: ML mais volátil - peso conservador
            if abs(variacao_ml) > 2.0:
                peso_ml = 0.10
                print(f"⚠️  ML horário com tendência forte ({variacao_ml:+.2f}%), peso: 10%")
            else:
                peso_ml = 0.15
                print(f"✅ ML horário estável, peso: 15%")
                
        elif tipo_analise == 'DIARIA':
            # Diário: ML mais confiável - peso moderado
            if metricas_ml and metricas_ml.get('test_score', 0) > 0.5:
                peso_ml = 0.25  # Boa performance
                print(f"✅ ML diário com boa performance (R²: {metricas_ml['test_score']:.3f}), peso: 25%")
            else:
                peso_ml = 0.20  # Performance regular
                print(f"✅ ML diário com performance regular, peso: 20%")
                
        else:  # MENSAL
            # Mensal: ML mais estratégico - peso maior
            if metricas_ml and metricas_ml.get('test_score', 0) > 0.6:
                peso_ml = 0.30  # Excelente performance
                print(f"🎯 ML mensal com excelente performance (R²: {metricas_ml['test_score']:.3f}), peso: 30%")
            else:
                peso_ml = 0.25  # Performance boa
                print(f"✅ ML mensal com performance boa, peso: 25%")
        
        projecoes['ML'] = projecao_ml
        pesos_por_tipo['ML'] = peso_ml
        modelos_ativos.append('ML')
        ml_valido = True
    else:
        print("❌ ML: Não disponível ou inválido")

    # CONFIGURAÇÃO DE PESOS BASE POR TIPO DE ANÁLISE
    print(f"\n📊 CONFIGURAÇÃO PARA {tipo_analise}:")
    
    if tipo_analise == 'HORARIA':
        pesos_base = {
            'Economico': 0.35,   # Alto: fundamentos sólidos
            'Monte_Carlo': 0.30, # Alto: estatística robusta
            'ARIMA': 0.25,       # Médio: tendências
            'ML': 0.10           # Baixo: volátil no curto prazo
        }
    elif tipo_analise == 'DIARIA':
        pesos_base = {
            'Economico': 0.30,   # Alto: fundamentos
            'Monte_Carlo': 0.25, # Médio-Alto: estatística
            'ARIMA': 0.25,       # Médio: tendências
            'ML': 0.20           # Médio: mais confiável no diário
        }
    else:  # MENSAL
        pesos_base = {
            'Economico': 0.40,   # Médio-Alto: fundamentos
            'Monte_Carlo': 0.40, # Médio: estatística
            'ARIMA': 0.29,       # Médio-Alto: tendências de longo prazo
            'ML': 0.01,           # Alto: estratégico no longo prazo
        }
    
    # AJUSTAR PESOS SE ML NÃO ESTIVER DISPONÍVEL
    if not ml_valido:
        peso_redistribuir = pesos_base['ML']
        for modelo in ['Economico', 'Monte_Carlo', 'ARIMA']:
            if modelo in modelos_ativos:
                pesos_base[modelo] += peso_redistribuir / 3
        print(f"🔄 ML não disponível - pesos redistribuídos")
    
    # APLICAR PESOS APENAS AOS MODELOS ATIVOS
    pesos_finais = {}
    for modelo in modelos_ativos:
        # Usar peso específico do ML se disponível, senão usar peso base
        if modelo == 'ML' and ml_valido:
            pesos_finais[modelo] = pesos_por_tipo['ML']
        else:
            pesos_finais[modelo] = pesos_base.get(modelo, 0.25)
    
    # NORMALIZAR PARA 100%
    total_pesos = sum(pesos_finais.values())
    pesos_finais = {modelo: peso/total_pesos for modelo, peso in pesos_finais.items()}
    
    print("🎯 PESOS FINAIS NO ENSEMBLE:")
    for modelo, peso in pesos_finais.items():
        print(f"   • {modelo}: {peso:.1%}")
    
    # CALCULAR ENSEMBLE
    ensemble_final = []
    n_periodos = len(projecao_economica)
    
    for i in range(n_periodos):
        valor_ensemble = 0
        for modelo, serie in projecoes.items():
            if isinstance(serie, pd.Series):
                valor_ensemble += serie.iloc[i] * pesos_finais[modelo]
            else:
                valor_ensemble += serie[i] * pesos_finais[modelo]
        ensemble_final.append(valor_ensemble)

    # ANALISAR RESULTADO
    variacao_ensemble = ((ensemble_final[-1] / ensemble_final[0]) - 1) * 100
    print(f"✅ Ensemble {tipo_analise} criado com sucesso!")
    print(f"   📈 Períodos: {len(ensemble_final)}")
    print(f"   🎯 Variação Ensemble: {variacao_ensemble:+.2f}%")
    
    return ensemble_final, pesos_finais
    
# ========== BLOCO 12: GERAÇÃO DE PDF ==========
def gerar_recomendacoes(metricas, dados_historicos):
    """Gera recomendações baseadas na análise"""
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

def gerar_relatorio_pdf(fig, dados_historicos, metricas, parametros, projecao_economica, projecao_arima, projecao_mc, projecao_ml, projecao_ensemble, pesos_ensemble, datas_projecao, par_moedas, tipo_analise, metricas_ml=None):
    """Gera um relatório completo em PDF"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Relatorio_{par_moedas.replace('/', '_')}_{tipo_analise}_{timestamp}.pdf"
        
        print(f"📄 Gerando PDF: {filename}")
        
        # Preparar valores formatados
        valor_arima = f"R$ {projecao_arima.iloc[-1]:.4f}" if not projecao_arima.empty else 'N/A'
        valor_mc = f"R$ {projecao_mc.iloc[-1]:.4f}" if not projecao_mc.empty else 'N/A'
        valor_ml = f"R$ {projecao_ml[-1]:.4f}" if projecao_ml else 'N/A'
        valor_ensemble = f"R$ {projecao_ensemble[-1]:.4f}" if projecao_ensemble else 'N/A'
        variacao_economica = ((projecao_economica[-1] / dados_historicos['Cotação'].iloc[-1]) - 1) * 100
        variacao_ensemble = ((projecao_ensemble[-1] / dados_historicos['Cotação'].iloc[-1]) - 1) * 100 if projecao_ensemble else 0
        
        # Definir texto do período
        if tipo_analise == 'DIARIA':
            periodo_texto = "12 DIAS"
        elif tipo_analise == 'HORARIA':
            periodo_texto = "24 HORAS"
        else:
            periodo_texto = "12 MESES"
        
        with PdfPages(filename) as pdf:
            # Página 1: Gráfico principal
            print("💾 Salvando gráfico no PDF...")
            fig.savefig(pdf, format='pdf', bbox_inches='tight', dpi=300)
            
            # Página 2: Análise e métricas
            print("📊 Criando página de análise...")
            plt.figure(figsize=(11.69, 8.27))  # A4
            plt.axis('off')
            
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
            
            # Página 3: Tabela detalhada
            print("📋 Criando página com tabela...")
            plt.figure(figsize=(11.69, 8.27))
            plt.axis('off')
            
            if tipo_analise == 'DIARIA':
                col_labels = ['Data', 'Econômica (R$)', 'ARIMA (R$)', 'Monte Carlo (R$)', 'Machine Learning (R$)', 'ENSEMBLE (R$)', 'Variação %']
            elif tipo_analise == 'HORARIA':
                col_labels = ['Data/Hora', 'Econômica (R$)', 'ARIMA (R$)', 'Monte Carlo (R$)', 'Machine Learning (R$)', 'ENSEMBLE (R$)', 'Variação %']
            else:
                col_labels = ['Mês/Ano', 'Econômica (R$)', 'ARIMA (R$)', 'Monte Carlo (R$)', 'Machine Learning (R$)', 'ENSEMBLE (R$)', 'Variação %']
            
            cell_text = []
            
            for i, data in enumerate(datas_projecao):
                variacao = ((projecao_ensemble[i] / dados_historicos['Cotação'].iloc[-1]) - 1) * 100 if projecao_ensemble else 0
                if tipo_analise == 'DIARIA':
                    data_str = data.strftime('%d/%m/%Y')
                elif tipo_analise == 'HORARIA':
                    data_str = data.strftime('%d/%m %H:%M')
                else:
                    data_str = data.strftime('%b/%Y')
                    
                linha = [
                    data_str,
                    f'{projecao_economica[i]:.4f}',
                    f'{projecao_arima.iloc[i]:.4f}' if not projecao_arima.empty else 'N/A',
                    f'{projecao_mc[i]:.4f}' if not projecao_mc.empty else 'N/A',
                    f'{projecao_ml[i]:.4f}' if projecao_ml else 'N/A',
                    f'{projecao_ensemble[i]:.4f}' if projecao_ensemble else 'N/A',
                    f'{variacao:+.2f}%'
                ]
                cell_text.append(linha)
            
            # Criar tabela com posicionamento correto
            table = plt.table(cellText=cell_text, colLabels=col_labels, 
                            loc='center', cellLoc='center',
                            bbox=[0.1, 0.1, 0.8, 0.8])
            
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.8)
            
            plt.title(f'PROJEÇÕES DETALHADAS - {par_moedas} - PRÓXIMOS {periodo_texto}', 
                     pad=30, fontsize=14, fontweight='bold')
            pdf.savefig(bbox_inches='tight')
            plt.close()
            
        print(f"✅ PDF gerado com sucesso: {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Erro ao gerar PDF: {e}")
        import traceback
        traceback.print_exc()
        return None

# ========== BLOCO 13: EXECUÇÃO PRINCIPAL ==========
def executar_analise_completa():
    """Executa toda a análise e retorna os objetos para PDF"""
    print("=" * 60)
    
    # SELECIONAR TIPO DE ANÁLISE
    tipo_analise, periodos_projecao, periodo_texto, freq = selecionar_tipo_analise()
    
    # SELECIONAR PAR DE MOEDAS
    par_selecionado = selecionar_par_moedas()
    
    # OBTER PARÂMETROS
    parametros = obter_parametros(tipo_analise, par_selecionado)
    
    print(f"🌍 ANÁLISE {par_selecionado} - {tipo_analise} ({periodo_texto})")
    print("=" * 60)
    
    # VERIFICAÇÃO DE COTAÇÃO ANTES DE GERAR DADOS
    print(f"\n💵 CONFIGURAÇÃO INICIAL DA COTAÇÃO")
    usar_cotacao_padrao = input("🔧 Usar cotação padrão dos dados? (s/n): ").strip().lower()
    
    cotacao_inicial = None
    if usar_cotacao_padrao in ['n', 'não', 'nao', 'no']:
        cotacao_inicial = obter_cotacao_atual(par_selecionado)
        print(f"🎯 Cotação definida pelo usuário: R$ {cotacao_inicial:.4f}")
    else:
        print("ℹ️  Usando cotação padrão dos dados históricos")
    
    # COLETA DE DADOS - AGORA COM A COTAÇÃO CORRETA DESDE O INÍCIO
    dados_historicos = criar_dados_historicos(tipo_analise, par_selecionado, cotacao_inicial)
    dados_historicos = dados_historicos.dropna().to_frame('Cotação')
    
    # VERIFICAÇÃO FINAL ABSOLUTA
    print(f"\n🔍 VERIFICAÇÃO FINAL DA COTAÇÃO:")
    print(f"📊 Cotação nos dados: R$ {dados_historicos['Cotação'].iloc[-1]:.4f}")
    
    # PERGUNTAR SE QUER AJUSTAR (APENAS PARA CONFIRMAÇÃO)
    ajustar_final = input("🔄 Deseja ajustar a cotação final? (s/n): ").strip().lower()
    if ajustar_final in ['s', 'sim', 'y', 'yes']:
        cotacao_final = obter_cotacao_atual(par_selecionado)
        print(f"🎯 Ajustando para: R$ {cotacao_final:.4f}")
        
        # MÉTODO DIRETO E INFALÍVEL
        dados_historicos['Cotação'].iloc[-1] = cotacao_final
        
        print(f"✅ Cotação final confirmada: R$ {dados_historicos['Cotação'].iloc[-1]:.4f}")
    
    # ANÁLISE TÉCNICA
    print("\n📈 Calculando indicadores técnicos...")
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
    
    # MODELAGEM
    projecao_arima = pd.Series(dtype=float)
    projecao_mc = pd.Series(dtype=float)
    projecao_ml = []
    metricas_ml = None
    
    # MACHINE LEARNING - PRIMEIRO PARA OBTER datas_projecao
    try:
        dados_com_features = criar_features_ml(dados_historicos, parametros, tipo_analise)
        modelo_ml, feature_columns, metricas_ml = treinar_modelo_ml(dados_com_features, parametros, tipo_analise)
        projecao_ml, datas_projecao = prever_com_ml(
            modelo_ml, feature_columns, dados_historicos, parametros, periodos_projecao, tipo_analise
        )
    except Exception as e:
        print(f"⚠️ Erro no Machine Learning: {e}")
        projecao_ml = []
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
            
            # CORREÇÃO HORÁRIA: Gerar 24 horas consecutivas, pulando finais de semana - FOREX
            horas_geradas = 0
            while horas_geradas < periodos_projecao:
                current_date += timedelta(hours=1)
                # Pular finais de semana (sábado = 5, domingo = 6)
                if current_date.weekday() < 5:
                    datas_projecao.append(current_date)
                    horas_geradas += 1
        else:
            datas_projecao = [dados_historicos.index[-1] + relativedelta(months=i) 
                             for i in range(1, periodos_projecao+1)]
    
    # CORREÇÃO: Usar o número REAL de períodos baseado em datas_projecao
    periodos_reais = len(datas_projecao)
    print(f"📊 Períodos reais de projeção: {periodos_reais} (solicitados: {periodos_projecao})")
    
    # ARIMA
    if len(dados_historicos) > 24:
        try:
            print("🤖 Ajustando modelo ARIMA...")
            modelo_arima = ARIMA(dados_historicos['Cotação'].dropna(), order=parametros['arima_order'])
            modelo_ajustado = modelo_arima.fit()
            previsao_arima = modelo_ajustado.get_forecast(steps=periodos_reais)
            projecao_arima = previsao_arima.predicted_mean
            print("✅ Modelo ARIMA ajustado com sucesso")
        except Exception as e:
            print(f"⚠️ Erro no ARIMA: {e}")
    
    # MONTE CARLO
    projecao_mc, simulacoes = simulacao_monte_carlo(dados_historicos, parametros, periodos_reais, tipo_analise)
    
    # PROJEÇÃO ECONÔMICA - CORRIGIDA: usar periodos_reais
    projecao_economica = calcular_projecao_economica(dados_historicos, parametros, periodos_reais, tipo_analise)
    
    # ENSEMBLE LEARNING
    projecao_ensemble, pesos_ensemble = criar_ensemble_previsoes(
        projecao_economica, projecao_arima, projecao_mc, projecao_ml, metricas_ml
    )
    
    # MÉTRICAS
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
    
    metricas['variacao_projetada'] = ((projecao_economica[-1] / dados_historicos['Cotação'].iloc[-1]) - 1) * 100
    
    # VISUALIZAÇÃO
    print("🎨 Gerando visualizações...")
    
    if tipo_analise == 'DIARIA':
        fig = plt.figure(constrained_layout=True, figsize=(20, 16))
        gs = fig.add_gridspec(4, 3)

        ax_principal = fig.add_subplot(gs[:2, :])
        
        dados_recentes = dados_historicos.tail(60)
        dados_recentes['Cotação'].plot(ax=ax_principal, color='navy', label='Histórico (60 dias)', linewidth=2)
        dados_recentes['MM20'].plot(ax=ax_principal, color='green', label='MM 20 Dias', linestyle='--', linewidth=1.5)

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
            projecao_ensemble_series.plot(ax=ax_principal, color='gold', marker='*', 
                                        label='ENSEMBLE', linestyle='-', linewidth=3, markersize=6)

        ax_principal.set_title(f'ANÁLISE {tipo_analise} {par_selecionado} - PROJEÇÕES PARA {periodos_reais} DIAS', 
                             pad=20, fontsize=16, fontweight='bold')
        ax_principal.set_ylabel('R$/' + parametros['moeda_base'], fontsize=12)
        ax_principal.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_principal.grid(True, alpha=0.3)

        ax_rsi = fig.add_subplot(gs[2, 0])
        dados_recentes['RSI'].dropna().plot(ax=ax_rsi, color='brown', linewidth=2)
        ax_rsi.axhline(70, color='red', linestyle='--', alpha=0.7, label='Sobrevendido')
        ax_rsi.axhline(30, color='green', linestyle='--', alpha=0.7, label='Sobrecomprado')
        ax_rsi.set_title('RSI (10 períodos)', fontsize=12)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.legend()
        ax_rsi.grid(True, alpha=0.3)

        ax_returns = fig.add_subplot(gs[2, 1])
        returns.tail(100).hist(ax=ax_returns, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax_returns.set_title('Distribuição dos Retornos Diários', fontsize=12)
        ax_returns.set_xlabel('Retorno Diário')
        ax_returns.set_ylabel('Frequência')
        ax_returns.grid(True, alpha=0.3)

        ax_mc = fig.add_subplot(gs[2, 2])
        for i in range(min(30, parametros['n_simulacoes'])):
            ax_mc.plot(datas_projecao, simulacoes[i], alpha=0.05, color='blue')
        ax_mc.plot(datas_projecao, projecao_mc, color='red', linewidth=2, label='Média')
        ax_mc.set_title('Simulações Monte Carlo', fontsize=12)
        ax_mc.legend()
        ax_mc.grid(True, alpha=0.3)

        ax_tabela = fig.add_subplot(gs[3, :])
        ax_tabela.axis('off')
        
        colunas = ['Data', 'Econômica', 'ARIMA', 'Monte Carlo', 'Machine Learning', 'ENSEMBLE']
        dados_tabela = []
        
        for i, data in enumerate(datas_projecao):
            data_str = data.strftime('%d/%m')
            linha = [
                data_str,
                f'R$ {projecao_economica[i]:.4f}',
                f'R$ {projecao_arima.iloc[i]:.4f}' if not projecao_arima.empty else 'N/A',
                f'R$ {projecao_mc[i]:.4f}',
                f'R$ {projecao_ml[i]:.4f}' if projecao_ml else 'N/A',
                f'R$ {projecao_ensemble[i]:.4f}' if projecao_ensemble else 'N/A'
            ]
            dados_tabela.append(linha)
        
        tabela = ax_tabela.table(cellText=dados_tabela,
                                colLabels=colunas,
                                colColours=['#f0f0f0'] * 6,
                                cellLoc='center', 
                                loc='center',
                                bbox=[0, 0, 1, 1])
        tabela.set_fontsize(9)
        tabela.scale(1, 1.5)
        
    elif tipo_analise == 'HORARIA':
        fig = plt.figure(constrained_layout=True, figsize=(20, 16))
        gs = fig.add_gridspec(4, 3)

        ax_principal = fig.add_subplot(gs[:2, :])
        
        dados_recentes = dados_historicos.tail(48)
        dados_recentes['Cotação'].plot(ax=ax_principal, color='navy', label='Histórico (48 horas)', linewidth=2)
        dados_recentes['MM12'].plot(ax=ax_principal, color='green', label='MM 12 Horas', linestyle='--', linewidth=1.5)

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
            projecao_ensemble_series.plot(ax=ax_principal, color='gold', marker='*', 
                                        label='ENSEMBLE', linestyle='-', linewidth=3, markersize=6)

        ax_principal.set_title(f'ANÁLISE {tipo_analise} {par_selecionado} - PROJEÇÕES PARA {periodos_reais} HORAS', 
                             pad=20, fontsize=16, fontweight='bold')
        ax_principal.set_ylabel('R$/' + parametros['moeda_base'], fontsize=12)
        ax_principal.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_principal.grid(True, alpha=0.3)

        ax_rsi = fig.add_subplot(gs[2, 0])
        dados_recentes['RSI'].dropna().plot(ax=ax_rsi, color='brown', linewidth=2)
        ax_rsi.axhline(70, color='red', linestyle='--', alpha=0.7, label='Sobrevendido')
        ax_rsi.axhline(30, color='green', linestyle='--', alpha=0.7, label='Sobrecomprado')
        ax_rsi.set_title('RSI (12 períodos)', fontsize=12)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.legend()
        ax_rsi.grid(True, alpha=0.3)

        ax_returns = fig.add_subplot(gs[2, 1])
        returns.tail(100).hist(ax=ax_returns, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax_returns.set_title('Distribuição dos Retornos Horários', fontsize=12)
        ax_returns.set_xlabel('Retorno Horário')
        ax_returns.set_ylabel('Frequência')
        ax_returns.grid(True, alpha=0.3)

        ax_mc = fig.add_subplot(gs[2, 2])
        for i in range(min(20, parametros['n_simulacoes'])):
            ax_mc.plot(datas_projecao, simulacoes[i], alpha=0.05, color='blue')
        ax_mc.plot(datas_projecao, projecao_mc, color='red', linewidth=2, label='Média')
        ax_mc.set_title('Simulações Monte Carlo', fontsize=12)
        ax_mc.legend()
        ax_mc.grid(True, alpha=0.3)

        ax_tabela = fig.add_subplot(gs[3, :])
        ax_tabela.axis('off')
        
        colunas = ['Data/Hora', 'Econômica', 'ARIMA', 'Monte Carlo', 'Machine Learning', 'ENSEMBLE']
        dados_tabela = []
        
        for i, data in enumerate(datas_projecao):
            data_str = data.strftime('%d/%m %H:%M')
            linha = [
                data_str,
                f'R$ {projecao_economica[i]:.4f}',
                f'R$ {projecao_arima.iloc[i]:.4f}' if not projecao_arima.empty else 'N/A',
                f'R$ {projecao_mc[i]:.4f}',
                f'R$ {projecao_ml[i]:.4f}' if projecao_ml else 'N/A',
                f'R$ {projecao_ensemble[i]:.4f}' if projecao_ensemble else 'N/A'
            ]
            dados_tabela.append(linha)
        
        tabela = ax_tabela.table(cellText=dados_tabela,
                                colLabels=colunas,
                                colColours=['#f0f0f0'] * 6,
                                cellLoc='center', 
                                loc='center',
                                bbox=[0, 0, 1, 1])
        tabela.set_fontsize(8)
        tabela.scale(1, 1.8)
        
    else:
        fig = plt.figure(constrained_layout=True, figsize=(20, 16))
        gs = fig.add_gridspec(4, 3)
        
        ax_principal = fig.add_subplot(gs[:2, :])
        dados_historicos['Cotação'].plot(ax=ax_principal, color='navy', label='Histórico', linewidth=2.5)
        dados_historicos['MM12'].plot(ax=ax_principal, color='green', label='MM 12M', linestyle='--', linewidth=2)
        
        if not projecao_arima.empty:
            projecao_arima.plot(ax=ax_principal, color='orange', marker='s', label='ARIMA', linestyle=':', linewidth=2, markersize=4)
        
        projecao_economica_series = pd.Series(projecao_economica, index=datas_projecao)
        projecao_economica_series.plot(ax=ax_principal, color='red', marker='o', label='Econômica', linestyle='--', linewidth=2, markersize=4)
        
        projecao_mc_series = pd.Series(projecao_mc, index=datas_projecao)
        projecao_mc_series.plot(ax=ax_principal, color='purple', marker='^', label='Monte Carlo', linestyle='-.', linewidth=2, markersize=4)
        
        if projecao_ml:
            projecao_ml_series = pd.Series(projecao_ml, index=datas_projecao)
            projecao_ml_series.plot(ax=ax_principal, color='brown', marker='D', label='Machine Learning', linestyle='-', linewidth=2, markersize=4)
        
        if projecao_ensemble:
            projecao_ensemble_series = pd.Series(projecao_ensemble, index=datas_projecao)
            projecao_ensemble_series.plot(ax=ax_principal, color='gold', marker='*', 
                                        label='ENSEMBLE', linestyle='-', linewidth=3, markersize=8)
        
        ax_principal.set_title(f'ANÁLISE {tipo_analise} {par_selecionado} - PROJEÇÕES PARA {periodos_reais} MESES', pad=20, fontsize=16, fontweight='bold')
        ax_principal.set_ylabel('R$/' + parametros['moeda_base'], fontsize=12)
        ax_principal.legend()
        ax_principal.grid(True, alpha=0.3)
        
        ax_rsi = fig.add_subplot(gs[2, 0])
        dados_historicos['RSI'].dropna().plot(ax=ax_rsi, color='brown', linewidth=2)
        ax_rsi.axhline(70, color='red', linestyle='--', alpha=0.7, label='Sobrevendido')
        ax_rsi.axhline(30, color='green', linestyle='--', alpha=0.7, label='Sobrecomprado')
        ax_rsi.set_title('RSI (14 períodos)')
        ax_rsi.set_ylim(0, 100)
        ax_rsi.legend()
        ax_rsi.grid(True, alpha=0.3)
        
        ax_returns = fig.add_subplot(gs[2, 1])
        returns.hist(ax=ax_returns, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
        ax_returns.set_title('Distribuição dos Retornos')
        ax_returns.set_xlabel('Retorno')
        ax_returns.set_ylabel('Frequência')
        ax_returns.grid(True, alpha=0.3)
        
        ax_mc = fig.add_subplot(gs[2, 2])
        for i in range(min(30, parametros['n_simulacoes'])):
            ax_mc.plot(datas_projecao, simulacoes[i], alpha=0.05, color='blue')
        ax_mc.plot(datas_projecao, projecao_mc, color='red', linewidth=2, label='Média')
        ax_mc.set_title('Simulações Monte Carlo')
        ax_mc.legend()
        ax_mc.grid(True, alpha=0.3)
        
        ax_tabela = fig.add_subplot(gs[3, :])
        ax_tabela.axis('off')
        
        colunas = ['Mês/Ano', 'Econômica', 'ARIMA', 'Monte Carlo', 'Machine Learning', 'ENSEMBLE']
        dados_tabela = []
        
        for i, data in enumerate(datas_projecao):
            data_str = data.strftime('%m/%Y')
            linha = [
                data_str,
                f'R$ {projecao_economica[i]:.4f}',
                f'R$ {projecao_arima.iloc[i]:.4f}' if not projecao_arima.empty else 'N/A',
                f'R$ {projecao_mc[i]:.4f}',
                f'R$ {projecao_ml[i]:.4f}' if projecao_ml else 'N/A',
                f'R$ {projecao_ensemble[i]:.4f}' if projecao_ensemble else 'N/A'
            ]
            dados_tabela.append(linha)
        
        tabela = ax_tabela.table(cellText=dados_tabela,
                                colLabels=colunas,
                                colColours=['#f0f0f0'] * 6,
                                cellLoc='center', 
                                loc='center',
                                bbox=[0, 0, 1, 1])
        tabela.set_fontsize(9)
        tabela.scale(1, 1.5)
    
    plt.suptitle(f'ANÁLISE COMPLETA {par_selecionado} - {tipo_analise} COM ML + ENSEMBLE', 
                fontsize=18, fontweight='bold', y=0.98)
    
    return (fig, dados_historicos, metricas, parametros, projecao_economica, 
            projecao_arima, projecao_mc, projecao_ml, projecao_ensemble, 
            pesos_ensemble, datas_projecao, par_selecionado, tipo_analise, metricas_ml)
    
# ========== BLOCO 14: EXECUÇÃO FINAL ==========
if __name__ == "__main__":
    # Executar análise
    resultado = executar_analise_completa()
    
    # Desempacotar todos os resultados
    (fig, dados_historicos, metricas, parametros, projecao_economica, 
     projecao_arima, projecao_mc, projecao_ml, projecao_ensemble, 
     pesos_ensemble, datas_projecao, par_selecionado, tipo_analise, metricas_ml) = resultado
    
    # Mostrar gráfico
    plt.show()
    
    # Perguntar se quer gerar PDF
    print("\n" + "="*60)
    gerar_pdf = input("📄 Deseja gerar relatório em PDF? (s/n): ").strip().lower()
    
    if gerar_pdf in ['s', 'sim', 'y', 'yes']:
        filename = gerar_relatorio_pdf(fig, dados_historicos, metricas, parametros, 
                                     projecao_economica, projecao_arima, projecao_mc, 
                                     projecao_ml, projecao_ensemble, pesos_ensemble, 
                                     datas_projecao, par_selecionado, tipo_analise, metricas_ml)
        if filename:
            print(f"📁 Arquivo salvo em: {os.path.abspath(filename)}")
        else:
            print("❌ Erro ao gerar PDF")
    else:
        print("📊 Análise concluída sem gerar PDF")
    
    print("=" * 60)
    print("✅ Análise concluída!")
    print("=" * 60)