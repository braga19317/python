import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import gdown
import os

# Função para baixar o arquivo do Google Drive
def baixar_arquivo_google_drive(url, caminho_local):
    gdown.download(url, caminho_local, quiet=False)

def carregar_dados():
    # Defina o URL do Google Drive e o caminho local
    url_clientes = 'https://drive.google.com/uc?id=12doumGMLErxW6j1KM5idWHAzXAH1Woqd&export=download'
    caminho_clientes = 'estatistica_clientes.xlsx'
    url_vendas = 'https://drive.google.com/uc?id=1dYHZlfvZlwOhJP1cJlQRbMowoVRBY78N&export=download'
    caminho_vendas = 'Vendas_Credito.xlsx'
    
    # Baixe os arquivos
    baixar_arquivo_google_drive(url_clientes, caminho_clientes)
    baixar_arquivo_google_drive(url_vendas, caminho_vendas)

    # Verifique se os arquivos foram baixados corretamente
    if not os.path.exists(caminho_clientes) or not os.path.exists(caminho_vendas):
        st.error("Erro ao baixar os arquivos. Verifique os URLs e tente novamente.")
        return None, None

    # Carregue os arquivos usando o caminho completo
    clientes_df = pd.read_excel(caminho_clientes, engine='openpyxl')
    vendas_credito_df = pd.read_excel(caminho_vendas, engine='openpyxl')

    # Corrigir os nomes das colunas
    clientes_df.columns = [
        "Inativo", "Nro.", "Empresa", "Cliente ", "Fantasia", "Referência", "Vencimento",
        "Vl.liquido", "TD", "Nr.docto", "Dt.pagto", "Vl.pagamento", "TP", "Nr.pagamento",
        "Conta", "Dt.Emissão", "Cobrança", "Modelo", "Negociação", "Duplicata", "Razão Social",
        "CNPJ/CPF", "PDD"
    ]

    vendas_credito_df.columns = [
        "Inativo","Nro.", "Empresa", "Cliente1", "Fantasia1", "Referência", "Vencimento1", "Vl.liquido1",
        "TD", "Nr.docto", "Dt.pagto", "Vl.pagto", "TP", "Nr.pagto", "Conta", "Dt.Emissão1",
        "Cobrança","Modelo", "Negociação","Duplicata", "Razão Social", "CNPJ/CPF", "PDD"
    ]

    # Cria uma nova coluna combinando "Cliente" e "Fantasia"
    clientes_df["Cliente_Fantasia"] = clientes_df.apply(lambda row: f"{row['Cliente ']} - {row['Fantasia']}", axis=1)

    return clientes_df, vendas_credito_df

def categorizar_cliente_por_faturamento(faturamento):
    if faturamento <= 10000:
        return 'Até 10 mil'
    elif faturamento <= 50000:
        return 'De 11 mil a 50 mil'
    elif faturamento <= 100000:
        return 'De 51 mil a 100 mil'
    elif faturamento <= 150000:
        return 'De 101 mil a 150 mil'
    elif faturamento <= 350000:
        return 'De 151 mil a 350 mil'
    elif faturamento <= 1000000:
        return 'De 151 mil até 1 milhão'
    else:
        return 'Acima de 1 milhão'
        
def grafico_regua_faturamento(total_geral):
    fig, ax = plt.subplots(figsize=(10, 2))
    categorias = ['Até 10 mil', 'De 11 mil a 50 mil', 'De 51 mil a 100 mil', 'De 101 mil a 150 mil', 'De 151 mil a 350 mil', 'De 351 mil até 1 milhão', 'Acima de 1 milhão']
    posicoes = [10000, 50000, 100000, 150000, 350000, 1000000, 1500000]  # Adicionei uma posição extra para corresponder ao número de categorias
    ax.hlines(1, xmin=0, xmax=1500000, color='gray', linewidth=5)  # Ajustei o limite máximo para corresponder à nova posição
    ax.plot(total_geral, 1, 'ro')  # Marca a posição do cliente
    ax.set_xlim(0, 1500000)  # Ajustei o limite máximo para corresponder à nova posição
    ax.set_xticks(posicoes)
    ax.set_xticklabels(categorias, rotation=45, ha='right')  # Rotaciona os rótulos em 45 graus
    ax.set_yticks([])
    plt.tight_layout()  # Ajusta o layout para evitar sobreposição
    st.pyplot(fig)


def main():
    st.title("Análise de Clientes")
    st.sidebar.title("Filtros")

    # Carregar os dados
    clientes_df, vendas_credito_df = carregar_dados()
    if clientes_df is None or vendas_credito_df is None:
        return

    # Exibe a lista suspensa com as opções de cliente + fantasia
    opcoes = clientes_df["Cliente_Fantasia"].unique().tolist()
    escolha = st.sidebar.selectbox("Escolha um Cliente_Fantasia:", ["Selecione um cliente"] + opcoes, key='selectbox_cliente_fantasia')

    if escolha == "Selecione um cliente":
        st.warning("Por favor, selecione um cliente.")
        return
        
    # Exibe a identificação do cliente em análise logo abaixo do título
    st.subheader(f"Cliente em Análise: {escolha}")

    # Filtra os dados com base na escolha do usuário
    clientes_filtrados = clientes_df[clientes_df["Cliente_Fantasia"] == escolha].copy()

    # Converte as colunas de data
    for coluna in ["Vencimento", "Dt.Emissão"]:
        clientes_filtrados[coluna] = pd.to_datetime(clientes_filtrados[coluna], errors='coerce')

    # Filtra os dados de vendas a crédito para o cliente selecionado
    cliente_nome = clientes_filtrados["Cliente "].iloc[0]
    vendas_cliente = vendas_credito_df[vendas_credito_df["Cliente1"] == cliente_nome].copy()

    # Converte as colunas de data em vendas a crédito
    for coluna in ["Dt.pagto", "Vencimento1"]:
        vendas_cliente[coluna] = pd.to_datetime(vendas_cliente[coluna], errors='coerce')

    hoje = pd.Timestamp.today()

    # Filtra valores vencidos e a vencer
    valores_vencidos = clientes_filtrados[clientes_filtrados["Vencimento"] < hoje]
    valores_a_vencer = clientes_filtrados[clientes_filtrados["Vencimento"] >= hoje]

    # Cálculo dos totais
    total_vencidos = valores_vencidos["Vl.liquido"].sum()
    total_a_vencer = valores_a_vencer["Vl.liquido"].sum()
    total_geral = total_vencidos + total_a_vencer

    # Categorizar cliente por faturamento
    categoria_faturamento = categorizar_cliente_por_faturamento(total_geral)
    
    # Exibe os totais e a categoria de faturamento
    st.write(f"**Total de registros vencidos:** {len(valores_vencidos)}")
    st.write(f"**Total de registros a vencer:** {len(valores_a_vencer)}")
    st.write(f"**Total Vencidos:** R$ {total_vencidos:,.2f}")
    st.write(f"**Total A Vencer:** R$ {total_a_vencer:,.2f}")
    st.write(f"**Total Geral:** R$ {total_geral:,.2f}")
    st.write(f"**Categoria de Faturamento:** {categoria_faturamento}")

    # Chama a função para exibir o gráfico tipo régua
    grafico_regua_faturamento(total_geral)
        
    # Alertas baseados nos totais
    if total_vencidos > total_a_vencer:
        st.error("Atenção: Títulos vencidos são maiores que títulos a vencer!")
    elif total_vencidos < total_a_vencer:
        st.success("Títulos a vencer são maiores que títulos vencidos.")
    else:
        st.info("Títulos vencidos e títulos a vencer são iguais.")

    # Cálculo dos percentuais
    percentual_vencidos = (total_vencidos / total_geral * 100) if total_geral > 0 else 0
    percentual_a_vencer = (total_a_vencer / total_geral * 100) if total_geral > 0 else 0

    st.write(f"**Montante de Vencidos:** R$ {total_vencidos:,.2f} ({percentual_vencidos:.2f}% do total)")
    st.write(f"**Montante de A Vencer:** R$ {total_a_vencer:,.2f} ({percentual_a_vencer:.2f}% do total)")

    # Cálculo do prazo médio de faturamento ponderado
    clientes_filtrados["Prazo"] = (clientes_filtrados["Vencimento"] - clientes_filtrados["Dt.Emissão"]).dt.days
    prazo_medio_ponderado = (clientes_filtrados["Prazo"] * clientes_filtrados["Vl.liquido"]).sum() / clientes_filtrados["Vl.liquido"].sum()
    st.write(f"**Prazo Médio de Faturamento (ponderado):** {prazo_medio_ponderado:.2f} dias")

    # Cálculo do prazo médio de recebimento
    if 'Dt.pagto' in vendas_cliente.columns and 'Vencimento1' in vendas_cliente.columns and 'Vl.liquido1' in vendas_cliente.columns:
        vendas_cliente['Dias Para Recebimento'] = (vendas_cliente['Dt.pagto'] - vendas_cliente['Vencimento1']).dt.days
        soma_valores_recebidos = vendas_cliente['Vl.liquido1'].sum()
        prazo_medio_recebimento = (vendas_cliente['Dias Para Recebimento'] * vendas_cliente['Vl.liquido1']).sum() / soma_valores_recebidos if soma_valores_recebidos > 0 else 0
        st.write(f"**Prazo Médio de Recebimento (ponderado):** {prazo_medio_recebimento:.2f} dias")
    else:
        st.write("**Informações insuficientes para calcular o prazo médio de recebimento.**")

    # Cálculo do faturamento diário médio (ADP)
    dias_no_periodo = (vendas_cliente["Dt.Emissão1"].max() - vendas_cliente["Dt.Emissão1"].min()).days
    if dias_no_periodo > 0:
        faturamento_diario_medio = vendas_cliente["Vl.liquido1"].sum() / dias_no_periodo
    else:
        faturamento_diario_medio = 0

    # Cálculo do DSO
    contas_receber_total = clientes_filtrados["Vl.liquido"].sum()
    if faturamento_diario_medio > 0:
        DSO = contas_receber_total / faturamento_diario_medio
    else:
        DSO = 0

    st.write(f"**DSO (Days Sales Outstanding) para o cliente selecionado:** {DSO:.2f} dias")

    # Cálculo do CEI (Collection Effectiveness Index)
    total_vendas_credito = vendas_cliente["Vl.liquido1"].sum()
    total_pagamentos_recebidos = vendas_cliente["Vl.pagto"].sum()
    if total_vendas_credito > 0:
        CEI = (total_pagamentos_recebidos / total_vendas_credito) * 100
    else:
        CEI = 0

    st.write(f"**CEI (Collection Effectiveness Index) para o cliente selecionado:** {CEI:.2f}%")

    # Análise de Crédito
    st.subheader("Análise de Crédito")
    turnover_ratio = total_vendas_credito / contas_receber_total if contas_receber_total > 0 else 0
    st.write(f"**Índice de Rotatividade de Contas a Receber:** {turnover_ratio:.2f}")

    # Comentário sobre o Índice de Rotatividade
    if turnover_ratio > 10:
        st.write("Comentário: O índice de rotatividade está alto, indicando que a empresa está eficiente em cobrar suas contas a receber.")
    elif turnover_ratio < 5:
        st.write("Comentário: O índice de rotatividade está baixo, sugerindo possíveis problemas na cobrança ou clientes com dificuldades financeiras.")
    else:
        st.write("Comentário: O índice de rotatividade está dentro da média, indicando uma eficiência razoável na cobrança das contas a receber.")

    # Gráfico de pizza para totais
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.pie(
        [total_vencidos, total_a_vencer],
        labels=["Vencidos", "A Vencer"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#FF6F61", "#6FA2FF"],
        wedgeprops={"linewidth": 1, "edgecolor": "white"}
    )
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # Análise de Tendências
    st.subheader("Análise de Tendências")
    fig, ax = plt.subplots(figsize=(10, 6))
    clientes_filtrados.set_index("Vencimento", inplace=True)
    valores_vencidos.set_index("Vencimento", inplace=True)
    valores_a_vencer.set_index("Vencimento", inplace=True)
    ax.bar(valores_vencidos.index, valores_vencidos["Vl.liquido"], label='Valores Vencidos', color='red', width=0.4, align='center')
    ax.bar(valores_a_vencer.index, valores_a_vencer["Vl.liquido"], label='Valores a Vencer', color='green', width=0.4, align='edge')
    ax.set_title('Tendência de Valores Vencidos e a Vencer')
    ax.set_xlabel('Data de Vencimento')
    ax.set_ylabel('Valor (R$)')
    ax.legend()
    st.pyplot(fig)

    # Comentário sobre a Análise de Tendências
    st.write("Comentário: A análise de tendências mostra como os valores vencidos e a vencer variam ao longo do tempo. Um aumento nos valores vencidos pode indicar problemas na cobrança, enquanto um aumento nos valores a vencer pode sugerir um crescimento nas vendas a crédito.")
   
    # Análise de Desempenho
    st.subheader("Análise de Desempenho")
    desempenho_anterior = clientes_filtrados["Vl.liquido"].sum()
    desempenho_atual = vendas_cliente["Vl.liquido1"].sum()
    periodo_anterior = f"{clientes_filtrados['Dt.Emissão'].min().date()} a {clientes_filtrados['Dt.Emissão'].max().date()}"
    periodo_atual = f"{vendas_cliente['Dt.Emissão1'].min().date()} a {vendas_cliente['Dt.Emissão1'].max().date()}"
    if desempenho_anterior > 0:
        variacao_desempenho = ((desempenho_atual - desempenho_anterior) / desempenho_anterior) * 100
    else:
        variacao_desempenho = 0
    st.write(f"**Período Anterior:** {periodo_anterior}")
    st.write(f"**Total Anterior:** R$ {desempenho_anterior:,.2f}")
    st.write(f"**Período Atual:** {periodo_atual}")
    st.write(f"**Total Atual:** R$ {desempenho_atual:,.2f}")
    st.write(f"**Variação de Desempenho:** {variacao_desempenho:.2f}%")

    # Comentário sobre a Variação de Desempenho
    if variacao_desempenho > 0:
        st.write("Comentário: O desempenho melhorou em relação ao período anterior, indicando um crescimento nas vendas ou recebimentos.")
    elif variacao_desempenho < 0:
        st.write("Comentário: O desempenho piorou em relação ao período anterior, sugerindo uma queda nas vendas ou recebimentos.")
    else:
        st.write("Comentário: O desempenho permaneceu estável em relação ao período anterior.")

    # Análise de Inadimplência
    st.subheader("Análise de Inadimplência")
    inadimplencia = total_vencidos / total_geral * 100 if total_geral > 0 else 0
    st.write(f"**Taxa de Inadimplência:** {inadimplencia:.2f}%")

    # Comentário sobre a Taxa de Inadimplência
    if inadimplencia > 20:
        st.write("Comentário: A taxa de inadimplência está alta, indicando que muitos clientes estão atrasados nos pagamentos.")
    elif inadimplencia < 5:
        st.write("Comentário: A taxa de inadimplência está baixa, indicando que a maioria dos clientes está pagando em dia.")
    else:
        st.write("Comentário: A taxa de inadimplência está dentro de um intervalo aceitável.")

    # Análise de Sazonalidade
    st.subheader("Análise de Sazonalidade")
    vendas_cliente['Mês'] = vendas_cliente['Dt.Emissão1'].dt.month
    sazonalidade = vendas_cliente.groupby('Mês')['Vl.liquido1'].sum()

    fig, ax = plt.subplots(figsize=(10, 6))
    sazonalidade.plot(kind='bar', ax=ax)
    ax.set_title('Sazonalidade das Vendas')
    ax.set_xlabel('Mês')
    ax.set_ylabel('Valor das Vendas (R$)')
    st.pyplot(fig)

    # Comentário sobre a Análise de Sazonalidade
    st.write("Comentário: A análise de sazonalidade ajuda a identificar padrões de vendas ao longo do ano. Picos em determinados meses podem indicar sazonalidade nas vendas, o que pode ser útil para planejamento de estoque e estratégias de marketing.")

if __name__ == "__main__":
    main()
