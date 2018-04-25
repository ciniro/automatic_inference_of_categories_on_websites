"""
CLASSIFICAO DE SITES COM MLP - PROFESSOR CINIRO NAMETALA
INSTITUTO FEDERAL DE MINAS GERAIS - CAMPUS BAMBUI
"""
# IMPORTANDO PACOTES
import os.path
import pandas as pd
import sklearn
from nltk.corpus import stopwords
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from bs4 import BeautifulSoup
from urllib.request import urlopen
# -----------------------------------------------


# FUNCAO PARA TREINAR O MODELO DE CLASSIFICACAO
def treina(dados, modelo_arquivo, tam_treino):
    # INICIA O TREINAMENTO DO MODELO
    print("\nTreinando o modelo...")

    # ELIMINA CAMPOS NA
    dados = dados.dropna()

    # GERA UM VETOR QUE IRA ARMAZENAR A FREQUENCIA DE CADA PALAVRA
    vectorizer = CountVectorizer(
        analyzer="word",
        tokenizer=None,
        preprocessor=None,
        stop_words=None
    )

    # PREPARA UM MODELO DE REGRESSAO LOGISTICA PARA TREINAMENTO
    # classificador = LogisticRegression()
    classificador = MLPClassifier()

    # ADICIONA A REGRESSAO E O VETOR DE CATEGORIAS A UM PIPELINE
    pipe = Pipeline([('vect', vectorizer), ('logreg', classificador)])

    # SEPARA OS CONJUNTOS DE TREINAMENTO E TESTE
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(dados.texto, dados.categoria, train_size=tam_treino)

    # RODA O PIPELINE EXECUTANDO O TREINAMENTO COM O MODELO
    pipe.fit(x_train, y_train)

    # CALCULA A ACURACIA
    acuracia = pipe.score(x_test, y_test)

    # MOSTRA A ACURACIA NA TELA
    msg = "\n{:.0%} de acuracia nos dados de treinamento e {:.1%} nos de teste\n".format(tam_treino, acuracia)
    print(msg)

    # EXECUTA AGORA O AJUSTE NA AMOSTRA DE DADOS COMPLETA
    pipe.fit(dados.texto, dados.categoria)

    # ARMAZENA O MODELO TREINADO NO PIPE NO ARQUIVO FIT
    joblib.dump(pipe, modelo_arquivo)
# -----------------------------------------------


# FUNCAO PARA PREVER COM BASE NO MODELO TREINADO
def previsor(url, modelo_arquivo):
    # CARREGA O ARQUIVO FIT COM O MODELO TREINADO
    pipe = joblib.load(modelo_arquivo)

    # BUSCA AS PALAVRAS NO SITE DE INTERESSE EM SER CATEGORIZADO
    words = pre_processa(url)

    # USA O MODELO TREINADO PARA PREVER COM BASE NAS PALAVRAS EXTRAIDAS
    resp = pipe.predict([words])

    # MOSTRA NA TELA A CATEGORIA ENCONTRADA
    print("\nCategoria: %s \n" % resp[0])

    # MOSTRA NA TELA A PROBABILIDADE RELACIONADA COM CADA CATEGORIA
    resp = zip(pipe.classes_, pipe.predict_proba([words])[0])
    for cat, prob in resp:
        print("Categoria {:16s} com {:.1%} de probabilidade.".format(cat, prob))
# -----------------------------------------------


# FUNCAO QUE ACESSA A PAGINA E PROCESSA A LISTA DE PALAVRAS
def pre_processa(link):
    # CARREGA AS PALAVRAS MAIS COMUNS NO PORTUGUES DO BRASIL
    comuns = set(stopwords.words("portuguese"))

    # VARIAVEL QUE IRA ARMAZENA A LISTA DE PALAVRAS DA PAGINA
    palavras = ""

    # CASO NAO CONSIGA LER O SITE ENTAO IGNORA ESSA OBSERVACAO DA AMOSTRA
    try:
        # ACESSA O SITE PROGRAMATICAMENTE
        html = urlopen(link)

        # CARREGA AS PALAVRAS NO ENCODING CORRETO
        codigo = BeautifulSoup(html, "lxml")

        # ELIMINA TODAS AS REFERENCIAS A HTML, JAVASCRIPT E CSS
        for script in codigo(["script", "style"]):
            script.extract()

        # SEPARA O TEXTO
        texto = codigo.get_text()

        # QUEBRA O TEXTO PALAVRA POR PALAVRA
        palavras = texto.lower().split()

        # COMPARA COM O ARQUIVO DE PALAVRAS COMUNS E AS ELIMINA SE EXISTIREM
        palavras = ' '.join([p for p in palavras if not p in comuns])

    except Exception:
        print("erro na importacao")
        pass

    return palavras
# -----------------------------------------------


# FUNCAO QUE PREPARA OS DADOS ACESSANDO TODAS AS PAGINAS DA AMOSTRA DE TREINAMENTO
def prepara_dados(links_arquivo):

    # CRIA UMA MATRIZ QUE IRA ARMAZENAR TODAS AS CATEGORIAS E PALAVRAS DE CADA SITE
    linhas = []

    # LE O ARQUIVO COM A AMOSTRA DE DADOS
    if links_arquivo:
        links = pd.read_csv(links_arquivo, sep=';')
        links = ((r['link'], r['categ']) for i, r in links.iterrows())

    # MOSTRA NA TELA O PROGRESSO DA LEITURA
    print("Downloading and processing data...\n")
    cont = 0

    # LINK A LINK ACESSA O SITE, CAPTURA AS PALAVRAS E REALIZA TRATAMENTO
    for link, categ in links:
        print(cont)
        cont += 1
        words = pre_processa(link)
        print("{:6d} words in: \t {:.70}".format(len(words), link))
        linhas.append((link, categ, words))

    # ADICIONA TODAS AS PALAVRAS DOS SITES NO ARQUIVO DE DADOS E SALVA PARA POSTERIOR USO
    dados = pd.DataFrame(linhas)
    dados.columns = ['link', 'categoria', 'texto']
    dados.to_csv('arquivo_palavras.csv', sep=';', encoding='utf-8')
    print("\nSalvando arquivo_palavras.csv para uso futuro.")
    return dados
# -----------------------------------------------

'''
--------------------------------------------------------------------
CONFIGURACOES
--------------------------------------------------------------------
'''
# ARQUIVO QUE ARMAZENA O MODELO TREINADO------------------------------------
# modelorg = "fit_ln.txt"
modelo = "fit_mlp.txt"

# ARQUIVO COM A AMOSTRA DE DADOS--------------------------------------------
amostra = "sites.csv"
# amostra = "teste.csv"

# SITE PARA TESTE DE PREVISAO-----------------------------------------------
# teste_previsao = "http://g1.globo.com/tecnologia/noticia/playbor-abre-inscricoes-para-o-1-programa-de-pre-acelaracao-de-startups-de-jogos-digitais-no-brasil.ghtml"
# teste_previsao = "http://g1.globo.com/economia/mercados/noticia/ibovespa-futuro-desaba-10-apos-denuncias-contra-michel-temer.ghtml"
# teste_previsao = "http://g1.globo.com/sp/piracicaba-regiao/noticia/santa-barbara-doeste-confirma-primeiro-caso-de-gripe-h3n2.ghtml"
# teste_previsao = "https://educacao.uol.com.br/noticias/2017/05/20/enem-tem-mais-de-65-milhoes-de-inscritos-n-e-inferior-ao-do-ano-passado.htm"
# teste_previsao = "http://ciniro.blogspot.com.br/2016/11/10-ferramentas-pra-quem-gosta-de-ver-as.html"
teste_previsao = "https://educacao.uol.com.br/noticias/2017/05/17/educacao-sobre-genero-previne-violencia-sexual-diz-especialista.htm"

# DEFINE O TAMANHO DO CONJUNTO DE TREINO (80% treino, 20% teste)------------
tam_conjunto_treino = 0.8

'''
--------------------------------------------------------------------
EXECUCAO DO ALGORITMO
--------------------------------------------------------------------
'''
# CASO O ARQUIVO DO MODELO TREINADO NAO EXISTA ENTAO ELE EXECUTA O TREINAMENTO E DEPOIS O CRIA
if not os.path.isfile(modelo):
    amostra_preparada = prepara_dados(amostra)
    treina(amostra_preparada, modelo, tam_conjunto_treino)

# CASO O ARQUIVO DO MODELO TREINADO JA EXISTA ENTAO ELE APENAS EXECUTA A PREVISAO
if teste_previsao:
    previsor(teste_previsao, modelo)

                      