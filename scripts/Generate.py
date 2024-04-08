import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVR
import textstat
from textblob import TextBlob
import Util as ut

def mean_absolute_error(y_true, y_pred):
  '''
  Calcula o Mean Absolute Error (MAE), entre os valores verdadeiros (y_true) e os valores previstos (y_pred)

  Args:
    y_true: Uma lista ou array Numpy dos valores verdadeiros.
    y_pred: Uma lista ou array NumPy dos valores previstos.

  Returns:
    mae: O Mean Absolute Error entre y_true e y_pred
  '''
  if len(y_true) != len(y_pred):
    raise ValueError('Os tamanhos de y_true e y_pred devem ser iguais')
  absolute_Errors =[abs(true-pred) for true, pred in zip(y_true, y_pred)]
  mae = sum(absolute_Errors) / len(y_true)
  return mae

def generate_mae_teste_fig(name):
    '''
    Gera um imagem do MAE de teste dos resultados

    Args:
        name: Nome do arquivo CSV
    '''
    filename = 'data/neo/{}.csv'.format(name)
    df_full = pd.read_csv(filename)
    
    # nova coluna
    df_full['context'] = df_full['title'] + df_full['description']
    df_full['context'] = df_full['context'].astype(str)
    df_full = df_full.drop(['created', 'issuekey', 'title', 'description'], axis=1)
    
    # remoção de outlier
    mean = df_full['storypoints'].mean()
    std_dev = df_full['storypoints'].std()
    outlier_cutoff = 2 * std_dev
    df_clean = df_full[(df_full['storypoints'] >= mean - outlier_cutoff) & (df_full['storypoints'] <= mean + outlier_cutoff)]
    
    #pré-processamento do texto
    df_clean['context'] = df_clean['context'].apply(ut.remover_stopwords)
    df_clean['context'] = df_clean['context'].apply(ut.remover_urls)
    df_clean['context'] = df_clean['context'].apply(ut.remover_html_tags)
    df_clean['context'] = df_clean['context'].apply(ut.remover_palavras_com_numeros)
    df_clean['context'] = df_clean['context'].apply(ut.remover_pontuacoes)
    df_clean['context'] = df_clean['context'].apply(ut.remover_caracteres_especiais)
    df_clean['context'] = df_clean['context'].apply(ut.remover_espacos_branco)
    
    # separação treino e teste
    percent_treino = 0.7
    num_linhas_treino = int(len(df_clean) * percent_treino)
    dados_treino = df_clean.iloc[:num_linhas_treino]
    dados_teste = df_clean.iloc[num_linhas_treino:]
    
    # resultado story point médio
    storypoint_medio = df_clean['storypoints'].mean()
    lista_y_pred = [storypoint_medio] * len(dados_teste)
    mae_media_sp = mean_absolute_error(dados_teste['storypoints'], lista_y_pred)
    
    df_results = pd.DataFrame(data=[['mae_media_sp', mae_media_sp, 'red']], columns=['modelo', 'MAE Teste', 'cor'])
    
    # bow com SVR
    vec = CountVectorizer()
    bow_treino = vec.fit_transform(dados_treino['context'])
    bow_df_treino = pd.DataFrame(bow_treino.toarray(), columns=vec.get_feature_names_out())
    bow_teste = vec.transform(dados_teste['context'])
    bow_df_teste = pd.DataFrame(bow_teste.toarray(), columns=vec.get_feature_names_out())
    model = SVR(kernel='linear')
    model.fit(bow_df_treino, dados_treino['storypoints'])
    y_pred = model.predict(bow_df_teste)
    mae_bow = mean_absolute_error(dados_teste['storypoints'], y_pred)
    df_results = df_results.append({'modelo':'mae_bow', 'MAE Teste': mae_bow, 'cor':'green'}, ignore_index=True)
    
    # tf-idf com SVR
    vec = TfidfVectorizer(max_features=50)
    tfidf_matrix_treino = vec.fit_transform(dados_treino['context'])
    tfidf_matrix_teste = vec.transform(dados_teste['context'])
    model = SVR(kernel='linear')
    model.fit(tfidf_matrix_treino, dados_treino['storypoints'])
    y_pred = model.predict(tfidf_matrix_teste)
    mae_tfidf = mean_absolute_error(dados_teste['storypoints'], y_pred)
    df_results = df_results.append({'modelo':'mae_tfidf', 'MAE Teste': mae_tfidf, 'cor':'blue'}, ignore_index=True)
    
    ## =-=-=-=- legibility -=-=-=-
    # carregando os dados
    df_full = pd.read_csv(filename)
    # nova coluna
    df_full['context'] = df_full['title'] + df_full['description']
    df_full['context'] = df_full['context'].astype(str)
    df_full = df_full.drop(['created', 'issuekey', 'title', 'description'], axis=1)
    # remoção outlier
    mean = df_full['storypoints'].mean()
    std_dev = df_full['storypoints'].std()
    outlier_cutoff = 2 * std_dev
    df_clean = df_full[(df_full['storypoints'] >= mean - outlier_cutoff) & (df_full['storypoints'] <= mean + outlier_cutoff)]
    # features legibility
    colunas = ['gunning_fog', 'polarity','subjectivity']
    df_clean['gunning_fog'] = df_clean['context'].apply(textstat.gunning_fog)
    df_clean['polarity'] = df_clean['context'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df_clean['subjectivity'] = df_clean['context'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    colunas = ['gunning_fog', 'polarity','subjectivity']
    # separação entre treino e teste
    dados_treino = df_clean.iloc[:num_linhas_treino]
    dados_teste = df_clean.iloc[num_linhas_treino:]
    # modelo legibility
    model = SVR()
    model.fit(dados_treino[colunas], dados_treino['storypoints'])
    y_pred = model.predict(dados_teste[colunas])
    mae_tfidf = mean_absolute_error(dados_teste['storypoints'], y_pred)
    df_results = df_results.append({'modelo':'mae_legibility', 'MAE Teste': mae_tfidf, 'cor':'orange'}, ignore_index=True)
    ## =-=-=-=- legibility -=-=-=-
    
    # salvar grafico MAE
    df_results = df_results.sort_values(by='MAE Teste')
    plt.figure(figsize=(10, 10))
    plt.xticks(rotation=45, ha='right')
    plt.title('MAE Teste do Projeto {}'.format(name))
    plt.xlabel('Modelos')
    plt.ylabel('MAE')
    plt.bar(df_results['modelo'], df_results['MAE Teste'],color=df_results['cor'])
    
    plt.savefig('figuras' + '/' + name)
    
if __name__=='__main__':
    print('Iniciou programa!')
    
    projetos = ['7764', '4456656', '21149814', '10171270', 
                '12450835','28847821', '10171280', '1304532', 
                '2670515', '10171263', '10174980', '12584701', 
                '12894267', '14052249', '14976868', '1714548', 
                '19921167', '2009901', '23285197', '250833',
                '28419588', '28644964', '3828396', '3836952', 
                '5261717', '6206924', '7071551',  '7128869', 
                '734943', '7603319', '7776928', '15502567', 
                '10152778']
    
    for name in projetos:
        print('Iniciou projeto: {}'.format(name))
        generate_mae_teste_fig(name)
        print('Finalizou projeto: {}'.format(name))
    print('Finalizou programa!')