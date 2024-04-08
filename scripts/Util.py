import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
# processamento do texto
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words("english"))

def remover_stopwords(texto):
    palavras = word_tokenize(texto)
    palavas_sem_stopwords = [palavra for palavra in palavras if palavra.lower() not in stop_words]
    return ' '.join(palavas_sem_stopwords)

def remover_urls(texto):
    return re.sub(r'http\S+|www\S+', ' ', texto)

def remover_html_tags(texto):
    return re.sub(r'<[^>]+>', ' ', texto)

def remover_palavras_com_numeros(texto):
    return re.sub(r'\b\w*\d\w*\b', ' ', texto)

def remover_pontuacoes(texto):
    return re.sub(r'[^\w\s]', ' ', texto)

def remover_caracteres_especiais(texto):
    return re.sub(r'[^A-Za-z0-9\s]', '', texto)

def remover_espacos_branco(texto):
    return re.sub(r'\s+', ' ', texto)