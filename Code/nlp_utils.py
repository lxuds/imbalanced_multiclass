import re
import sys
import nltk
import textacy
import string
import re


from textacy import preprocess_text
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textacy.preprocess import (remove_accents, 
                                fix_bad_unicode, transliterate_unicode,normalize_whitespace,
                                )
import ftfy
reload(sys)
sys.setdefaultencoding('utf-8')


################
## Stop Words ##
################
stopwords = nltk.corpus.stopwords.words("english")
stopwords = set(stopwords)


##############
## Stemming ##
##############
#if config.stemmer_type == "porter":
#    english_stemmer = nltk.stem.PorterStemmer()
#elif config.stemmer_type == "snowball":
english_stemmer = nltk.stem.SnowballStemmer('english')


'''
def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed
'''
############
## TF-IDF ##
############
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

token_pattern = r"(?u)\b\w\w+\b"
#token_pattern = r'\w{1,}'
#token_pattern = r"\w+"
#token_pattern = r"[\w']+"
tfidf__norm = "l2"
tfidf__max_df = 0.9

tfidf__min_df = 1

def getTFV(token_pattern = token_pattern,
           norm = tfidf__norm,
           max_df = tfidf__max_df,
           min_df = tfidf__min_df,
           ngram_range = (1, 3),
           vocabulary = None,
           stop_words = 'english'):
    tfv = StemmedTfidfVectorizer(min_df=min_df, max_df=max_df, max_features=None,
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
                                 stop_words = stop_words, norm=norm, vocabulary=vocabulary)
    return tfv



#########
## BOW ##
#########
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(CountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

token_pattern = r"(?u)\b\w\w+\b"
#token_pattern = r'\w{1,}'
#token_pattern = r"\w+"
#token_pattern = r"[\w']+"
bow__max_df = 0.75
bow__min_df = 1
def getBOW(token_pattern = token_pattern,
           max_df = bow__max_df,
           min_df = bow__min_df,
           ngram_range = (1, 3),
           vocabulary = None,
           stop_words = 'english'):
    bow = StemmedCountVectorizer(min_df=min_df, max_df=max_df, max_features=None,
                                 strip_accents='unicode', analyzer='word', token_pattern=token_pattern,
                                 ngram_range=ngram_range,
                                 stop_words = stop_words, vocabulary=vocabulary)
    return bow



def drop_html(html):
    return BeautifulSoup(html,"lxml").get_text(separator=" ",strip=True)


def read_clean_text(file):
    f = open('./data/FixedJudgements/'+ file + '.txt')
    raw = f.read()
    text = drop_html(raw)
    text = textacy.preprocess.unpack_contractions(text)
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    text = regex.sub(' ', text)
    text = text.replace('\n', ' ')
    text = preprocess_text(text,lowercase=True, fix_unicode=True, no_numbers=True)
    return text
