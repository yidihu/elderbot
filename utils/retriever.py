import re, copy, time, warnings
import numpy as np

from scipy.stats import entropy
from nltk import FreqDist, pos_tag
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
from gensim.models import LdaModel
from urllib.request import urlopen, Request   # get request to url
from bs4 import BeautifulSoup as bs


lemmer = WordNetLemmatizer()
stop_words = stopwords.words('english')
TfidfVec = TfidfVectorizer()
warnings.filterwarnings("ignore")

def process_html(post):
    url = post.a['href']   #href is the html tag where the url for a website is found
    print(url)
    
    req = Request(url, headers = {'User-Agent': 'Mozilla/5.0'})
    link_page = urlopen(req).read()
    link_soup = bs(link_page)
    sentences = link_soup.findAll("p")
    
    combine_para = ""
    for sentence in sentences:     # loop through each paragraph in an article
        txt = sentence.text.lower()
        txt = txt.replace('\r', '').replace('\t', '').replace('\n','').replace('\xa0', '')
        if "webmd" in txt or "all rights" in txt: break
        combine_para += " " + txt
    return combine_para.strip()

def process_query(query,topic):
    query = re.sub(r"[^\w\s]","", query)
    query = re.sub(r"\s+"," ", query)
    query = word_tokenize(query)
    search = ""
    for i in query:
        if i not in stop_words:
            search += i + "%20"

    # websites to crawl for reliable response
    websites = {'Med': 'https://www.webmd.com/search/search_results/default.aspx?query={}&page='.format(search),
                "politics": 'https://www.businesstimes.com.sg/search/{}?page='.format(search)}
    website_tags = {'Med': "search-results-doc-title",
                    'politics': "media-body"}
    return " ".join(query), websites[topic], website_tags[topic]

def get_posts(url,tag):
    req = Request(url, headers = {'User-Agent': 'Mozilla/5.0'})
    page = urlopen(req).read()

    #parsing facebook using html.parser from BeautifulSoup, to extract text values from html pages
    soup = bs(page, features="html.parser")  
    posts = soup.findAll("p", {"class": tag}) #extract text by finidng html tags. 
    return posts


def get_para(article_list, query):
    articles = [sent_tokenize(a) for a in article_list]
    articles = [[sent for sent in a if len(sent)>0] for a in articles if len(a)>0]
    result = []
    for a in articles:
        order = rank(a, query)
        try:
            idx = a.index(order[0])
        except:
            print(a)
            print(order)
        if idx==0: result.append(a[:3])
        elif idx==(len(a)-1): result.append(a[-3:])
        else: result.append(a[idx-1:idx+2])
    return result
    
    
def rank(corpus, query):
    """sort corpus in relevance order"""
    corpus = corpus + [query]
    tfidf = TfidfVectorizer().fit_transform(corpus)
    pairwise_similarity = tfidf * tfidf.T
    arr = pairwise_similarity.toarray()     
    np.fill_diagonal(arr, 0)
    input_idx = corpus.index(query)
    desc_order = (-arr[input_idx]).argsort()
    corpus = np.array(corpus)
    corpus = corpus[desc_order].tolist()
    corpus.remove(query)
    result = []
    for i in corpus:
        if i not in result:
            result.append(i)
    return result


## page > article > para: return another article; if no more article, increment k and do another search
def crawl_web(topic, query, k):
    query, web, tag = process_query(query,topic)
    results = get_posts(web +str(k),tag)
    article_text = [process_html(a) for a in results] 
    article_rank = rank(article_text, query) ## sorted articles by relevance
    paras = get_para(article_rank, query) ## para from each article that is relevant
    ans = [" ".join(p) for p in paras]
    return ans