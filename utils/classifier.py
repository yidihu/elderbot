from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import linear_model, naive_bayes, ensemble
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, nps_chat

import random, pickle, nltk, re, warnings
import pandas as pd
import numpy as np

# nltk.download('nps_chat')
# nltk.download("stopwords")
stop_words = stopwords.words('english')
warnings.filterwarnings("ignore")

#========================0. read ========================#
def save_dict(fillename, d):
    with open(fillename, 'w') as f:
        for k,v in d.items():
            f.write('{}:{}\n'.format(k,v))
        f.close()

def save_data(prefix, data):
    data[0].to_csv(prefix+'/train_text.csv', index=False)
    data[1].to_csv(prefix+'/test_text.csv', index=False)
    data[2].to_csv(prefix+'/train_label.csv', index=False)
    data[3].to_csv(prefix+'/test_label.csv', index=False)
        
def read_dict(filename):
    with open(filename, 'r') as f:
        pairs = f.readlines()
        f.close()
    x = [kv.split(":") for kv in pairs]
    if x[0][0].isdigit():
        x = [[int(p[0]),p[1][:-1]]for p in x]
    if x[0][1][0].isdigit():
        x = [[p[0],float(p[1])]for p in x]
    x = {p[0]:p[1] for p in x}
    return x

def read_data(prefix):
    train_text = pd.read_csv(prefix+'/train_text.csv',header=None)[0]
    test_text = pd.read_csv(prefix+'/test_text.csv',header=None)[0]
    train_label = pd.read_csv(prefix+'/train_label.csv',header=None)[0]
    test_label = pd.read_csv(prefix+'/test_label.csv',header=None)[0]
    return train_text, test_text, train_label, test_label


#========================1. Sampling========================#
def oversample(df,label,total_size):
    valid = df[(df['Label']==label) & (df['len']>1)].index
    df_new = df.copy(deep = True)
    cur = df_new[(df_new['Label']==label)].shape[0]
    i = 0
    while cur<total_size:
        if i == len(valid): i = 0
        row = df.iloc[valid[i],:]
        text = row['Text'].split(' ')
        new_text = ' '.join(random.sample(text, len(text)))
        row['Text'] = new_text
        
        df_new = df_new.append(row)
        df_new = df_new.drop_duplicates()
        cur = df_new[(df_new['Label']==label)].shape[0]
        i+=1
    return df_new.reset_index(drop=True)

def undersample(df,label,total_size):
    cur = df[(df['Label']==label)].shape[0]
    idx = df[df['Label']==label].sample(n=cur-total_size, replace=False, random_state=1).index
    df_new = df.drop(idx)
    return df_new.reset_index(drop=True)

def rebalance(df):
    counts = df['Label'].value_counts()
    size = int(counts.median())
    for i,c in counts.iteritems():
        if c>size:
            df = undersample(df,i,size)
        elif c<size:
            df = oversample(df,i,size)
    return df


#========================2. Modelling========================#
def get_models():
    count_vect = CountVectorizer()
    tfidf_vect = TfidfVectorizer()
    tfidf_vect_ngram = TfidfVectorizer(ngram_range=(2,3))
    tfidf_vect_chars = TfidfVectorizer(analyzer='char')
    vec_name = ['count','tdidf','ngram','char']
    vec_model = [count_vect, tfidf_vect, tfidf_vect_ngram, tfidf_vect_chars]
    
    nb_cls = naive_bayes.MultinomialNB()
    lr_cls = linear_model.LogisticRegression()
    rf_cls = ensemble.RandomForestClassifier()
    cls_name = ['NB','LR','RF']
    cls_model = [nb_cls,lr_cls,rf_cls]
    return vec_name, vec_model, cls_name, cls_model


def get_pipeline(prefix, train_text, train_label, vec_list, cls_list):
    ppl_list = []
    for c_name, c_model in cls_list:
        for v_name, v_model in vec_list:
            ppl = Pipeline([(v_name, v_model),(c_name, c_model)])
            ppl.fit(train_text, train_label)
            filename = prefix+c_name+v_name+ '.sav'
            pickle.dump(ppl, open(filename, 'wb'))
            ppl_list.append(filename)
    return ppl_list


def get_acc(ppl_list, test_text, test_label):
    acc_list = {}
    for filename in ppl_list:
        model = pickle.load(open(filename, 'rb'))
        predicted = model.predict(test_text)
        acc = np.mean(predicted == test_label)
        acc_list[filename]= acc
    return acc_list            


def setup(data,prefix):
    train_text, test_text, train_label, test_label = data
    vec_name, vec_model, cls_name, cls_model = get_models()
    vec_list = [(vec_name[i],vec_model[i]) for i in range(len(vec_name))]
    cls_list = [(cls_name[i],cls_model[i]) for i in range(len(cls_name))]
    
    ppl_list = get_pipeline(prefix, train_text, train_label, vec_list, cls_list)
    acc_list = get_acc(ppl_list, test_text, test_label)
    return acc_list

def classify(query, class_dict, acc_list, acc_min):
    """
    aggregate the classifications from different models using majority vote
    this is the final classification of a query
    """
    query = preprocess_text(query)
    preds, out = [], []
    for k,v in acc_list.items():
        if v>acc_min:
            model = pickle.load(open(k, 'rb'))
            pred = model.predict([query]) 
            preds.append(str(pred[0]))
        
    for i in range(len(class_dict)):
        cnt = preds.count(str(i))
        out.append(cnt)
    if sum(out)==0: return []
    
    pred = max(set(preds), key=preds.count)
    try:
        pred = int(pred)
        return class_dict[pred]
    except:
        return class_dict[pred]

def preprocess_text(x):
    """
    Preprocess the text inside the training corpus: lower case
    To be applied on the text of each row inside the training corpus.
    """
    x = word_tokenize(str(x))
    x = [word.lower() for word in x if "user" not in word and word!=".action" ]
    x = ['yes' if word in ['yes','yep','yea'] and "user" not in word and word!=".action" else word for word in x ]
    return ' '.join(x)

def preprocess_topic(x):
    pattern1 = r'[^A-Za-z0-9\-,‚.()#&\']+|[[0-9]+[)+]|R[0-9]+'
    pattern2 = r'[a-zA-Z0-9]+[\.|@][a-zA-Z0-9]+|\.[edu|com]+|From|Subject|\\n'
    pattern3 = r"\s(?=\W+)"
    pattern4 = r"\W(?=\W+\W+)"
    x = re.sub("please|thank|thanks","",x, flags=re.IGNORECASE)
    x = re.sub(pattern1, ' ', x)
    x = re.sub(pattern2, '', x)
    x = re.sub(r"[\s]+", ' ', x)
    x = re.sub(r"-+", '-', x).strip()

    x = word_tokenize(str(x))
    x = [word.lower() for word in x if word.lower() not in stop_words]
    x = ' '.join(x)
    x = re.sub(pattern3, '', x)
    x = re.sub(pattern4, '', x)
    x = re.sub("'s ", '', x)
    return x

#========================3. Intent Classifier========================#
class intent_classifier:
    def __init__(self, acc_min):
        self.acc_min = acc_min
        self.prefix = "./assets/intent"
        self.id_class_dict = read_dict(self.prefix+"/id_class_dict.txt")
        self.acc_list = read_dict(self.prefix+"/acc_list.txt")
        self.data = read_data(self.prefix)
       
    def predict(self, query):
        return classify(query, self.id_class_dict, self.acc_list, self.acc_min)
    
    def reset(self):
        self.data, self.id_class_dict = intent_get_data()
        self.acc_list = setup(self.data,"assets/intent/")
        save_dict(self.prefix + "/acc_list.txt", self.acc_list)
        save_dict(self.prefix + "/id_class_dict.txt", self.id_class_dict)
        save_data(self.prefix, self.data)
    
    def intent_get_data():
        data, id_class_dict = intent_get_corpus()
        return train_test_split(data.Text, data.Label), id_class_dict

    def intent_get_corpus():
        posts = nltk.corpus.nps_chat.xml_posts()
        data = [[p.text,p.get('class')] for p in posts]
        chat = pd.DataFrame(data, columns = ['Text','Class'])
        chat['Class'] = chat.Class.apply(lambda x: intent_group(x))

        class_id_dict = {}
        classes = chat.Class.unique().tolist()
        for i in range(len(classes)):
            class_id_dict[classes[i]] = i
        chat['Label'] = chat['Class'].apply(lambda x: class_id_dict[x])
        for i in range(2):
            chat['Text'] = chat['Text'].apply(lambda x: preprocess_text(x))
            chat = chat[chat['Text']!='']
            chat = chat.drop_duplicates()
            id_class_dict = {v:k for k,v in class_id_dict.items()}

        chat['len'] = chat['Text'].apply(lambda x: len(x.split(' ')))
        chat = chat.reset_index(drop=True)
        chat = rebalance(chat)
        return chat, id_class_dict

    def intent_group(x):
        """
        Group the label of training corpus into fewer categories to have better model performance
        To be applied on the label of each row inside the training corpus
        """
        if x in ['whQuestion','ynQuestion']: return 'Question'
        if x in ['Accept', 'yAnswer', 'Continuer']: return 'Accept'
        if x in ['Reject','nAnswer']: return 'Reject'
        if x in ['System','Other', 'Statement','Emphasis']: return 'Other'
        return x


#========================4. Topic Classifier========================#
class topic_classifier:
    def __init__(self, acc_min):
        self.acc_min = acc_min
        self.prefix = "./assets/topic"
        self.id_class_dict = read_dict(self.prefix + "/id_class_dict.txt")
        self.acc_list = read_dict(self.prefix + "/acc_list.txt")
        self.data = read_data(self.prefix)
        
    def predict(self, query):
        query = preprocess_topic(query)
        return classify(query, self.id_class_dict, self.acc_list, self.acc_min)
    
    def reset(self):
        self.id_class_dict, self.data = get_topic_corpus()
        self.acc_list = setup(self.data,"assets/topic/")
        save_dict(self.prefix + "/acc_list.txt", self.acc_list)
        save_dict(self.prefix + "/id_class_dict.txt", self.id_class_dict)
        save_data(self.prefix, self.data)
    
    def group_topics(x):
        if x in ['talk.politics.mideast','talk.politics.misc','talk.politics.guns']: return 'politics'
        if x in ['sci.space','sci.electronics','sci.crypt']: return 'SnT'
        if x in ['sci.med']: return "Med"
        if x in ['soc.religion.christian', 'talk.religion.misc', 'alt.atheism']: return 'religion'
        if x in ['rec.sport.hockey','rec.sport.baseball']: return 'sports'
        return 'others'

    def get_topic_corpus():
        topic = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
        topic = topic[['content','target_names']].drop_duplicates().dropna()

        topic_id_dict = {}
        topics = topic.target_names.unique().tolist()
        for i in range(len(topics)):
            topic_id_dict[topics[i]] = i
        topic['target'] = topic['target_names'].apply(lambda x: topic_id_dict[x])
        topic['content'] = topic['content'].apply(lambda x: preprocess_topic(x))
        topic = topic.rename(columns={"content":"Text","target_names":"Class",
                                      "target":"Label"})
        topic["len"] = topic['Text'].apply(lambda x: len(x.split(' ')))

        topic = topic.reset_index(drop=True)
        topic = rebalance(topic)
        topic = topic.drop_duplicates()

        topic['Class'] = topic['Class'].apply(lambda x: group_topics(x))
        topic_id_dict = {}
        topics = topic.Class.unique().tolist()
        for i in range(len(topics)):
            topic_id_dict[topics[i]] = i
        topic['Label'] = topic['Class'].apply(lambda x: topic_id_dict[x])
        topic = topic.reset_index(drop=True)
        topic = rebalance(topic)
        topic = topic.drop_duplicates()

        id_topic_dict = {v:k for k,v in topic_id_dict.items()}
        return id_topic_dict, topic