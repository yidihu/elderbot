from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
from random import choice
from utils.retriever import crawl_web
from utils.classifier import topic_classifier, intent_classifier
from utils.posterior import analysis
import re,csv
import pandas as pd

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

POSITIVE = ["I am glad you are well!", "That's great!", "Well done!", 
            "So thrilled to hear from you!", "You should be proud of yourself!"]
NEGATIVE = ["Cheer up! You can talk to me about it.", "Share your feelings with me. I am here to listen!", \
            "Not to worry, bad days will pass.", "Stay positive! :)"]

history_file = "./assets/history.csv"
sia = SentimentIntensityAnalyzer()
find_topic = topic_classifier(0.5)
find_intent = intent_classifier(0.5)

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return choice(GREETING_RESPONSES)
        
def update_history(query, response, t, score):    
    with open(history_file, "a", newline = '') as f:
        writer = csv.writer(f)
        writer.writerow([query, response, score, t]) 
        
def clarify_ans(history):
    last  = history.iloc[len(history) - 1]
    query = last.query
    ans = last.response
    t = last.timestamp
    return "On {}, \n --> You said to me: {}\n And I answered: {}".format(t, query, ans)

def emotion_ans(user_input):
    sentiment = sia.polarity_scores(user_input)["compound"]
    if sentiment > 0:
        return choice(POSITIVE)
    if sentiment < 0:
        return choice(NEGATIVE)
    
class bot:
    def __init__(self, bot_name="Bob"):
        self.flag = True
        self.name = bot_name
        self.summary = analysis()
        self.search_page = 1
        self.cues = {"last conversation": "see our last conversation",
                     "Bye":"exit", "analysis": "see our conversation summary"}
        self.response = "\n{}:  ".format(self.name)
        self.talk()
    
    
    def talk(self):
        self.flag = True
        self.__start()
        self.__interact()
            
    def __reply(self,msg):
        print(self.response+msg)
    
    def __start(self):
        print("{}: Good day! My name is {}. Ask away!".format(self.name.upper(), self.name))
        for k,v in self.cues.items():
            print("{}If you want to {}: type '{}'".format(" "*(len(self.name)+2),v,k))
    
    
    def __clarify(self, user_input):
        history = pd.read_csv(history_file)
        clar_return = clarify_ans(history)
        self.sent_input = sia.polarity_scores(user_input)["compound"]
        self.__reply(clar_return)
        
        
    def __comfort(self, user_input):
        emotion_res = emotion_ans(user_input)
        if emotion_res: 
            print(self.response + emotion_res)
        update_history(user_input,emotion_res, self.timing, self.sent_input)
        
        
    def __answer(self):
        user_input = self.user_qtn
        
        topic = find_topic.predict(user_input)
        self.__reply("I see you are asking a question about {}!".format(topic))
        
        if topic not in ["Med","politics"]:
            print(self.response+"I am so sorry my database currently does not cover this topic yet..")
        else:
            self.__reply("I find some useful websites! Let me look into them :)")
            ans = crawl_web(topic, user_input, self.search_page)
            self.ans = ans
            self.__answer_text()
            
    def __answer_text(self):
        for i in range(3):
            self.__reply(self.ans[i])
        response = " ".join(self.ans[:3])
        score = sia.polarity_scores(self.user_qtn)["compound"]
        update_history(self.user_qtn, response, self.timing, score)
        self.ans = self.ans[3:]
        self.__reply("Is this response what you were looking for?")
    
    def __analyze(self):
        self.__reply("Let's see recent moments of happiness :)")
        print(self.summary.recent_pos()[['query','score']])
        
        self.__reply("These moments you sounds tired, but we are always with you :)")
        print(self.summary.recent_neg()[['query','score']])
        
        self.__reply("Your recent sentiment score is like this:")
        plt = self.summary.sent_plot()
        plt.show()
        
        self.__reply("Your like to use these words recently:")
        plt = self.summary.word_cloud()
        plt.show()
        
            
    def __interact(self):
        store_user_questions = []
        store_previous_corpus = {}     # if user talk about same topic, no need crawl again

        while(self.flag==True):
            # User's Input
            print("YOU:", end=" ")
            user_input = input()
            user_input=user_input.lower()
            self.timing = datetime.now().strftime("%d/%m/%Y, %H: %M: %S")
            intent = find_intent.predict(user_input)

            # If user wishes to Clarify
            if (user_input == "last conversation"): self.__clarify(user_input)
            elif (user_input == "analysis"): self.__analyze()
            elif(intent=="Greet"): print(self.response+choice(GREETING_RESPONSES))
            elif (intent=="Emotion"):self.__comfort(user_input)
            elif intent in ["Accept","Reject","Question"]:
                if intent == "Reject":
                    self.__reply("I'm sorry the answer is not good. I will improve on it")
                    if len(self.ans)>2: self.__answer_text()
                    else: 
                        self.search_page+=1
                        self.__answer(user_input)
                        
                elif intent == "Accept":
                    self.__reply("Brilliant!")
                    self.flag = False

                else: 
                    self.user_qtn = user_input
                    self.__answer()
                    
                    
            elif(intent=='Bye'):
                self.flag=False
                self.__reply("Bye! take care..")
            elif("thank" in user_input):
                self.flag=False
                self.__reply("You are welcome..")
            else:
                msg = "I am sorry! I don't understand you"
                score = sia.polarity_scores(self.user_qtn)["compound"]
                update_history(user_input, msg, self.timing, score)
                self.__reply(msg)
                