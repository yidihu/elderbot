from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import pandas as pd
stop_words = stopwords.words('english')
history_file = "./assets/history.csv"

class analysis:
    def __init__(self):
        self.data = pd.read_csv(history_file)
        self.data_recent = self.data.iloc[-100:]
        
    def recent_pos(self,n=10):
        return self.data_recent.nlargest(n, ['score'])
    
    def recent_neg(self,n=10):
        return self.data_recent.nsmallest(n, ['score'])
    
    def sent_plot(self):
        plt.figure(figsize = (12, 6))
        plt.plot(self.data_recent.score)
        plt.ylabel("Sentiment Score", fontsize = 12)
        plt.xlabel("Query Number", fontsize = 12)
        plt.title("Change in Sentiment Score", fontsize = 16)
        plt.plot
        return plt
    
    def word_cloud(self):
        all_queries = ""
        for i in self.data_recent['query']:
            oneQ = i.lower()
            all_queries += ' ' + oneQ

        wordcloud = WordCloud(width = 1500, height = 500, background_color = "white", 
                              stopwords = stop_words, min_font_size = 10).generate(all_queries)

        plt.figure(figsize = (8, 8), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.tight_layout(pad = 0)
        plt.title("Most commonly used words", fontsize = 20)
        plt.plot
        return plt