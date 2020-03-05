import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import timeit


sentiment140 = pd.read_csv('../training.1600000.processed.noemoticon.csv', encoding="ISO-8859-1")
sentiment140.columns = ['label', 'id', 'date', 'flag', 'user', 'text']
sentiment140['length'] = sentiment140['text'].apply(len)

# sns.barplot('label','length',data = sentiment140,palette='PRGn')
# plt.title('Average Word Length vs Label')
# plt.show()

# fig2 = sns.countplot(x= 'label',data = sentiment140)
# plt.title('Label Counts')
# plot = fig2.get_figure()
# plt.show()

def text_processing(tweet):
    
    #Generating the list of words in the tweet (hastags and other punctuations removed)
    def form_sentence(tweet):
        tweet_blob = TextBlob(tweet)
        return ' '.join(tweet_blob.words)
    new_tweet = form_sentence(tweet)
    
    #Removing stopwords and words with unusual symbols
    def no_user_alpha(tweet):
        tweet_list = [ele for ele in tweet.split() if ele != 'user']
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
        return clean_mess
    no_punc_tweet = no_user_alpha(new_tweet)
    
    #Normalizing the words in tweets 
    def normalization(tweet_list):
        lem = WordNetLemmatizer()
        normalized_tweet = []
        for word in tweet_list:
            normalized_text = lem.lemmatize(word,'v')
            normalized_tweet.append(normalized_text)
        return normalized_tweet
    
    return normalization(no_punc_tweet)

asd = timeit.default_timer()
sentiment140['tweet_list'] = sentiment140['text'][:100000].apply(text_processing)

print(text_processing(sentiment140['text'][1]))
print(sentiment140['text'].iloc[1])


msg_train, msg_test, label_train, label_test = train_test_split(sentiment140['text'], sentiment140['label'], test_size=0.2)

asdi = CountVectorizer(analyzer=text_processing)
basdi = TfidfTransformer(asdi)
print(timeit.default_timer() - asd)

