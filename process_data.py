import pandas as pd
import re, pickle, os
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def getData(save=False):
    LABEL = ['Sentiment', 'ID', 'Date', 'Query', 'Username', 'Tweet']
    data = pd.read_csv("data/training.csv", encoding='ISO-8859-1', header=None, names=LABEL)
    data = data[['Tweet', 'Sentiment']]
    if save: data.to_csv("processed_data/data.csv")
    return data

def clean_text(tweet):
    clean_regex = r'@\S+|https?:\S+|\S+\.(com|edu|net)|[^a-zA-Z0-9 ]+'
    stop = stopwords.words('english')
    stem = SnowballStemmer('english')
    tweet = re.sub(clean_regex, '', tweet.lower()).strip()
    res = []
    for word in tweet.split():
        if word not in stop:
            res.append(stem.stem(word))
    return ' '.join(res)

def clean_data(df, save=False):
    df['Tweet'] = df['Tweet'].apply(lambda col: clean_text(col))
    if save: df.to_csv("processed_data/processed_data.csv", encoding='utf-8')
    return df

def default(get=False):
    data = getData()
    cleaned = clean_data(data, True)
    if get: return cleaned

def get_avg_length(data):
    avg_len = round(sum(data['Tweet'].apply(lambda x: len(x)))/len(data))
    return avg_len

def process_data(data=None, save=False)
    if not data:
        if os.path.isfile("processed_data/processed_data.csv"):
            data = pd.read_csv("processed_data/processed_data.csv").fillna('')
        else:
            data = default(True)

    train, test = train_test_split(data, test_size=0.7, random_state=12)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train.Tweet)

    vocab_size = len(tokenizer.word_index) + 1
    avg = get_avg_length(data)

    X_train = pad_sequences(tokenizer.texts_to_sequences(train.Tweet),maxlen=avg)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test.Tweet), maxlen=avg)
    y_train = list(train.Sentiment)
    y_test = list(test.Sentiment)

    if save:
        pickle.dump(X_train, open("processed_data/X_train.pyb", "wb"))
        pickle.dump(X_test, open("processed_data/X_test.pyb", "wb"))
        pickle.dump(y_train, open("processed_data/y_train.pyb", "wb"))
        pickle.dump(y_test, open("processed_data/y_test.pyb", "wb"))

    return (vocab_size, avg), (X_train, y_train), (X_test, y_test)

