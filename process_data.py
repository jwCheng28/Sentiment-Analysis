import pandas as pd
import re
import pickle
import os
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def getData(save=False, folder='processed_data/'):
    LABEL = ['Sentiment', 'ID', 'Date', 'Query', 'Username', 'Tweet']
    data = pd.read_csv("data/training.csv", encoding='ISO-8859-1', header=None, names=LABEL)
    data = data[['Tweet', 'Sentiment']]
    data.loc[data.Sentiment == 4, "Sentiment"] = 1
    if save: 
        data.to_csv(folder + "data.csv")
    print("Successfully loaded Data")
    return data

def clean_text(tweet):
    clean_regex = r'@\S+|https?:\S+|\S+\.[a-z]{3}|[^a-zA-Z0-9]+'
    stop = stopwords.words('english')
    stem = SnowballStemmer('english')
    tweet = re.sub(clean_regex, ' ', tweet.lower()).strip()
    res = []
    for word in tweet.split():
        if word not in stop:
            res.append(stem.stem(word))
    return ' '.join(res)

def clean_data(df, save=False, folder='processed_data/'):
    df['Tweet'] = df['Tweet'].apply(lambda col: clean_text(col))
    if save: 
        df.to_csv(folder + "processed_data.csv", encoding='utf-8')
    print("Data Cleaned")
    return df

def default(get=False):
    data = getData()
    cleaned = clean_data(data, True)
    if get: 
        return cleaned

def get_avg_length(data):
    avg_len = round(sum(data['Tweet'].apply(lambda x: len(x)))/len(data))
    return avg_len

def save_pkl(save_var, name, folder='processed_data/'):
    for var, n in zip(save_var, name):
        pickle.dump(var, open(folder + n + '.pyb', 'wb'))

def tokenize_data(train, test, avg, save=False):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train.Tweet)
    vocab_size = len(tokenizer.word_index) + 1

    X_train = pad_sequences(tokenizer.texts_to_sequences(train.Tweet), maxlen=avg)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test.Tweet), maxlen=avg)
    y_train = list(train.Sentiment)
    y_test = list(test.Sentiment)

    save_var = (X_train, X_test, y_train, y_test)
    name = ('X_train', 'X_test', 'y_train', 'y_test')
    if save:
        save_pkl(save_var, name)
    print("Data tokenized and padded")
    return vocab_size, (X_train, y_train), (X_test, y_test)

def process_data(data=None, save=False, folder='processed_data/'):
    if not data:
        if os.path.isfile(folder + "processed_data.csv"):
            data = pd.read_csv(folder + "processed_data.csv").fillna('')
        else:
            data = default(True)

    train, test = train_test_split(data, test_size=0.2, random_state=12)
    train.to_csv(folder + "train.csv", encoding='utf-8')
    test.to_csv(folder + 'test.csv', encoding='utf-8')

    avg = get_avg_length(data)
    vocab_size, train, test = tokenize_data(train, test, avg, save=save)
    print("Data Processing Completed")
    return (vocab_size, avg), train, test


if __name__ == "__main__":
    process_data(save=True)