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
        df.to_csv(folder + "processed_data.csv", encoding='utf-8', index=False)
    print("Data Cleaned")
    return df

def default_clean():
    data = getData()
    cleaned = clean_data(data, True)
    return cleaned

def get_avg_length(data):
    avg_len = round(sum(data['Tweet'].apply(lambda x: len(x)))/len(data))
    return avg_len

def save_pkl(save_var, name, folder='processed_data/'):
    for var, n in zip(save_var, name):
        pickle.dump(var, open(folder + n + '.pyb', 'wb'))

def load_pkl(name, folder='processed_data/'):
    res = []
    for n in name:
        res += [pickle.load(folder + n +'pyb', 'rb')]
    return res

def tokenize_data(train, test, size=1):
    if os.path.isfile("processed_data/tokenizer_half.pyb"):
        tokenizer = pickle.load(open("processed_data/tokenizer_half.pyb", "rb"))
    else:
        tokenizer = Tokenizer()
        sub_r = round(len(train)*size)
        tokenizer.fit_on_texts(train[:sub_r].Tweet)
    vocab_size = len(tokenizer.word_index) + 1
    print("Data Tokenized")
    return vocab_size, tokenizer

def _data_padding(tokenizer, train, test, avg, save=False):
    X_train = pad_sequences(tokenizer.texts_to_sequences(train.Tweet), maxlen=avg)
    X_test = pad_sequences(tokenizer.texts_to_sequences(test.Tweet), maxlen=avg)
    y_train = list(train.Sentiment)
    y_test = list(test.Sentiment)

    save_var = (X_train, X_test, y_train, y_test, tokenizer)
    name = ('X_train', 'X_test', 'y_train', 'y_test', 'tokenizer')
    if save:
        save_pkl(save_var, name)
    return (X_train, y_train), (X_test, y_test)

def pad_data(tokenizer, train, test, avg, save=False, folder="processed_data/")
    if os.path.isfile(folder + "X_train.pyb"):
        name = ('X_train', 'y_train', 'X_test', 'y_test')
        res = load_pkl(name)
        train, test = (res[0], res[1]), (res[2], res[3])
    else:
        train, test = _data_padding(tokenizer, train, test, avg)
    print("Data Padded")
    return train, test

def formatted_data(data=None, size=1, save=False, folder='processed_data/'):
    if not data:
        if os.path.isfile(folder + "processed_data.csv"):
            data = pd.read_csv(folder + "processed_data.csv").fillna('')
        else:
            data = default_clean()
    avg = get_avg_length(data)

    if os.path.isfile(folder + "test.csv") and os.path.isfile(folder + "train.csv"):
        train = pd.read_csv(folder + "train.csv").fillna('')
        test = pd.read_csv(folder + "test.csv").fillna('')
    else:
        train, test = train_test_split(data, test_size=0.2, random_state=12)
        train.to_csv(folder + "train.csv", encoding='utf-8', index=False)
        test.to_csv(folder + "test.csv", encoding='utf-8', index=False)

    vocab_size, tokenizer = tokenize_data(train, test, avg, size=size, save=save)
    train, test = pad_data(tokenizer, train, test, avg)

    print("Data Processing Completed")
    return (vocab_size, avg), train, test
