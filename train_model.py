import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense
from tensorflow.keras.layers import Dropout, SpatialDropout1D, Embedding
from tensorflow.keras.optimizers import Adam
import process_data

attr, train, test = process_data.formatted_data()

def glove_data(save=False, folder="processed_data/"):
    glove_map = {}
    with open('data/glove.6B.300d.txt') as f:
        for line in f:
            vals = line.split()
            word = vals[0]
            vect = vals[1:]
            glove_map[word] = vect
    if save:
        pickle.dump(glove_map, open(folder + "glove_map.pyb", "wb"))    
    return glove_map

def glove_matrix():
    glove_map = glove_data()
    matrix = np.zeros((attr[0], 300))
    word_index = pickle.load(open("process_data/word_index.pyb", "rb"))
    for word, i in word_index():
        glove_vect = glove_map.get(word)
        if glove_vect != None:
            matrix[i] = glove_vect
    return matrix

def train(save_all=False, folder="process_data/"):
    matrix = glove_matrix()
    X_train, y_train = train
    model = Sequential([
        Embedding(input_dim=attr[0], output_dim=300, weights=[matrix], input_length=avg, trainable=False),
        SpatialDropout1D(0.4),
        Conv1D(128, 5, activation='relu'),
        Bidirectional(LSTM(128, dropout=0.4)),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_split = 0.1, batch_size=1024, epochs=12, verbose=2)
    if save_all:
        model.save(folder + 'sentiment_model.h5')
        pickle.dump(model.history.history, open(folder + "history.pyb", 'wb'))
    return model

def plot_accur_hist(history, save=False, folder='pics/'):
    plt.plot(history['accuracy'], c='#00d2ff', label='Train Accuracy')
    plt.plot(history['val_accuracy'], c='#fb2e01', label='CV Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Model Accuracy')
    plt.legend()
    if save:
        plt.savefig(folder + 'accuracy_history.png')
    plt.show()

def plot_loss_hist(history, save=False, folder='pics/'):
    plt.plot(history['loss'], c='#00d2ff', label='Train Loss')
    plt.plot(history['val_loss'], c='#fb2e01', label='CV Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Model Loss')
    plt.legend()
    if save:
        plt.savefig(folder + 'loss_history.png')
    plt.show()