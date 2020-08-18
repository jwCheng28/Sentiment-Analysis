import pickle
from os import path
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense
from tensorflow.keras.layers import Dropout, SpatialDropout1D, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import process_data

attr, train, test = process_data.formatted_data()
vocab_size, avg_len = attr

def glove_data(save=False, folder="processed_data/"):
    if path.isfile(folder + "glove_map.pyb"):
        glove_map = pickle.load(open(folder + "glove_map.pyb", "rb"))
    else:
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
    matrix = np.zeros((vocab_size, 300))
    word_index = pickle.load(open("processed_data/word_index.pyb", "rb"))
    for word, i in word_index():
        glove_vect = glove_map.get(word)
        if glove_vect != None:
            matrix[i] = glove_vect
    return matrix

def _create_model(vocab_size, avg_len, matrix):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=300, weights=[matrix], input_length=avg_len, trainable=False),
        SpatialDropout1D(0.4),
        Conv1D(128, 5, activation='relu'),
        Bidirectional(LSTM(128, dropout=0.4)),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    return model

def train(save_all=False, folder="processed_data/"):
    if path.isfile(folder + 'sentiment_model.h5'):
        model = load_model(folder + 'sentiment_model.h5')
    else:
        matrix = glove_matrix()
        X_train, y_train = np.asarray(train)
        model = _create_model(vocab_size, avg_len, matrix)
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_split = 0.1, batch_size=1024, epochs=12, verbose=2)
        if save_all:
            model.save(folder + 'sentiment_model.h5')
            pickle.dump(model.history.history, open(folder + "history.pyb", 'wb'))
    return model

def test_accuracy(model, ret=False):
    X_test, y_test = np.asarray(test)
    pred = model.predict(X_test)
    int_pred = np.rint(pred).reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    accuracy = np.mean(int_pred == y_test)
    print("Accuracy on Test Data: ", round(accuracy, 4), '%')
    if ret:
        return accuracy

def plot_template(history, ref, save=False, folder='pics/'):
    plt.plot(history[ref[0]], c='#00d2ff', label=ref[2])
    plt.plot(history[ref[1]], c='#fb2e01', label=ref[3])
    plt.title(ref[0].title() + ' History')
    plt.xlabel('Epoch')
    plt.ylabel('Model ' + ref[0].title())
    plt.legend()
    if save:
        plt.savefig(folder + ref[0] + '_history.png')
    plt.show()

def history_plot(look='accuracy', save=False, folder='pics/'):
    if look == 'accuracy':
        ref = ('accuracy', 'val_accuracy', 'Train Accuracy', 'CV Accuracy')
    elif look == 'loss':
        ref = ('loss', 'val_loss', 'Train Loss', 'CV Loss')
    else:
        print(look + " Data Doesn't Exist")
        return

    if path.isfile("processed_data/history.pyb"):
        history = pickle.load(open("processed_data/history.pyb", "rb"))
    else:
        history = train().history.history
    
    plot_template(history, ref, save=save, folder=folder)
