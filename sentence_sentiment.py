import pickle
from process_data import clean_text as ct
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def process_text(text, file="processed_data/tokenizer_half.pyb"):
    text = ct(text)
    tokenizer = pickle.load(open(file, "rb"))
    formatted_text = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=40)
    return formatted_text

def get_sentiment(model_f="processed_data/sentiment_model.h5", tokenizer_f="processed_data/tokenizer_half.pyb"):
    key = ['Negative', 'Positive']
    model = load_model(model_f)
    print("Model Loaded")
    text = input("\nEnter a sentence:\n")
    text = process_text(text, file=tokenizer_f)
    pred = model.predict(text).item()
    int_pred = int(round(pred))
    conf = pred if int_pred else (1.0 - pred)
    print("\nSentiment of Text: {}, Confidence: {}".format(key[int_pred].upper(), conf))

if __name__ == "__main__":
    get_sentiment()