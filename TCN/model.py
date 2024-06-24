import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence, text
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed, Bidirectional
from sklearn.model_selection import train_test_split

def seq2ngrams(seqs, n=3, maxlen=None, padding_value=''):
    ngrams_list = []
    if maxlen is None:
        maxlen = max(len(seq) for seq in seqs)
    for seq in seqs:
        padded_seq = seq + padding_value * (maxlen - len(seq))
        seq_ngrams = [padded_seq[i:i+n] for i in range(len(padded_seq) - n + 1)]
        ngrams_list.append(seq_ngrams)
    return np.array(ngrams_list, dtype="object")

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

class MBPredictor:
    def __init__(self, data_paths, params, network_name, loggername):
        self.seq_path = data_paths[0]
        self.struct_path = data_paths[1]
        self.params = params
        self.fc_size = 2 * params['lstm_cell_size']
        self.network_name = network_name
        self.loggername = loggername

    def train(self):
        train_data, train_labels = self.load_data(self.seq_path, self.struct_path)
        model = self.build_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_data, train_labels, batch_size=self.params['batch_size'], epochs=self.params['num_epochs'])
        model.save(self.network_name + '.h5')

    def test(self):
        test_data, test_labels = self.load_data(self.seq_path, self.struct_path)
        model = tf.keras.models.load_model(self.network_name + '.h5')
        loss, accuracy = model.evaluate(test_data, test_labels)
        print("Test Loss:", loss)
        print("Test Accuracy:", accuracy)

    def build_model(self):
        model_input = Input(shape=(self.params['max_seq_len'], self.params['structures'] + 4))
        bi_lstm_layer = Bidirectional(LSTM(units=self.params['lstm_cell_size'], return_sequences=True))(model_input)
        hidden_layer = Dense(self.fc_size, activation='relu')(bi_lstm_layer)
        preds = Dense(self.params['structures'], activation='softmax')(hidden_layer)
        model = Model(inputs=model_input, outputs=preds)
        return model

def main():
    seq_path = 'path/to/sequences.csv'
    struct_path = 'path/to/structures.csv'

    params = {
        'max_seq_len': 250,
        'structures': 5,
        'batch_size': 32,
        'num_epochs': 100,
        'lstm_cell_size': 64
    }

    data_paths = [
        'data/data10.h5',
        'data/data11.h5'
    ]

    loggername = "results.txt"

    network_name = "protein_lstm_network"

    predictor = MBPredictor(data_paths, params, network_name, loggername)

    df = load_data(data_paths[0])

    maxlen_seq = params['max_seq_len']
    input_seqs, target_seqs = df[['seq', 'sst3']][(df.len <= maxlen_seq) & (~df.has_nonstd_aa)].values.T
    input_grams = seq2ngrams(input_seqs)

    tokenizer_encoder = text.Tokenizer()
    tokenizer_encoder.fit_on_texts(input_grams)
    input_data = tokenizer_encoder.texts_to_sequences(input_grams)
    input_data = sequence.pad_sequences(input_data, maxlen=maxlen_seq, padding='post')

    tokenizer_decoder = text.Tokenizer(char_level=True)
    tokenizer_decoder.fit_on_texts(target_seqs)
    target_data = tokenizer_decoder.texts_to_sequences(target_seqs)
    target_data = sequence.pad_sequences(target_data, maxlen=maxlen_seq, padding='post')
    target_data = to_categorical(target_data)

    X_train, X_test, y_train, y_test = train_test_split(input_data, target_data, test_size=.2, random_state=42)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    predictor.train()

if __name__ == "__main__":
    main()
