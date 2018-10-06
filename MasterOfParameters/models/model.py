import pandas as pd
from MasterOfParameters.utility.utils import get_results


class model:
    def __init__(self, myDataLoader, predpath):
        self.n_words = len(myDataLoader.tokenizer_encoder.word_index) + 1
        self.n_tags = len(myDataLoader.tokenizer_decoder.word_index) + 1
        self.maxlen_seq = myDataLoader.maxlen_seq
        self.predpath = predpath

        self.test_result_template = myDataLoader.test_df.copy()
        self.X_train = myDataLoader.X_train
        self.X_val = myDataLoader.X_val
        self.y_train = myDataLoader.y_train
        self.y_val = myDataLoader.y_val
        self.x_test = myDataLoader.test_input_data
        self.x_test_seqs = myDataLoader.test_input_seqs
        self.revsere_decoder_index = {value: key for key, value in myDataLoader.tokenizer_decoder.word_index.items()}

        self.model = None
        self.build_model()

    def build_model(self):
        raise NotImplementedError

    def train(self):
        # Training the model on the training data and validating using the validation set
        self.model.fit(self.X_train, self.y_train, batch_size=128, epochs=5,
                       validation_data=(self.X_val, self.y_val), verbose=1)

    def predict(self):
        pred = self.model.predict(self.x_test[:])
        self.save_prediction(pred)

    def save_prediction(self, prediction):
        y_test_string = []
        for i in range(len(self.x_test)):
            y_test_string.append(get_results(self.x_test_seqs[i], prediction[i], self.revsere_decoder_index))
        test_df = self.test_result_template
        test_df['expected'] = pd.Series(y_test_string)
        del test_df['len']
        del test_df['input']
        print("result path:", self.predpath)
        test_df.to_csv(self.predpath, sep=',', encoding='utf-8', index=False)
