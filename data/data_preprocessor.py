# data_preprocessor.py
class DataPreprocessor:
    @staticmethod
    def preprocess_data(x_train, x_val, x_test):
        x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
        x_val = x_val.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
        x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
        return x_train, x_val, x_test
