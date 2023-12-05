# data_loader.py
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

class DataLoader:
    @staticmethod
    def load_data():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
        return x_train, y_train, x_val, y_val, x_test, y_test
