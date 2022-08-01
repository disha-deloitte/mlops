#pip install -U scikit-learn
import numpy
from random import randint
from sklearn.linear_model import LinearRegression

def preprocess_data_model():
    TRAIN_SET_LIMIT = 1000
    TRAIN_SET_COUNT = 100
    numpy.save('TRAIN_SET_LIMIT.npy', TRAIN_SET_LIMIT)
    numpy.save('TRAIN_SET_COUNT.npy', TRAIN_SET_COUNT)
     

if __name__ == "__main__":
    preprocess_data_model()