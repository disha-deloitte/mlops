#pip install -U scikit-learn

from random import randint
from sklearn.linear_model import LinearRegression

def preprocess_data_model():
    # Create a range limit for random numbers in the training set, and a count of the number of rows in the training set
    TRAIN_SET_LIMIT = 1000
    TRAIN_SET_COUNT = 100
    # Create an empty list of the input training set 'X' and create an empty list of the output for each training set 'Y'
    TRAIN_INPUT = list()
    TRAIN_OUTPUT= list()
    np.save('TRAIN_SET_LIMIT.npy', TRAIN_SET_LIMIT)
    np.save('TRAIN_SET_COUNT.npy', TRAIN_SET_COUNT)
     

if __name__ == "__main__":
    preprocess_data_model()