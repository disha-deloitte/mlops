#pip install -U scikit-learn

import argparse
import joblib
from random import randint
from sklearn.linear_model import LinearRegression


def training_model(TRAIN_SET_COUNT,TRAIN_SET_LIMIT):
    #Create and append a randomly generated data set to the input and output
    for i in range(TRAIN_SET_COUNT):
        a = randint(0, TRAIN_SET_LIMIT)
        b = randint(0, TRAIN_SET_LIMIT)
        c = randint(0, TRAIN_SET_LIMIT)
    #Create a linear function for the output dataset 'Y'
    op = (10*a) + (2*b) + (3*c)
    TRAIN_INPUT.append([a,b,c])
    TRAIN_OUTPUT.append(op)
    predictor = LinearRegression(n_jobs=-1) #Create a linear regression object NOTE n_jobs = the number of jobs to use for computation, -1 means use all processors
    predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)  #fit the linear model (approximate a target function)
    
    joblib.dump(predictor, 'predictor.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--TRAIN_SET_COUNT')
    parser.add_argument('--TRAIN_SET_LIMIT')
    args = parser.parse_args()
    training_model(args.TRAIN_SET_COUNT, args.TRAIN_SET_LIMIT)

