#pip install -U scikit-learn

import argparse
from random import randint
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

def testing_model(predictor):
    X_TEST = [[10,20,30]] #Create our testing data set, the ouput should be 10*10 + 2*20 + 3*30 = 230
    model = joblib.load(predictor)
    outcome = model.predict(X=X_TEST) # Predict the ouput of the test data using the linear model
    
    coefficients = model.coef_  #The estimated coefficients for the linear regression problem.
    with open('output.txt', 'a') as f:
        f.write(str(outcome))
    print('Outcome: {} \n Coefficients: {}'.format(outcome, coefficients))
    
   
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictor')
    args = parser.parse_args()
    testing_model( args.predictor)