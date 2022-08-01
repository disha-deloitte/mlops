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
    
    print('Outcome: {} \n Coefficients: {}'.format(outcome, coefficients))
   
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictor')
    args = parser.parse_args()
    predictor = LinearRegression(n_jobs=-1) #Create a linear regression object NOTE n_jobs = the number of jobs to use for computation, -1 means use all processors
    predictor.fit(X=[[20],[20]], y=[[20],[20]])  #fit the linear model (approximate a target function)
    args = parser.parse_args()
    joblib.dump(predictor, 'predictor.pkl')
    testing_model( args.predictor)
    #testing_model( args.predictor)