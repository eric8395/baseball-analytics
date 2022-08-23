# import libraries 
import pandas as pd
import numpy as np
import math
from scipy.stats import kurtosis, skew


from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, \
RandomizedSearchCV, StratifiedKFold, KFold

# create function that gives performance metrics
def performance(y_true, y_predict):
    """ 
    Calculates and returns the two performance scores between 
    true and predicted values - first R-Squared, then RMSE
    """

    # Calculate the r2 score between 'y_true' and 'y_predict'
    r2 = r2_score(y_true, y_predict)

    # Calculate the root mean squared error between 'y_true' and 'y_predict'
    rmse = mean_squared_error(np.exp(y_true), np.exp(y_predict), squared = False) # False gives RMSE

    # Return the score
    return [r2, rmse]

# define a simple function that returns cross validation score for a 5 fold
def get_cv_score(model, X, y):
    
    # R2 score CV
    cv_r2_score = np.mean(cross_val_score(model, 
                                       X, y, 
                                       scoring = 'r2', 
                                       cv = 5))
    # RMSE score CV
    cv_rmse_score = np.mean(cross_val_score(model,
                                           X,
                                           np.exp(y),
                                           scoring = 'neg_root_mean_squared_error',
                                           cv = 5))
    
    print(model,"Cross Validation R2:      ",round(cv_r2_score, 4))
    print(model,"Cross Validation RMSE:   ",-round(cv_rmse_score))

def model_results(model, X_train, y_train, X_test, y_test):
    """
    Helper function that takes input of model, and train-test split sets 
    and returns the model R2, RMSE scores
    """
    # fit the model
    model.fit(X_train, y_train)
    
    # Make predictions on the training and test data
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate performance using the performance() function 
    train_scores = performance(y_train, y_pred_train)
    test_scores = performance(y_test, y_pred_test)

    # Training
    print(" "*12, model, "RESULTS")
    print(model, 'Training R2:              ', round(train_scores[0],4)) # R2
    print(model, 'Test R2:                  ', round(test_scores[0],4)) # R2
    print('-----' * 11)
    # Validation
    get_cv_score(model, X_train, y_train)
    print('-----' * 11)
    # Testing
    print(model, 'Training RMSE:           ', round(train_scores[1])) # RMSE
    print(model, 'Test RMSE:               ', round(test_scores[1])) # RMSE