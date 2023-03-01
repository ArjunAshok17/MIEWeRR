# Program Description #
# Defines the functions necessary to run linear regression on a dataset #

from sklearn import linear_model    # linear regrssion
from data_management import *        # data management


# returns optimized linear model #
def optimize(input, exp_out):
    # model creation #
    regr_look = linear_model.LinearRegression()
    
    # train #
    regr_look.fit(input, exp_out)

    # return model & parameters #
    return (regr_look, regr_look.coef_)


"""
    The following code is simply to experiment with manually coding gradient descent.
    Coming soon.
"""
