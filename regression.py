# Program Description #
# Defines the functions necessary to run linear regression on a dataset #

from sklearn import linear_model    # linear regression
from data_management import *       # data management


# returns optimized linear model #
def optimize(input, exp_out, cur_val):
    # model creation #
    regr_look = linear_model.LinearRegression(fit_intercept=False)
    exp_out = exp_out - cur_val
    
    # train #
    regr_look.fit(input, exp_out)
    # regr_look.intercept_ = cur_val  # warning, does not change intercept val

    # return model & parameters #
    return [ regr_look, regr_look.coef_[0], cur_val ]


"""
    The following code is simply to experiment with manually coding gradient descent.
    Coming soon.
"""
