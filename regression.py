"""
    This file defines the functions necessary to run linear regression on a dataset. The customized functions 
    are currently implemented using pure statistics (line of best fit using LMSR) in sklearn, but will later be 
    reimplemented using gradient descent.
"""
# Defines the functions necessary to run linear regression on a dataset #

from sklearn import linear_model    # linear regression
from data_management import *       # data management


# returns optimized linear model #
def optimize(input, exp_out, cur_val, fix_intercept):
    # model creation #
    regr_look = linear_model.LinearRegression(fit_intercept=(not fix_intercept))
    if fix_intercept:
        exp_out = exp_out - cur_val
    
    # train #
    regr_look.fit(input, exp_out)
    # regr_look.intercept_ = cur_val  # warning, does not change intercept val

    # return model & parameters #
    return [ regr_look, regr_look.coef_[0], cur_val if fix_intercept else regr_look.intercept_[0] ]


"""
    The following code is simply to experiment with manually coding gradient descent.
    Coming soon.
"""
