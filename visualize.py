# Program Description #
# Defines the functions necessary to visualize all the regressive looks #

import numpy as np                      # working with data
import matplotlib.pyplot as plt         # graphing
from sklearn import linear_model        # graph regressive looks
from data_management import format_data # format data

# # garbage to test the program #
# def test_program():
#     # X = np.atleast_2d(np.arange(0, 3.14*2, 0.05)).T
#     X = format_data(np.arange(0, 3.14 * 2, .05))
  
#     # Assign variables to the y axis part of the curve
#     y = np.sin(X)
#     z = np.cos(X)

#     fig, axs = plot_whole(regr_predictions=[y], input=X, output=z, cols=["test1", "test2"])

#     plt.show()


# plot dataset & model #
def plot_whole(regr_predictions, input, output, cols):
    # variable handling #
    num_elements, num_features = np.atleast_2d(input).shape
    pred_type = cols[num_features]

    # initialize plot #
    fig, axs = plt.subplots((num_features + 1) // 2, 2)
    fig.suptitle(f"{pred_type} Prediction")

    # # check subplot size # => doesn't work with 1D + inefficient
    # if num_features % 2 == 1:
    #     fig.delaxes(axs[(num_features + 1) // 2, 1])

    # plot data points #
    plt_num = 0
    for ax in axs.flat:
        # check subplot size #
        if plt_num >= num_features:
            break

        # plot data #
        # ax.scatter(input, output, color='black')
        ax.scatter(np.atleast_2d(input)[ : , plt_num], np.atleast_2d(output)[ : , : ], color='black')
        
        # plot regressive looks #
        plot_regressive_looks(ax, regr_preds=regr_predictions, input=input)

        # labeling #
        ax.set_title(f"{pred_type} correlation w/ {cols[plt_num]}")
        # ax.xlabel(cols[plt_num])
        # ax.ylabel(pred_type)

        # increment #
        plt_num += 1
    
    # return plot #
    return (fig, axs)


# plots all regressive looks #
def plot_regressive_looks(ax, regr_preds, input):
    # variable handling #
    num_elements, num_features = np.atleast_2d(input).shape

    # draw each prediction #
    for regr_pred in regr_preds:
        for feature in range(num_features):
            ax.plot(np.atleast_2d(input)[ : , feature], regr_pred, color='blue', linewidth=2, label='Linear')
        

# # call main #
# if __name__ == "__main__":
#     test_program()