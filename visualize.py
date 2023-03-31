"""
    This file defines the functions necessary to easily visualize multiply (self-referential) regressive 
    looks at the same dataset (for individual features), and plotting a predictive line for forecasting 
    all features in the future combined with their associative power (with respect to the final predictive 
    quantity).
"""

import numpy as np                      # working with data
import matplotlib.pyplot as plt         # graphing
from sklearn import linear_model        # graph regressive looks
from data_management import format_data # format data


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
        # plot_regressive_looks(ax, regr_preds=regr_predictions, input=input)

        # labeling #
        ax.set_title(f"{pred_type} correlation w/ {cols[plt_num]}")
        # ax.xlabel(cols[plt_num])
        # ax.ylabel(pred_type)

        # increment #
        plt_num += 1
    
    # return plot #
    return (fig, axs)


# plots all regressive looks #
def plot_forecasts(data, exp_out, future_data, pred_data):
    # plot data #
    plt.scatter(data, exp_out, color="black")

    # plot forecasts #
    plt.plot(future_data, pred_data, color="forestgreen", label="Multi-Regressive Forecast")

    # labels #
    plt.legend()
    plt.xlabel("features")
    plt.ylabel("model output")

    # show plot #
    plt.show()


# plots regressive looks for one feature #
def plot_feature_looks(regr_preds, self_pred, input, output, pred_data, time_frames, col_labels):
    # plot data #
    plt.plot(input, output, color="black")
    num_frames = len(time_frames)

    # draw each regressive look #
    color = iter(plt.cm.rainbow(np.linspace(0, 1, num_frames + 1)))
    for f in range(num_frames):
        c = next(color)
        plt.plot(input[ : time_frames[f] ], regr_preds[f], color=c, label=f"Regressive Look {f}")

    # draw self predictive look #
    c = next(color)
    plt.plot(pred_data, self_pred, color=c, label=f"Self-Predictive Look")

    # labels #
    plt.legend()
    plt.xlabel(col_labels[0])
    plt.ylabel(col_labels[1])

    # show plot #
    plt.show()


# # garbage to test the program #
# def test_program():
#     # X = np.atleast_2d(np.arange(0, 3.14*2, 0.05)).T
#     X = format_data(np.arange(0, 3.14 * 2, .05))
  
#     # Assign variables to the y axis part of the curve
#     y = np.sin(X)
#     z = np.cos(X)

#     fig, axs = plot_whole(regr_predictions=[y], input=X, output=z, cols=["test1", "test2"])

#     plt.show()


# run as script #
# if __name__ == "__main__":
#     test_program()