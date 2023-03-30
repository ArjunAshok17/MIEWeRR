"""
    This algorithm conducts the several regressive looks on only one feature with respect to time, 
    allowing us to predict with the highest accuracy the future outlook for one feature. At least, as well 
    as a linear model allows us.
"""

# import files #
from data_management import *       # import, track, and clean data
# from data_scrape import *           # real-time stock data
from regression import *            # run regression w/ gradient descent
# from performance_sim import *       # simulate the performance of the final model
from visualize import *             # plotting regressive looks
from math import floor              # time split


# set params #
global weight_distribution
weight_distribution = [15, 25*100000000, 30, 25, 15, 5, 2.5]
global time_frame_ratio
time_frame_ratio = [1, .75, .5, .25, .1, .05, .01]

global cur_val
cur_val = -1.0
global cur_date
cur_date = -1
global pred_range
pred_range = 3*365      # measured in days, number of days to predict to


# conduct algorithm #
def main():
    # declare vars #
    regr_looks = []

    # import #
    feature_data = feature_import("./NYSE_sample_data/prices_adjusted.csv", "price")

    col_labels = feature_data[0]
    data = feature_data[1]
    feature_output = format_data(feature_data[2])
    
    global cur_date
    cur_date = feature_data[3][0]
    global cur_val
    cur_val = feature_data[3][1]

    # time frames #
    time_frames = split_time_frame(time_data=data, frame_ratio=time_frame_ratio)
    
    # train models #
    regr_looks = train_regr_looks(time_frames=time_frames, input=data, output=feature_output)

    # multi-regressive model #
    self_regr = regr_weighted(regr_looks=regr_looks, weight_distribution=weight_distribution)
    
    # predictions #
    regr_preds = regr_prediction(regr_looks=regr_looks, input=data, time_frames=time_frames)
    print(cur_date, cur_val)

    self_pred_data = np.arange(cur_date, cur_date + pred_range).reshape(-1, 1)
    self_pred = self_regr.predict(self_pred_data)

    # visualize #
    plot_feature_looks(regr_preds=regr_preds, self_pred=self_pred, input=data, output=feature_output, pred_data=self_pred_data,\
                       time_frames=time_frames, col_labels=col_labels)


# weighted distribution of regressive looks #
def regr_weighted(regr_looks, weight_distribution):
    # apply weighting #
    self_coef = []
    sum_weights = np.sum( np.array(weight_distribution) )

    for look_num in range(len(regr_looks)):
        self_coef.append( weight_distribution[look_num] * regr_looks[look_num][0] )
    
    # averaging #
    self_coef = np.array([ np.asarray(self_coef).mean(axis=0) ]) / sum_weights

    # return model #
    self_regr = linear_model.LinearRegression()
    self_regr.coef_ = self_coef
    self_regr.intercept_ = cur_val

    return self_regr


# conducts predictions for all models #
def regr_prediction(regr_looks, input, time_frames):
    # make prediction #
    regr_preds = []

    for look_num in range(len(time_frames)):
        regr_look_info = regr_looks[look_num]
        regr_look = linear_model.LinearRegression()

        regr_look.coef_ = np.array([regr_look_info[0]])
        regr_look.intercept_ = np.array([regr_look_info[1]])

        regr_preds.append(regr_look.predict(input[ : time_frames[look_num]]))

    # return regressive predictions #
    return regr_preds


# train regressive looks #
def train_regr_looks(time_frames, input, output):
    """
        add trained model for each regressive look:
              regr_look[i]      = ith regressive output
              regr_look[i][0]   = ith regressive output's coefficient coefficient
              regr_look[i][1]   = ith regressive output's intercept
    """
    regr_looks = []
    
    for frame in time_frames:
        model_info = optimize( input[ : frame], output[ : frame], cur_val=output[frame - 1] )
        regr_looks.append([ model_info[1][0], model_info[2][0] ])
    
    return np.array(regr_looks)


# create custom weighting based on predictive range #
def distribute_weights(pred_range, skew, num_timeframes):
    """
        pred_range: number of time units to predict to
        skew:       -1 is left (longer term) skew,
                    0 is normal curve,
                    1 is right (shorter term skew)
    """
    # to be implemented #
    return weight_distribution


# divide data into time frames #
def split_time_frame(time_data, frame_ratio):
    # range #
        # begin = np.min(time_data)
        # end = np.max(time_data)
        # range = end - begin
    range = len(time_data)

    # divide #
    return [ floor(ratio * range) for ratio in frame_ratio ]

if __name__ == "__main__":
    main()