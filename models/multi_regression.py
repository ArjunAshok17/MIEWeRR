"""
    This algorithm combines all the sub-functions defined elsewhere, allowing us to generate a self-
    sufficient predictive model that predicts how each of the features considered trend in the future, 
    and then uses that to predict how the final output evolves in the future.

    Note: the final model doesn't rely on time at all, opting to defer any forecasting to the feature 
          models and only taking responsibility for the final associations with the data. This allows 
          us to produce an optimal relationship between all the features while maintaing a predictive 
          ability with respect to time.
"""

# import files #
        # from data_scrape import *           # real-time stock data
        # from performance_sim import *       # simulate the performance of the final model
from data_management import *       # import, track, and clean data
from regression import *            # run regression w/ gradient descent
from visualize import *             # plotting regressive looks
from self_contained_regr import *   # for each feature's self-referential regressive model


"""
    ======= Set Params for Testing ========
    dir             = ""        | string for the directory to source data from
    feature_set     = [""]      | features to predict and consider; num_features inferred from here; time assumed always a feature
    weight_dists    = [[#.#]]   | weight distributions for each feature in feature_set
    time_ratios     = [[#.#]]   | time frame ratios for each weight distribution in weight_dists    
    pred_yrs        = #.#       | number of years to predict to (will output predictions for every day till then)
"""
global dir
dir = "./NYSE_sample_data/prices_adjusted.csv"

global feature_set
feature_set = ["price"]
global final_feature
final_feature = "price"

global weight_dists
weight_dists = [[25, 30, 25, 20, 15, 5, 1]]
global time_ratios
time_ratios = [[1, .75, .5, .25, .1, .05, .01]]

global pred_yrs
pred_yrs = 3


# conduct algorithm #
def multi_regression():
    # check params #
    check_params()

    # declare vars #
    self_contained_models = []      # stores final models for each feature
    self_contained_forecasts = []   # stored final forecasts (for the given predictive range) for each feature

    # import #
    dataset = data_import(dir)
    col_labels = dataset[0]

    data = dataset[1]
    output = format_data(dataset[2])
    
    # build associations for each feature #
    multi_regr_model = optimize(input=data, exp_out=output, cur_val=-1, fix_intercept=False)
    multi_regr = multi_regr_model[0]

    # train self-referencing feature predictions #
    for feat_num in range(len(feature_set)):
        self_contained_out = self_contained_regression(dir,
                                                       feature_name=feature_set[feat_num],
                                                       weights=weight_dists[feat_num],
                                                       time_ratio=time_ratios[feat_num],
                                                       pred_yrs=pred_yrs
                                                      )
        self_contained_models.append(self_contained_out[0])
        self_contained_forecasts.append(self_contained_out[1])
    
    # predictions #
    forecasted_feature_data = produce_forecasted_data(forecasts=self_contained_forecasts)
    forecast_predictions = multi_regr.predict(forecasted_feature_data)

    # visualize #
    plot_forecasts(data=data, exp_out=output, future_data=forecasted_feature_data, pred_data=forecast_predictions)


# produces the final predictive dataset for all features #
def produce_forecasted_data(forecasts):
    # combine all forecasts #
    forecasted_data = np.stack(forecasts, axis=1)
    return forecasted_data

# checks params are correctly inputted #
def check_params():
    num_features = len(feature_set)

    if num_features <= 0:
        print("ERROR: positive integer number of features required")
        quit()
    
    if len(weight_dists) != num_features:
        print("ERROR: check matching length for weight_dists and feature_set")
        quit()

    if len(time_ratios) != num_features:
        print("ERROR: check matching length for time_ratios and feature_set")
        quit()

    if len(time_ratios[0]) != len(weight_dists[0]):
        print("ERROR: check matching length for weight_dists[0] and time_ratios[0]")
        quit()


# run as script #
if __name__ == "__main__":
    multi_regression()