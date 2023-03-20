# Program Description #
# Defines the functions necesary to load, modify, convert, and otherwise manage datasets #

import numpy as np      # cleanse data
import pandas as pd     # load dataframe


# full data process #
def data_import(dir):
    # load datasets #
    data, cols = read_data(dir)

    # find current price #
    indx = cols.index("price")
    current_price = np.max( np.atleast_2d(data)[ : , indx ] , axis=0)

    # split data #
    train_data, test_data, cv_data = split_data(data)
    train_input, train_exp_output = split_io(train_data)
    test_input, test_exp_output = split_io(test_data)
    cv_input, cv_exp_output = split_io(cv_data)

    # normalize input #
    train_input = normalize(train_input)
    test_input = normalize(test_input)
    cv_input = normalize(cv_input)

    # return datasets #
    return [cols, train_input, train_exp_output, test_input, test_exp_output, cv_input, cv_exp_output, current_price]


# reads dataset #
def read_data(dir):
    # read into panda dataframe #
    data = pd.read_csv(dir, delimiter=',')

    # convert to numpy array #
    data_arr = np.array(data.values, "float")
    data_cols = data.columns.to_list()

    # give back info #
    return data_arr, data_cols


# ensure data format #
def format_data(data):
    # # check dimensions #
    # if np.atleast_2d(data).shape[0] != 1:
    #     return data
    
    # check transposing #
    if np.atleast_2d(data).shape[0] < np.atleast_2d(data).shape[1]:
        data = np.atleast_2d(data).T

    # default #
    return data


# normalize data #
def normalize(input):
    # normalize inputs #
    # input = (input - input.min(axis=0)) * 100 / (input.max(axis=0) - input.min(axis=0))
    input = (input - input.min(axis=0))

    # return split data #
    return input


# split data into input & output #
def split_io(data):
    # dimensions #
    num_elements, num_features = data.shape
    num_features -= 1

    # split data #
    input = np.atleast_2d(data)[ : , : num_features]
    expected_output = np.atleast_2d(data)[ :, num_features]

    # return split #
    return input, expected_output


# splits into training, test, and cross-validation sets #
def split_data(data):
    # dimensions #
    num_elements, num_features = np.atleast_2d(data).shape

    # percent split #
    train_entries = int(num_elements * .7)
    test_entries = int(num_elements * .15)
    cv_entries = num_elements - (train_entries + test_entries)

    test_start = train_entries + 1
    cv_start = train_entries + test_entries + 1

    # randomize #
    np.random.shuffle(data)

    # split #
    train_data = np.atleast_2d(data)[ : test_start, : ]
    test_data = np.atleast_2d(data)[test_start : cv_start, : ]
    cv_data = np.atleast_2d(data)[cv_start : , : ]

    # return #
    return train_data, test_data, cv_data
