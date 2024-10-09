"""
    This file defines the functions necessary to load, modify, convert, and otherwise manage the data 
    used in the algorithm.

    Additional functionality will be implemented in future iterations.
"""

import numpy as np      # cleanse data
import pandas as pd     # load dataframe


# full data process #
def data_import(dir):
    # load datasets #
    data, cols = read_data(dir)

    # disregard time #
    cols.remove("time")
    col_indxs = [ cols.index(col) for col in cols ]

    data = data[ : , col_indxs ]

    # split data #
    train_data, train_output = split_io(data)

    # normalize input #
    train_data = normalize(train_data)

    # return datasets #
    return [ cols, train_data, train_output ]


# full data process #
def feature_import(dir, feature_name):
    # load datasets #
    data, cols = read_data(dir)

    # focus data #
    time_indx = cols.index("time")
    feature_indx = cols.index(feature_name)
    data = data[ : , [time_indx, feature_indx] ]

    # sort data #
    data = np.flip( data[ data[:, 0].argsort() ], axis=0)

    # split data #
    time_data, feature_data = split_io(data)

    # normalize input #
    time_data = normalize(time_data)

    # find current info #
    cur_date = time_data[0][0]
    cur_val = feature_data[0]

    # return datasets #
    return [ ["time", feature_name], time_data, feature_data, (cur_date, cur_val) ]


# exports data #
def data_export(dataset, col_labels, output_dir):
    # dataframe creation #
    df = pd.DataFrame(dataset)

    # deploy data #
    df.to_csv(output_dir, columns=col_labels)


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
