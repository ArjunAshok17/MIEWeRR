# Program Description #
# Defines the functions necesary to load, modify, convert, and otherwise manage datasets #

import numpy as np      # cleanse data
import pandas as pd     # load dataframe


# full data process #
def data_import(dir, cols):
    # load datasets #
    data = read_data(dir, cols)
    train_data, test_data, cv_data = split_data(data)

    # split data #
    train_input, train_exp_output = split_io(train_data)
    test_input, test_exp_output = split_io(test_data)
    cv_input, cv_exp_output = split_io(cv_data)

    # normalize input #
    train_input = normalize(train_input)
    test_input = normalize(test_input)
    cv_input = normalize(cv_input)

    # return datasets #
    return [train_input, train_exp_output, test_input, test_exp_output, cv_input, cv_exp_output]


# reads dataset #
def read_data(dir, cols):
    # read into panda dataframe #
    data = pd.read_csv(dir, ',', usecols=cols)

    # convert to numpy array #
    data_arr = np.array(data.values, 'float')

    # give back info #
    return data_arr


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
def normalize(data):
    # normalize inputs #
    input = input / input.max(axis=0)

    # return split data #
    return input


# split data into input & output #
def split_io(data):
    # dimensions #
    num_entries, num_cols = data.shape
    num_cols -= 1

    # split data #
    input = np.atleast_2d(data)[ : , : num_cols]
    expected_output = np.atleast_2d(data)[ :, num_cols]

    # return split #
    return input, expected_output


# splits into training, test, and cross-validation sets #
def split_data(data):
    # dimensions #
    num_cols, num_entries = np.atleast_2d(data).shape

    # percent split #
    train_entries = int(num_entries * .7)
    test_entries = int(num_entries * .15)
    cv_entries = num_entries - (train_entries + test_entries)

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
