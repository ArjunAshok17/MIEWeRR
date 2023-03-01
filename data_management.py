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


# normalize data #
def normalize(data):
    # split data #
    input, expected_output = split_io(data)

    # normalize inputs #
    input = input / input.max(axis=0)

    # return split data #
    return input, expected_output


# split data into input & output #
def split_io(data):
    # dimensions #
    num_entries, num_cols = data.shape
    num_cols -= 1

    # split data #
    input = data[ : , : num_cols]
    expected_output = data[ :, num_cols]

    # return split #
    return input, expected_output


# splits into training, test, and cross-validation sets #
def split_data(data):
    # dimensions #
    num_entries, num_cols = data.shape

    # percent split #
    train_entries = int(num_entries * .7)
    test_entries = int(num_entries * .15)
    cv_entries = num_entries - (train_entries + test_entries)

    test_start = train_entries + 1
    cv_start = train_entries + test_entries + 1

    # randomize #
    np.random.shuffle(data)

    # split #
    train_data = data[ : test_start, : ]
    test_data = data[test_start : cv_start, : ]
    cv_data = data[cv_start : , : ]

    # return #
    return train_data, test_data, cv_data
