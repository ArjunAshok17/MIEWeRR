# Program Description #
# Main program from which sub-functions are called to run the final algorithm #

# import files #
from data_management import *       # import, track, and clean data
from data_scrap import *            # real-time stock data
from regression import *            # run regression w/ gradient descent
from performance_sim import *       # simulate the performance of the final model
from visualize import *             # plotting regressive looks

# conduct algorithm #
def main(void):
    # user input #
    