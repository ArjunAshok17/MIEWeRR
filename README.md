# Multi-Regression-Analysis #

### Quick Look ###
This algorithm takes multiple regressive looks at the same set of data [with respect to different timeframes] to generate the most accurate linear outlook for a given time period. In other words, a regressive analog to LSTM (long-short-term-memory) models.

The algorithm will be implemented first in Python and later in C++, though the C++ version may not contain files for data manipulation.

## Problem Addressed ##
Often times in the real-world it benefits data analysts to consider long-term trends as much as short-term trends. In the case of stocks, for example, finding a balance in how much to weigh the long-term growth potential of a given stock versus its more recent performance can be challenging for humans to do quickly and with high accuracy.

Moreover, the balance that might be reached for a certain time period of forecasting may differ from the balance that is optimal for another time period. In other words, an analyst trying to predict a stock's performance (or at least as best as a linear model could manage) for the next three months, may choose to focus on or *weigh* more heavily the recent performance of the stock. In contrast, an analyst looking to invest long-term in a stock may choose to weigh long-term trends more heavily.

# Algorithm #

## Proposed Solution ##
It's with this context in mind that it becomes clear the need to conduct multiple regressive outlooks on the same piece of data, each successive model looking a smaller and smaller chunk of data.

For example, consider a company like Apple (AAPL). With data on Apple's stock performance over decades, we could train a linear model on its long term performance from its introduction into the stock market until today. A second "regressive look" (as I'm calling it) might take a look at a smaller chunk of data, say the performance in the last 10 years. Then another model for the last 5 years, 2 years, year, 6 months, 3 months, 1 month, etc.

Time frames will vary of course, and any users of the algorithm will have to implement their time frames manually.

The end product should be a line starting at the current price of the stock at today's date and projecting outward in a direction that weights all the slopes of previous lines.

## Design ##
The equation of a line is $y = mx + b$. Of course with more features, this equation looks more like $y = w_1*x_1 + w_2*x_2 + . . . + w_n*x_n + c_0$ where there are $n$ features, with $w_n$ being the weight of each feature $x_n$, and $c_0$ being the y-intercept. For every timeframe, $c_0$ will always be the stock price at the beginning of the time frame. These weights will be calculated using normal linear regression.

Once these weights are calculated for all the timeframes being considered, we can start the unique weighting process. Given the timeframe consideration, we lack data to do this weighting regressively. Instead, the algorithms opts for a more statistical approach that relies on the timeframe that needs to be predicted.

What we can assume is that behavior in the hyper-short-term is a lot less useful that data in the hyper-long-term. This means we can start distributing the weight with a right skew towards longer timeframes. But where to center the distribution? In this regard, the algorithm is not entirely based in mathematical truths. For the center, we (somewhat arbitrarily, but with reasonable assumption) pick the timeframe that most closely resembles the timeframe we try to predict.

For example, if a price at the end of 3-months wants to be predicted, the algorithm would weigh the performance calculated by the regressive look at the *past* 3 months the highest. The second highest would be the past 6 months. Then past month, past year, and so on.

## Important Considerations ##
With pricing information varying wildly across years, it's important to not only normalize stock data for performance, but also change numbers to reflect current inflation rates.

This repository will include the necessary tools for normalizing and working with the data. In the future, support may be added for automatically scraping stock (or other) data or potentially incorporating an API to avoid the task of scraping.
