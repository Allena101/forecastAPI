# forecastAPI_New
A fastAPI application demo that generates forecasts for electricity usage. The forecasting implements the Darts library and takes advantage of exogenous data to enhance the forecasts (e.g. weather, stock market, and google search trends). 

An STL function that imputes missing data is also implemented (which works on the principles of past Seasonal and Trend decomposition), with the intent of preserving individual usage patterns.

Both pSQL and mongoDB databases are implemented and functionality allows for comparisons between forecast values and actuals. These comparisons can be returned as either json or image format (FileResponse)
