# Stock-Price-Prediction
Developed Predictive Deep Model to Forecast the  Infosys Stock Closing Price 

### Introduction
- Since its inception in 1981, Infosys has been a key player in the global technology services and consulting sector. As a publicly traded company on the Indian Stock Exchange since 1996, Infosys's stock has witnessed significant growth, influenced by various market factors, industry trends, and global economic shifts.
- Predicting the stock price of such a dynamic entity requires not only an understanding of historical data but also the application of advanced machine learning techniques.
- In this presentation, we delve into the prediction of Infosys's stock prices using sophisticated time series models and machine learning algorithms. We start by exploring the historical data, beginning from 1996, to understand the trends and patterns that have shaped Infosys's stock performance over the years.

### Problem Statement
- In the volatile and rapidly changing stock market, accurately predicting stock prices is a significant challenge that investors, analysts, and financial institutions face daily.
- For a company like Infosys, a global leader in IT services, reliable stock price predictions are crucial for making informed investment decisions, managing risks, and developing strategies for financial growth. 
- Traditional methods of stock price prediction often fall short in capturing the complex, non-linear patterns present in financial time series data.
- Given the importance of accurate predictions in shaping investment strategies and understanding market behavior, this project aims to explore and implement advanced machine          learning models, including ARIMA, Auto ARIMA, XGBoost, Linear Regression, LSTM, and GRU, to enhance the accuracy and reliability of Infosys stock price predictions. 
- By leveraging historical data starting from 1996, the project seeks to provide deeper insights into the factors influencing stock prices and to develop a robust predictive          model that can assist stakeholders in making more informed decisions.

### Dataset

- The dataset used for this analysis was sourced from Yahoo Finance, utilizing the yfinance module to facilitate data extraction.
- It encompasses historical stock price data from January 1, 1996,through August 14, 2024, providing a comprehensive view of nearly three decades of market activity. 
- This extensive dataset includes 7,190 records, offering a rich foundation for time series analysis and predictive modeling.
#### Features: The dataset includes 7 features
- Date: The specific date of the trading data.
- Open: The price of the stock at the beginning of the trading day.
- High: The highest price the stock reached during the trading day.
- Low: The lowest price the stock reached during the trading day.
- Close: Target variable; represents the closing price of the stock, and it's continuous.
- Adj Close: The closing price adjusted for corporate actions like dividends, stock splits, etc.
- Volume: The number of shares traded during the day.

##### Treated missing data and conducted comprehensive exploratory data analysis (EDA), including daily return analysis, stock volume assessment, seasonal decomposition, and month-wise comparison of opening and closing prices.
##### Assessed data stationarity, performed Autocorrelation (ACF) and Partial Autocorrelation (PACF) analysis for feature selection, and applied scaling and re-sampling techniques.
##### Implemented and evaluated a suite of predictive models, including ARIMA, Auto ARIMA, XG-Boost, Linear Regression, Prophet, GRU and LSTM.
##### Successfully deployed the LSTM model with a 98% accuracy in predicting the closing prices, delivered through a web application using Streamlit.

