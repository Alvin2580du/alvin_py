# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn import metrics
#
# all_data = pd.read_csv("CBA_1991-2018.csv")
# #print(all_data)
#
# #Extract High stock prices
# high_stock_prices = all_data['High']
# #print(high_stock_prices)
# #print(high_stock_prices.name)
#
# #Convert Datetime
# data_time = all_data['Date']
# data_time = pd.to_datetime(data_time)
# #print(data_time)
#
#
# #Make it as a time series with datetime as index
# time_series = pd.Series(data = np.array(high_stock_prices), index = data_time)
# #print(time_series)
#
# #Save as CBA_1991 -2018High.csv
# time_series.to_csv("G:\individual assignment\individual assignment\CBA_1991 -2018High.csv", header = 1, index = 1)
#
# #Transform the time series data by the First Order Difference
# time_series_1st = time_series.diff()
# #Transform the time series data by the Second Order Difference
# time_series_2nd = time_series.diff(periods=2)
#
# #2.a
# #Plots
# plt.plot(time_series)
# plt.title('CBA_1991-2018High Time Series')
# plt.xlabel('Year')
# plt.ylabel('Price')
# plt.legend()
# plt.show()
#
# plt.plot(time_series_1st)
# plt.title('CBA_1991-2018High the First Order Difference')
# plt.xlabel('Year')
# plt.ylabel('Price')
# plt.legend()
# plt.show()
#
# plt.plot(time_series_2nd)
# plt.title('CBA_1991-2018High the Second Order Difference')
# plt.xlabel('Year')
# plt.ylabel('Price')
# plt.legend()
# plt.show()
#
# #2.b
# #CMA-24，k=12
# def cma_smoothing(time_series, k):
#      series = []
#      temp = []
#      for index in np.arange(0, time_series.__len__()):
#          if index - k < 0:
#              temp = None
#          elif index + k > time_series.__len__() - 1:
#              temp = None
#          else:
#              temp = (time_series[index - k] + time_series[index + k]) / (4 * k) \
#                  + (time_series[index - k + 1 : index + k].sum()) / (2 * k)
#          series.append(temp)
#      return pd.Series(data = series, index = time_series.index)
#
# #CMA-24 Smoothing
# time_series_smoothing_cma = cma_smoothing(time_series, 12)
# #pandas rolling
# #Rolling Windows
# time_series_smoothing_rolling = time_series.rolling(window = 12, center = True).mean().rolling(window = 2, center = True).mean()
# plt.figure()
# plt.plot(time_series_smoothing_cma, color = "red", label = "CMA-24 smoothing")
# plt.plot(time_series_smoothing_rolling, color = "blue", label = "Pandas smoothing")
# plt.title('Different method for smoothing time series')
# plt.xlabel('Year')
# plt.ylabel('Price')
# plt.legend()
# plt.show()
#
# plt.figure()
# plt.plot(time_series_smoothing_cma, color = "red", label = "CMA-24 smoothing")
# plt.xlabel('Year')
# plt.ylabel('Price')
# plt.legend()
# plt.show()
#
# plt.figure()
# plt.plot(time_series_smoothing_rolling, color = "blue", label = "Pandas smoothing")
# plt.xlabel('Year')
# plt.ylabel('Price')
# plt.legend()
# plt.show()
# #Calculating MAE
# def mae(series_1, series_2):
#     absError = []
#     for index in range(len(series_1)):
#         temp = abs(series_1[index] - series_2[index])
#         absError.append(temp)
#     return np.mean(absError)
#
# #2.c
# #Calculating RMSE/MAE(careful with the beginning & ending sides)
# #CMA-24 smoothing
# smoothing_cma24_RMSE = np.sqrt(metrics.mean_squared_error(time_series_smoothing_cma[12: -12], time_series[12:-12]))
# smoothing_cma24_MAE = mae(time_series_smoothing_cma[12: -12], time_series[12:-12])
# print("CMA-24 smoothing RMSE: ", smoothing_cma24_RMSE)
# print("CMA-24 smoothing MAE: ", smoothing_cma24_MAE)
# #Pandas smoothing
# smoothing_pandas_RMSE = np.sqrt(metrics.mean_squared_error(time_series_smoothing_rolling[12:-12], time_series[12:-12]))
# smoothing_pandas_MAE = mae(time_series_smoothing_rolling[12:-12], time_series[12:-12])
# print("Pandas smoothing RMSE: ", smoothing_pandas_RMSE)
# print("Pandas smoothing MAE: ", smoothing_pandas_MAE)
#
# #2.d
# #CMA-5 Predicting
# #Case 2 Using the original data to Predict forever.
# def cam5_predict(time_series):
#     predict_series = []
#     for index in range(len(time_series)):
#         if index - 5 < 0:
#             predict_series.append(None)
#         else:
#             predict_series.append(time_series[index - 5: index].mean())
#     return predict_series
#
# y_predict = cam5_predict(time_series)
# y_predict_last_four_month = y_predict[-4:]
# print("The predicted last four months: ", y_predict_last_four_month)
#
#
# #2.e
# #Least Linear Regression
# #Creating traning data set
# x5_train = time_series[0: time_series.__len__() - 5]
# x4_train = time_series[1: time_series.__len__() - 4]
# x3_train = time_series[2: time_series.__len__() - 3]
# x2_train = time_series[3: time_series.__len__() - 2]
# x1_train = time_series[4: time_series.__len__() - 1]
#
# x_train = pd.DataFrame({'x1': np.array(x1_train), 'x2': np.array(x2_train), 'x3': np.array(x3_train), 'x4': np.array(x4_train),'x5': np.array(x5_train)})
# y_train = np.array(time_series[5: ])
#
# x_train = np.reshape(x_train, (324, 5))
# y_train = np.reshape(y_train, 324)
#
#
# #Creating Training Model
# # 2(d)
# lm = LinearRegression(fit_intercept = False)
# lm.fit(x_train, y_train)
# print("Coefficients: {0}".format(lm.coef_))
# print("Intercept: {0}".format(lm.intercept_))
# print("Total model: y = {0} + {1} X1 + {2} X2 + {3} X3 + {4} X4 + {5} X5".format(lm.intercept_, lm.coef_[0], lm.coef_[1], lm.coef_[2], lm.coef_[3], lm.coef_[4]))
# print("Variance score (R^2): {0:.3f}".format(lm.score(x_train, y_train)))
# #Predict the data from last four month
# x_test = x_train[-4: ]
# y_pred_last= lm.predict(x_test)
#
# #Graphing
# plt.figure()
# plt.plot(np.array(time_series[-4:]), 'r-', label = "Orginal Data")
# plt.plot(y_pred_last, 'b-', label = "Predict data")
# plt.title('Orginal and predicted Price of the last four months')
# plt.xlabel('Year')
# plt.ylabel('Price')
# plt.legend()
#
#
# #Report the scale-dependent measures RMSE and MAE
# #Calculating MAE
# def mae(series_1, series_2):
#     absError = []
#     for index in range(len(series_1)):
#         temp = abs(series_1[index] - series_2[index])
#         absError.append(temp)
#     return np.mean(absError)
#
# #by using CMA-5
# y_pred_cam5 = x_train.T.mean()
# cma5_RMSE = np.sqrt(metrics.mean_squared_error(y_train, y_pred_cam5))
# cma5_MAE = mae(y_train, y_pred_cam5)
# print ("cma5_RMES:", cma5_RMSE)
# print ("cma5_MSE:", cma5_MAE)
#
# #By using Linear Regression
# y_pred_regress = lm.predict(x_train)
# regress_RMSE = np.sqrt(metrics.mean_squared_error(y_train, y_pred_regress))
# regress_MAE = mae(y_train, y_pred_regress)
# print ("regress_RMES:", regress_RMSE)
# print ("regress_MSE:", regress_MAE)
#
# plt.show()
#
#
# Task 3. Python Code.
# import math
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn import metrics
#
# all_data = pd.read_csv("plastics.csv")
# #print(all_data)
#
# month = all_data['Month']
# x_valua = all_data['x']
#
# plt.plot(x_valua)
#
#
# #plt.show()
# #the peak mouth appear from 7,19，32，44，53，Seasonality is 12 month，trend cycle is 6 month.
#
# #classical multiplicative decomposition
# #Using Sqrt
# x_valua_sqrt = np.sqrt(x_valua)
#
# #3.(b) classical multiplicative decomposition to calculate the trend-cucle and seasonal indices.
# #Smoothing 12X2 MA, update trend- cycle
# Trend = x_valua_sqrt.rolling(window = 12, center = True).mean().rolling(window = 2, center = True).mean()
# plt.figure()
# plt.plot(x_valua_sqrt, color='red')
# plt.plot(Trend, color='blue')
# plt.title('Initial TREND estimate ')
# plt.xlabel('Month')
# plt.ylabel('Number')
#
# # Calculating seasonal index
# x_valua_res = x_valua_sqrt / Trend
# x_res = np.nan_to_num(x_valua_res)
# monthly_s = np.reshape(x_res, (5, 12))
# monthly_avg = np.array(pd.DataFrame(monthly_s).mean())
# monthly_avg_normalized = monthly_avg / monthly_avg.mean()
#
#
# #3.(c)
# #Calculating seasonally adjusted data
# tiled_avg = np.tile(monthly_avg_normalized, 5)
# seasonally_adjusted =  x_valua_sqrt / tiled_avg
#
# fig, ax = plt.subplots(4, 1)
# ax[0].plot(x_valua_sqrt)
# ax[1].plot(Trend)
# ax[2].plot(x_valua_res)
# ax[3].plot(seasonally_adjusted)
# ax[0].legend(['sqrt'], loc=2)
# ax[1].legend(['trend'], loc=2)
# ax[2].legend(['seasonality'], loc=2)
# ax[3].legend(['seasonal adjusted'], loc=2)
#
#
# #re-estimate the trend-cycle
# Trend_final = seasonally_adjusted.rolling(window = 12, center = True).mean().rolling(window = 2, center = True).mean()
# plt.figure()
# plt.plot(Trend_final)
# plt.title('Final TREND')
# plt.xlabel('Month')
# plt.ylabel('Number')
#
# #3.(d)
# #Change one observation to be an outliner
# #Randomlly choose one variable add 500
# random = np.random.randint(low = 0, high = x_valua.size)
# x_valua_new = pd.Series(np.array(x_valua))
# x_valua_new[random] = x_valua_new[random] + 500
# x_valua_new_sqrt = np.sqrt(x_valua_new)
#
# #Re-calculate
# seasonally_adjusted_new =  x_valua_new_sqrt / tiled_avg
#
# plt.figure()
# plt.plot(seasonally_adjusted)
# plt.plot(seasonally_adjusted_new)
# plt.title('Adjusted Seasonally')
# plt.xlabel('Month')
# plt.ylabel('Number')
#
# #3.(e)
# #regression model for the trend -cycle component
# x_train = pd.DataFrame(all_data['Month'])
# y_train = pd.Series(Trend_final)
#
# #Delete NaN in ey_train
# x_train = x_train[~ np.isnan(y_train)]
# y_train = y_train[~ np.isnan(y_train)]
#
# lm = LinearRegression()
# lm.fit(x_train, y_train)
#
# print("Coefficients: {0}".format(lm.coef_))
# print("Intercept: {0}".format(lm.intercept_))
# print("Total model: y = {0} + {1} X1".format(lm.intercept_, lm.coef_[0]))
# print("Variance score (R^2): {0:.3f}".format(lm.score(x_train, y_train)))
#
# #predict
# #Calculate the prediction part in trend -cycle
# x_test = pd.DataFrame([61, 62, 63])
# y_pred = lm.predict(x_test)
#
# #Calculating seasonal indices
# season = x_test.mod(12)
# season_indices = monthly_avg_normalized[np.array(season)]
#
# #Calculate the final predict value
# T = np.array(y_pred)
# S = np.reshape(np.array(season_indices), (1, 3))
# print(T)
# print(S)
# predict_value = np.multiply(np.multiply(T, S), np.multiply(T, S))
#
#
# print(predict_value)
# plt.show()
#
# TASK 4. Python Code.
# import math
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.linear_model import LinearRegression
# from sklearn import metrics
# from datetime import datetime
#
# all_data = pd.read_csv("Airline.csv")
# #print(all_data)
#
# #decomposite data
# passengers = all_data['Passengers']
# month = np.array(all_data['Month'], dtype = np.str)
# year = np.array(all_data['Year'], dtype = np.str)
# #Month+Year=date
# date = pd.to_datetime(pd.DataFrame({'year': all_data['Year'], 'month': all_data['Month'], 'day': np.ones(all_data['Year'].size)}))
# #Create time_series
# data_series = pd.Series(data = np.array(passengers), index = date)
# plt.figure()
# plt.plot(data_series)
# plt.xlabel('Date')
# plt.ylabel('Passengers')
#
# #Holt’s linear trend method
# #Y_t+1 = l_t + h * b_t
# #level: l_t = alpha * y_t + (1 - alpha) * (l_t-1 + b_t-1)
# #trend: b_t = beta(l_t - l_t-1) + (1 - beta) * b_t-1
#
# def holt_linear_trend(series, alpha, beta, h):
#     #Identify the initial value for l,b
#     l = [series[0]]
#     b = [series[1] - series[0]]
#     series_smooth = []
#     for index in range(len(series)):
#         l.append(alpha * series[index] + (1 - alpha) * (l[index] + b[index]))
#         b.append(beta * (l[index + 1] - l[index]) + (1 - beta) * b[index - 1])
#         series_smooth.append(l[index + 1] + h * b[index + 1])
#     return series_smooth, l, b
#
#
#
# #Identify SSE
# def sse(x, y):
#     return np.sum(np.power(x - y,2))
#
# #Test alpha, beta
# alpha_test = [0.2, 0.4, 0.6, 0.8]
# beta_test = [0.2, 0.4, 0.6, 0.8]
# sse_test = []
# #calculating all the SSE combined by alpha, beta
# #For the one-step ahead，We calculate y_t by y_prend_t+1，which means y_0 to get y_prend_1， h = 1
# for alpha in alpha_test:
#     for beta in beta_test:
#         smooth_passengers, l, b = holt_linear_trend(passengers, alpha, beta, 1)
#         sse_test.append(sse(smooth_passengers[:-1], passengers[1:]))
#
# #Finding sse with the minimum alpha and beta
# min_index = np.where(sse_test == min(sse_test))
# alpha_best = alpha_test[(min_index[0] // 4)[0]]
# beta_best = beta_test[(min_index[0] % 4)[0]]
# print("Best alpha in 16 cases: ", alpha_best)
# print("Best beta in 16 cases: ", beta_best)
#
# #Choose the best alpha, beta
# alphas = np.arange(0.01,1,0.01)
# betas = np.arange(0.01,1,0.01)
# sses = []
#
# #When using the two-step ahead. We use Y_t to predict Y_(t+2), which means we use Y_0  predicting Y_2
# for alpha in alphas:
#     for beta in betas:
#         smooth_passengers, l, b = holt_linear_trend(passengers, alpha, beta, 2)
#         sses.append(sse(smooth_passengers[:-2], passengers[2:]))
#
# X, Y = np.meshgrid(alphas, betas)
# Z = np.reshape(sses, (len(alphas), len(betas)))
#
# #3D Plot
# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.plot_surface(X, Y, Z, rstride = 1, cstride = 10000, cmap = 'rainbow')
#
# #Find the minimum SSE index.
# min_sses_index = np.where(sses == min(sses))
# alpha_ = alphas[(min_sses_index[0] // len(alphas))[0]]
# beta_ = alphas[(min_sses_index[0] % len(betas))[0]]
# print("Best alpha in {0} cases: {1}".format(len(alphas), alpha_))
# print("Best beta in {0} cases: {1}".format(len(betas), beta_))
#
# #Use two-step ahead to predict next 4 months.
# passengers_prend, l, b = holt_linear_trend(passengers, alpha_, beta_, 2)
# predict_four_month = []
# for i in range(4):
#     predict_four_month.append(l[-1] + (i + 1) * b[-1])
# print("Predict passengers of the next four months: ", predict_four_month)
#
# # Compare original data  series  and the  smoothing series
# #orginial data
# original_data = passengers
# #smooth data
# smooth_data = passengers_prend
# #Graphing
# plt.figure()
# plt.plot(original_data, color = 'red', label = 'original data')
# plt.plot(smooth_data, color = 'blue', label = 'smooth data')
#
# plt.show()
#
#
