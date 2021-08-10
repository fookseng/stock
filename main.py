# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:36:39 2021

@author: wyyac
"""

import pandas as pd
from pandas_datareader import data as web
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pytickersymbols import PyTickerSymbols
import requests
import csv
import numpy as np
import time
from yahoo_fin import stock_info as si
from tqdm import tqdm
from ftplib import FTP
import os
#import webbrowser
from multiprocessing import Pool

class Stock:
    '''
    1. All the following functions are designed based on US stock market data from yahoo finance.
    For Taiwan stocks, the DataFrame might be a bit different. The below code is provided for your 
    reference to change the DataFrame in order to use other functions correctly if it is necessary.
        ->dataset.columns = dataset.columns.droplevel(1)
    
    e.g.
        > tsmc = Stock("2330.TW", 365)
        > dataset = tsmc.data
        > dataset.columns = dataset.columns.droplevel(1) 
        > tsmc.data = dataset
    
    2. The function 'job' is for multi-threading purpose, please design one based on your own needs.
    
    3. Please confirm the data is not 'None', else you will not be able to call functions.
    '''
    def __init__(self, symbol, days):
        self.symbol = symbol
        self.data = self.get_ndays_data(days)
    
    def get_ndays_data(self, days):
        try:
            return web.DataReader(self.symbol, data_source='yahoo', start=(datetime.now() - timedelta(days=days)).strftime('%d-%m-%Y'))
        except Exception as ex:
            print('Error: ' , ex)
            return None
        
    def get_data_on_date(self, year, month, day):
        start = datetime(year, month, day)
        try:
            data = web.DataReader(self.symbol, data_source='yahoo', start=start, end = start)
            return data, data.loc[start, 'Close']
        except Exception as ex:
            print('Error: ' , ex)
            return None
    
    # This function modify the content of 'self.data' to a specified range of date instead of the default values. 
    def modify_data_from_date_to_date(self, start_year, start_month, start_day, end_year, end_month, end_day):
        start = datetime(start_year, start_month, start_day)
        end = datetime(end_year, end_month, end_day)
        try:
            self.data = web.DataReader(self.symbol, data_source='yahoo', start=start, end = end)
        except Exception as ex:
            print('Error: ', ex)
            return None
    
    def real_time_data(self):
        return datetime.now().strftime('%b %d %Y  %T'), si.get_quote_table(self.symbol, dict_result=False), si.get_live_price(self.symbol)

    def kd_calculation(self):
        data_df = self.data.copy()
        data_df['min'] = data_df['Low'].rolling(9).min()
        data_df['max'] = data_df['High'].rolling(9).max()
        data_df['RSV'] = 100* (data_df['Close'] - data_df['min'])/(data_df['max'] - data_df['min'])
        data_df = data_df.dropna()
        # 計算K
        # K的初始值定為50
        K_list = [50]
        for num,rsv in enumerate(list(data_df['RSV'])):
            K_yestarday = K_list[num]
            K_today = 2/3 * K_yestarday + 1/3 * rsv
            K_list.append(K_today)
        data_df['K'] = K_list[1:]
        # 計算D
        # D的初始值定為50
        D_list = [50]
        for num,K in enumerate(list(data_df['K'])):
            D_yestarday = D_list[num]
            D_today = 2/3 * D_yestarday + 1/3 * K
            D_list.append(D_today)
        data_df['D'] = D_list[1:]
        self.data = pd.merge(self.data,data_df[['K','D']],left_index=True,right_index=True,how='left')

    def macd_calculation(self):
        data_df = self.data.copy()
        data_df['EMA_10'] = data_df['Close'].ewm(span=10, adjust=False).mean()
        data_df['EMA_20'] = data_df['Close'].ewm(span=20, adjust=False).mean()
        data_df['DIF'] = data_df['EMA_10'] - data_df['EMA_20']
        data_df['DEA'] = data_df['DIF'].ewm(span=9, adjust=False).mean()
        data_df['MACD'] = (data_df['DIF'] - data_df['DEA'])
        self.data = pd.merge(self.data, data_df[['DIF', 'DEA', 'MACD']], left_index=True, right_index=True, how='left')
        
    def ma_calculation(self):
        data_df = self.data.copy()
        data_df['Average5'] = self.data.Close.rolling(window=5, min_periods=1).mean()
        data_df['Average10'] = self.data.Close.rolling(window=10, min_periods=1).mean()
        data_df['Average20'] = self.data.Close.rolling(window=20, min_periods=1).mean()
        data_df['Average60'] = self.data.Close.rolling(window=60, min_periods=1).mean()
        self.data = pd.merge(self.data, data_df[['Average5', 'Average10', 'Average20', 'Average60']], left_index=True, right_index=True, how='left')

    def kd_golden_cross(self):
        if not "K" in self.data:
            self.kd_calculation()
        day = 10 # any large number will do
        if self.data.K[-1] > self.data.D[-1]:
            for i in range(2,6):
                if self.data.K[-i] < self.data.D[-i]:
                    day = i
                    break
        if ((self.data.K[-1] - self.data.K[-2] > 1) and (self.data.D[-1] - self.data.D[-2] > 1) and (self.data.K[-1] - self.data.D[-1] > 1) and day <= 3):
            return 1
        else:
            return 0
    
    def ma_queue_up(self):
        if not "Average5" in self.data:
            self.ma_calculation()
        if (self.data.Average5[-1] > self.data.Average10[-1] and self.data.Average5[-2] > self.data.Average10[-2]):
            if ((self.data.Average5[-1]>self.data.Average5[-2]) and (self.data.Average10[-1] > self.data.Average10[-2])):
                if (self.data.Average10[-1] > self.data.Average20[-1] and self.data.Average10[-2] > self.data.Average20[-2]):
                    if (self.data.Average20[-1] > self.data.Average20[-2]):
                        if (self.data.Average20[-1] > self.data.Average60[-1] and self.data.Average20[-2] > self.data.Average60[-2]):
                              if (self.data.Average60[-1] > self.data.Average60[-2]):
                                  return 4
                        else:
                            return 3
                else:
                    return 2
        else:
            return 0
    
    def ma_queue_down(self):
        if not "Average5" in self.data:
            self.ma_calculation()
        if (self.data.Average5[-1] < self.data.Average10[-1] and self.data.Average5[-2] < self.data.Average10[-2]):
            if ((self.data.Average5[-1]<self.data.Average5[-2]) and (self.data.Average10[-1]<self.data.Average10[-2])):
                if (self.data.Average10[-1] < self.data.Average20[-1] and self.data.Average10[-2] < self.data.Average20[-2]):
                    if (self.data.Average20[-1] < self.data.Average20[-2]):
                        if (self.data.Average20[-1] < self.data.Average60[-1] and self.data.Average20[-2] < self.data.Average60[-2]):
                              if (self.data.Average60[-1] < self.data.Average60[-2]):
                                  return 4
                        else:
                            return 3
                else:
                    return 2
        else:
            return 0
        
    def ma_golden_cross(self, ma1, ma2):
        if not "Average5" in self.data:
            self.ma_calculation()
        if ma1 == 5:
            temp1 = self.data.Average5
        elif ma1 == 10:
            temp1 = self.data.Average10
        elif ma1 ==20:
            temp1 = self.data.Average20
        elif ma1 == 60:
            temp1 = self.data.Average60
        if ma2 == 5:
            temp2 = self.data.Average5
        elif ma2 == 10:
            temp2 = self.data.Average10
        elif ma2 ==20:
            temp2 = self.data.Average20
        elif ma2 == 60:
            temp2 = self.data.Average60
        day = 10  # any large number will do
        if temp1[-1] > temp2[-1]:
            for i in range(2, 6):
                if temp1[-i] < temp2[-i]:
                    day = i
                    break
        if ((temp1[-1] > temp1[-2]) and (temp2[-1] > temp2[-2]) and (temp1[-1] > temp2[-1]) and day <= 3):
            return 1
        else:
            return 0
    
    def turning_point(self):
        data_cp = self.data.copy()
        # Since we going to convert dataframe to list, and using index to access the value. 
        # Hence, we must confirm that the list structure and elements always remain the same.
        # [Date, High, Low, Open, Close, Volume, Adj Close, Average5]
        selected_columns = data_cp[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']]
        data_df = selected_columns.copy()
        data_df['Average5'] = self.data.Close.rolling(window=5, min_periods=1).mean()
        df = pd.merge(self.data, data_df[['Average5']], left_index=True, right_index=True, how='left')
        df.reset_index(level=0, inplace=True)
        df = df.values.tolist()
        
        # [date, value]
        record = []
        maxv = []
        minv = []
        state = 0
        # check the first point(start from the 6th data) is above or below MA5
        if df[5][4] > df[5][7]:
            state = 1
            maxv = [df[0][0], df[0][1]]
        else:
            state = 0
            minv = [df[0][0], df[0][2]]
    
        # [Date, High, Low, Open, Close, Volume, Adj Close, Average5]
        for i in range(7, len(df)):
            if state:
                # continue above MA5
                if df[i][4] >= df[i][-1]:
                    state = 1
                    #print('0')
                    # update the maximum value
                    if df[i][1] > maxv[1]:
                        maxv = [df[i][0], df[i][1]]
                        #print('1')
                # go below ma5
                else:
                    state = 0
                    if df[i][1] > maxv[1]:
                        maxv[1] = df[i][1]
                    record.append(maxv)
                    maxv = [0, 0]
                    minv = [df[i][0], df[i][2]]
                    #print('2')
            else:
                # continue below MA5
                if df[i][4] <= df[i][-1]:
                    state = 0
                    #print('3')
                    if df[i][2] < minv[1]:
                        minv = [df[i][0], df[i][2]]
                        #print('4')
                # go above MA5
                else:
                    state = 1
                    if df[i][2] < minv[1]:
                        minv[1] = df[i][2]
                    record.append(minv)
                    minv = [0, 0]
                    maxv = [df[i][0], df[i][1]]
                    #print('5')
    
            if i == len(df) - 1:
                if df[i][4] <= df[i][-1]:
                    record.append(minv)
                else:
                    record.append(maxv)
        return pd.DataFrame(record, columns=['Date', 'Price']).set_index('Date')
    
    def trend(self):
        data = self.turning_point()
        data.reset_index(level=0, inplace=True)
        data = data.values.tolist()
    
        prev_head = 0
        prev_tail = 9999999
        mx = []
        mn = []
        r = []
        state = 0
        date_mn = data[0][0]
        date_mx = data[0][0]
    
        if data[0][1] >= data[1][1]:
            high = 0
            prev_head = data[0][1]
            mx.append([data[0][0], data[0][1]])
        else:
            high = 1
            prev_tail = data[0][1]
            mn.append([data[0][0], data[0][1]])
    
        for i in range(1, len(data)):
            if high:
                high = 0
                mx.append([data[i][0], data[i][1]])
                # check if there is HHH(toutougao)
                if data[i][1] >= prev_head:
                    if state != 1:
                        state = 1
                        # look for lowest value
                        lowest = mn[0][1]
                        res = mn[0]
                        date_mn = mn[0][0]
                        if len(mn) == 1:
                            pass
                        else:
                            for j in range(1, len(mn)):
                                if mn[j][1] <= lowest:
                                    lowest = mn[j][1]
                                    res = mn[j]
                                    date_mn = mn[j][0]
                        # Append result
                        r.append(res)
                        # update mn and mx list
                        mn = []
                        if len(mx) >= 1:
                            newmx = []
                            for k in range(len(mx)):
                                if mx[k][0] <= date_mn:
                                    newmx.append(mx[k])
                            for l in newmx:
                                mx.remove(l)
                # update prev_head
                prev_head = data[i][1]
            else:
                high = 1
                mn.append([data[i][0], data[i][1]])
                if data[i][1] <= prev_tail:
                    if state != 2:
                        state = 2
                        highest = mx[0][1]
                        res = mx[0]
                        date_mx = mx[0][0]
                        if len(mx) == 1:
                            pass
                        else:
                            for j in range(1, len(mx)):
                                if mx[j][1] >= highest:
                                    highest = mx[j][1]
                                    res = mx[j]
                                    date_mx = mx[j][0]
                        r.append(res)
                        mx = []
                        if len(mn) >= 1:
                            newmn = []
                            for k in range(len(mn)):
                                if mn[k][0] <= date_mx:
                                    newmn.append(mn[k])
                            for l in newmn:
                                mn.remove(l)
                prev_tail = data[i][1]
        r.append(data[-1])
        return pd.DataFrame(r, columns=['Date', 'Trend']).set_index('Date')
    
    '''
    This function should be updated for higher precision.
    '''
    def one_bottom(self, num_day):
        data = self.data.copy()
        today_data = self.data.iloc[-1]
        i = -2
        days = 0
        while True:
            if today_data.High > data.High[i]:
                if today_data.Volume > data.Volume[i]:
                    days += 1
                    i -= 1
                else:
                    break
            else:
                break
            if days >= num_day:
                return 1
        return 0
    
    '''
    Below functions are still under construction ... --------------------------------------------------------------------------------
    '''
    
    def reversal_candlestick(self):
        data = self.data.iloc[-1]    
        if(data.High == data.Close) and (((data.Close - data.Open) / data.Open) *100) >= 4.5:
            return 1
        if(data.Low == data.Close) and (((data.Open - data.Close) / data.Open) *100) >= 4.5:
            return 1
        if data.Open > data.Close and data.Close == data.Low:
            if (data.High - data.Open) >= 2* (data.Open - data.Close):
                return 1
        if data.Open < data.Close and data.Open == data.Low:
            if (data.High - data.Close) >= 2 * (data.Close - data.Open):
                return 1
        if data.Open > data.Close and data.High == data.Open:
            if (data.Close - data.Low) >= 2 * (data.Open - data.Close):
                return 1
        if data.Open < data.Close and data.High == data.Close:
            if (data.Open - data.Low) >= 2 * (data.Close - data.Open):
                return 1
        if(data.Open == data.Close):
            return 1
        return 0
    
    def N_shape(self):
        df1 = self.turning_point()
        df2 = self.trend()
        # bear to bull, bottom confirmation
        if df2.Trend[-3] - df2.Trend[-2] > 0 and df2.Trend[-1] - df2.Trend[-2] > 0:
            # n-shape confirmation
            if df2.Trend[-2] == df1.Price[-3] and df1.Price[-1] >= ((df1.Price[-2] - df1.Price[-3])/2)+df1.Price[-3]:
                return 1
            if df2.Trend[-2] == df1.Price[-4] and df1.Price[-2] >= ((df1.Price[-3] - df1.Price[-4])/2)+df1.Price[-4]:
                return 1
        return 0
    
    def bowl_shape(self):
        data = self.data.copy()

        ### Date of end of U-shape
        #end_date = datetime.strptime(datetime.now(), '%d/%m/%Y')
        #data = data[data.index <= end_date]
    
        current_close_price = data.iloc[-1]['Close']
        current_date = datetime.strptime(str(list(data.index)[-1]), '%Y-%m-%d %H:%M:%S')
    
        ### Look back for similar price
        possible_starting_date = data.iloc[:-1]
        ### Conditions checking
        exist = 0
        ### The current closing price is between low and high price of the starting date.
        for d in possible_starting_date:
            if (d['Low'] <= current_close_price) & (d['High'] >= current_close_price):
                possible_starting_date = d
                ### The possible starting date must be at least 10 days ago
                if d.index < (current_date - timedelta(days=10)):
                    return 0
                else:
                    number_of_days = d.index - current_date
                    exist = 1
                    break
        if exist:
            # make sure the price of the next few days is decreasing and dropping on each days.
            data_after_date = data[data.index >= possible_starting_date.index]
            for i in range(number_of_days//3):
                if (data_after_date.iloc[i + 1]["Open"] > data_after_date.iloc[i + 1]["Close"]) and (data_after_date.iloc[i]["Close"] > data_after_date.iloc[i + 1]["Close"]):
                    pass
                else:
                    return 0
            # make sure the price of the last few days is increasing
                if (data_after_date.iloc[-(i + 1)]["Open"] < data_after_date.iloc[-(i + 1)]["Close"]) and (data_after_date.iloc[-(i+1)]["Close"] > data_after_date.iloc[-(i + 2)]["Close"]):
                    pass
                else:
                    return 0
    
        ### Fit a polynomial equation
        close_price_data = np.array(data_after_date['Close'], dtype='float')
        x = np.arange(0, len(close_price_data), 1)
        model = np.poly1d(np.polyfit(x, close_price_data, 2))
        if model.c[0] > 0.1:
            #possible_starting_date = possible_starting_date.drop(index=date)
            #plt.plot(x, close_data)
            #plt.plot(x, model(x))
            #print(model.c[0])
            #print(model)
            #plt.show()
            return 1
        else:
            return 0
    
    # updating
    def support_pressure(self):
        df_low = self.data['Low'].values.tolist()
        df_high = self.data['High'].values.tolist()
        df_close = self.data['Close'].values.tolist()
        upward = []
        downward = []
        updel = []
        downdel = []
        dayslow = 0
        dayshigh = 0
        for i in range(1, len(df_low)):
            #upward jump
            if df_low[i] > df_high[i-1]:
                upward.append([df_low[i], df_high[i-1], i])
            #downward jump
            if df_low[i-1] > df_high[i]:
                downward.append([df_high[i], df_low[i-1], i])
        if upward:
            for j in upward:
                for k in df_close[j[2]:]:
                    if k < j[1]:
                        dayslow += 1
                if dayslow >= 2:
                    updel.append(j)
                    dayslow = 0
        if downward:
            for l in downward:
                for m in df_close[l[2]:]:
                    if m > l[1]:
                        dayshigh += 1
                if dayshigh >= 2:
                    downdel.append(l)
                    dayshigh = 0
        if updel:
            for n in updel:
                upward.remove(n)
        if downdel:
            for o in downdel:
                downward.remove(o)
        return upward, downward
    
    def met_sup_pres(self):
        _,_,price = self.real_time_data()
        up, down = self.support_pressure()
        total = up + down
        for i in total:
            if (price <= max(i[0], i[1]) and price >= min(i[0], i[1])):
                return 1
        return 0

    ''' 
    End of not tested functions -------------------------------------------------------------------------------------------------------------- 
    '''
    def plot_graph(self, display, save):
        up, down = self.support_pressure()
        if not "K" in self.data:
            self.kd_calculation()
        if not "Average5" in self.data:
            self.ma_calculation()
        if not "MACD" in self.data:
            self.macd_calculation()
        data = self.data.copy()
        df = self.turning_point()
        dt = self.trend()
        # Plot figure
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=("K graph", "Volume", "KD", "MACD"), row_width=[0.2, 0.2, 0.1, 0.5])
        fig.update_xaxes(type="category", categoryorder='category ascending', showgrid=False,
                         rangeslider=dict(visible=False), range=[-5, (datetime.date(data.index[-1]) - datetime.date(data.index[0])).days], showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(title_text="Price", row=1, col=1, zeroline = False,showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(title_text="Volume", row=2, col=1, nticks=2, zeroline=False)
        fig.update_yaxes(title_text="Value", range=[0, 100], row=3, col=1, nticks= 6, zeroline = False, showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(title_text="Value", row=4, col=1, zeroline = False, showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showgrid=True, gridwidth=0.1, gridcolor='slategray')

        # row 1:
        # K lines, MA5,10,20,60
        fig.add_trace(go.Candlestick(x=data.index, open=data.Open, close=data.Close, high=data.High, low=data.Low,
                                     increasing_line_color='darkgreen',
                                     decreasing_line_color='darkred', name='K lines/K线', showlegend=True), row=1, col=1)
        # Latest close price
        fig.add_trace(
            go.Scatter(x=data.index, y=[data.iloc[-1].Close]* len(data.index), mode='lines', line=dict(color='gray', width=1, dash='dash'),
                       name='Current Price/现价',
                       showlegend=True), row=1, col=1)
        # MA lines
        fig.add_trace(
            go.Scatter(x=data.index, y=data.Average5, mode='lines', line=dict(color='orangered', width=1.5), name='MA5/五均',
                       showlegend=True), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=data.index, y=data.Average10, mode='lines', line=dict(color='brown', width=1.5), name='MA10/十均',
                       showlegend=True), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=data.index, y=data.Average20, mode='lines', line=dict(color='midnightblue', width=1.5),
                       name='MA20/二十均', showlegend=True), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=data.index, y=data.Average60, mode='lines', line=dict(color='black', width=1.5), name='MA60/六十均',
                       showlegend=True), row=1, col=1)
        # Twist and Trend
        fig.add_trace(
            go.Scatter(x=df.index, y=df.Price, mode='markers', line=dict(color='black', width=5), name='Turning Point 转折点',
                       showlegend=True), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=dt.index, y=dt.Trend, mode='lines', line=dict(color='deeppink', width=3, dash='dot'),
                       name='Bull or Bear/趋势波',
                       showlegend=True), row=1, col=1)
        
        '''
        Additional
        '''
        '''
        fig.add_trace(
            go.Scatter(x=['2021-01-29','2021-05-28'], y=[414,212], mode='lines', line=dict(color='gray', width=2),
                       name='切线1',
                       showlegend=True), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=['2021-01-29','2021-05-28'], y=[10,152], mode='lines', line=dict(color='gray', width=2),
                       name='切线2',
                       showlegend=True), row=1, col=1)
        '''

        # Support & Pressure
        if up:
            for i in up:
                # fig.add_trace(go.Scatter(x=data.index[i[2]:], y=[i[0]] * days, mode='lines',
                #                          line=dict(color='green', width=1, dash='dash'), name='Boundary',
                #                          showlegend=True), row=1, col=1)
                # fig.add_trace(go.Scatter(x=data.index[i[2]:], y=[i[1]] * days, mode='lines',
                #                          line=dict(color='green', width=1, dash='dash'), name='Boundary',
                #                          showlegend=True), row=1, col=1)
                fig.add_shape(type="rect",
                              x0=data.index[i[2]], y0=i[0], x1=data.index[-1], y1=i[1],
                              line=dict(
                                  color='rgba(231,169,105,0.5)',
                                  width=2,
                              ),
                              fillcolor='rgba(255,156,73,0.4)',
                              layer = 'below',
                              opacity= 1,
                              row = 1,
                              col = 1
                              )
        if down:
            for j in down:
                fig.add_shape(type="rect",
                              x0=data.index[j[2]], y0=j[0], x1=data.index[-1], y1=j[1],
                              line=dict(
                                  color='rgba(231,169,105,0.5)',
                                  width=2,
                              ),
                              fillcolor='rgba(255,156,73,0.4)',
                              layer='below',
                              opacity=1,
                              row = 1,
                              col = 1
                              )

        # row 2:
        # Volume
        clr = [1]
        for y in range(1, len(data.Volume)):
            if data.Volume[y] >= 2*(data.Volume[y-1]):
                clr.append(5)
            elif data.Close[y] >= data.Close[y-1]:
                clr.append(1)
            else:
                clr.append(0)
        clr = np.array(clr)
        colors_vol = np.where(clr > 2, 'black', np.where(clr > 0, 'green', 'red'))
        fig.add_trace(go.Bar(x=data.index, y=data.Volume, marker_color=colors_vol, name='Volume-bar/交易量'),
                      row=2, col=1)
        # row 3: KD
        fig.add_trace(go.Scatter(x=data.index, y=data.K, mode='lines', line=dict(color='red', width=2), name='K-fast',
                                 showlegend=True), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data.D, mode='lines', line=dict(color='darkblue', width=2), name='D-slow',
                                 showlegend=True), row=3, col=1)
        # fig.add_trace(go.Scatter(x=data.index, y=[80] * days, mode='lines', line=dict(color='green', width=1, dash='dash'),
        #                name='Boundary', showlegend=True), row=2, col=1)
        # row4: MACD
        fig.add_trace(go.Scatter(x=data.index, y=data.DIF, mode='lines', line=dict(color='red', width=2), name='DIF-fast',
                       showlegend=True), row=4, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data.DEA, mode='lines', line=dict(color='royalblue', width=2), name='DEA-slow'), row=4, col=1)
        colors = np.where(data["MACD"] < 0, 'green', 'mediumpurple')
        fig.add_trace(go.Bar(x=data.index, y=data.MACD, marker_color=colors, name='MACD-bar'),
                      row=4, col=1)

        fig.update_layout(title_text=f"Stock: {self.symbol} ${{:.2f}}".format(self.data.iloc[-1].Close), height=1200, legend_title="I am a Legend",
                          plot_bgcolor='rgba(52,73,94,0.3)',
                          # paper_bgcolor='rgba(44,63,80,1)',
                          font=dict(
                              family='Courier New, monospace',
                              size=18,
                              color="darkblue"
                          ))
        if save:
            if not os.path.exists('stock_graph'):
                os.makedirs('stock_graph')
            fig.write_html(f"stock_graph//{self.symbol}.html")
        if display:
            fig.show()
        
# Function for multi-threading purpose, design your own specifications here.
    def job(self):
        # Avoid getting TypeError.
        if self.data is None or self.ma_queue_up() is None:
            return None
        # Design your own conditions, remain return value unchange.
        if self.ma_queue_up()>=3 and self.kd_golden_cross():
            self.plot_graph(0, 1)
            return self.symbol
        else:
            return None
    
    def job_2(self, n):
        # Avoid getting TypeError.
        if self.data is None:
            return None
        # Design your own conditions, remain return value unchange.
        if self.one_bottom(n):
            self.plot_graph(0, 1)
            return self.symbol
        else:
            return None
        
"""
mode 2 = 上市 mode 4 = 上櫃.
provide a filename if you wish to write the result into a .csv file, else provide integer 0.
"""
def get_TW_stock_symbol(mode, filename):
    print('Accessing data ...')
    res = requests.get(f"http://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}")
    data = pd.read_html(res.text)[0]
    # setting column name
    data.columns = data.iloc[0]
    # delete first row
    data = data.iloc[1:]
    # 先移除row，再移除column，超過三個NaN則移除
    data = data.dropna(thresh=3, axis=0).dropna(thresh=3, axis=1)
    data = data.set_index('有價證券代號及名稱')
    lst = []
    for i in data.index[1:]:
        split = i.split()
        if split[0][1].isnumeric():
            current = [f'{split[0]}.TW']
            lst.append(current)
    if filename != 0:
        print('Writing data to file ...')
        file = open(f'{filename}.csv', 'w+', newline='')
        with file:
            write = csv.writer(file)
            write.writerows(lst)
        print('Done!')
    return lst

def get_US_stock_symbol():
    stock_data = PyTickerSymbols()
    index_list = stock_data.get_all_indices()
    print(index_list)
    temp_list = []
    symbol_list = []
    for i in index_list:
        syml = stock_data.get_stocks_by_index(i)
        temp_list += list(syml)
    for i in range(len(temp_list)):
        symbol_list.append(temp_list[i].get('symbol'))
    return sorted(filter(None, set(symbol_list)))

class nasdaq_stock_symbol:
    def __init__(self):
        self.symbol=[]
    def __call__(self,s):
        s=s.decode(encoding='UTF-8')
        for sentence in s.split('\n'):
            new_symbol=sentence.split('|')[0]
            if new_symbol.isupper():
                self.symbol.append(new_symbol)
                
def get_nasdaq_symbol():
    ftp = FTP('ftp.nasdaqtrader.com')  # ftp://ftp.nasdaqtrader.com/SymbolDirectory/
    ftp.login()
    nasdaq = nasdaq_stock_symbol()
    for file in ['nasdaqlisted.txt', 'otherlisted.txt']:
        ftp.retrbinary('RETR /SymbolDirectory/{}'.format(file), nasdaq)
    return sorted(nasdaq.symbol)

def check_etoro_availability(symbol):
    from selenium import webdriver

    chrome_options = webdriver.ChromeOptions()
    #chrome_options.add_argument("--headless")
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    browser = webdriver.Chrome("C:\\Users\\user\\Downloads\\chromedriver_win32\\chromedriver.exe",
                               options=chrome_options)
    browser.get(f'https://www.etoro.com/markets/{symbol}')
    #time.sleep(2)
    count = 0
    while count < 2:
        browser.refresh()
        time.sleep(2)
        count += 1
    stocks = browser.find_elements_by_class_name("user-market-info")
    #stocks = browser.find_element_by_xpath("/html/body/ui-layout/div/div/div[2]/et-market/div/div/et-market-header/div/div[1]/div[2]/div[1]/div/h1").text

    if stocks:
        browser.quit()
        return 1
    else:
        browser.quit()
        return 0
    


''' 
main function 
'''
if __name__ == '__main__':
    
    gme = Stock('GME', 365)
    gme.plot_graph(0, 1)
    '''
    #stocks = get_US_stock_symbol()
    stocks = get_TW_stock_symbol(2, 0)
    ress = []
    ok=[]
    pool = Pool(processes=16)
    
    #f = open('C:\\Users\\wyyac\\Desktop\\Python_Project\\stock\\venv\\etoro.txt', 'r+')
    #total = [line[:-1] for line in f.readlines()]
    #f.close()
    
    for i in tqdm(stocks):
        ### for Taiwan stocks
        stock = Stock(i[0], 165) # i[0] is used instead of i, since the list looks like this: [[],[],[],[],...], not this [a,b,c,...].
        ### for US stocks
        #stock = Stock(i, 165)
        # *may* use more processes
        ress.append(pool.apply_async(stock.job_2, (15,)))
    for res in ress:
        if res.get() != None:
            ok.append(res.get())

    print(ok)
    '''    
    