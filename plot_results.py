import numpy as np
import cv2
import os
import datetime
import math
import imutils
import xlsxwriter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import pylab

max_val_idx = 88

#datetime vector
start_date = datetime.date(2019, 6, 27)
start_minutes = [datetime.datetime(2019, 6, 28, 5, 20), datetime.datetime(2019, 8, 1, 6, 11), datetime.datetime(2019, 6, 20, 5, 55), datetime.datetime(2019, 8, 15, 6, 10), datetime.datetime(2019, 7, 27, 5, 51), datetime.datetime(2019, 6, 28, 5, 31), datetime.datetime(2019, 8, 14, 6, 14)]
sheets = ["1001 min", "1002 min", "1003 min", "1006 min", "1007 min", "1008 min", "1009 min"]
temp_date = start_date
delta_day = datetime.timedelta(days=1)

for i in range(len(sheets)):
    # import data from excel
    data = pd.read_excel("C:\\Users\\joelb\\Downloads\\comparison.xlsx",sheet_name=sheets[i])
    start_min = start_minutes[i]
    time_val = data['time_val'].dropna().tolist()
    cum_val = data['cum_val'].dropna().tolist()

    time1 = data['time_aaron'].dropna().tolist()
    cum1 = data['cum_aaron'].dropna().tolist()

    time2 = data['time_joel'].dropna().tolist()
    cum2 = data['cum_joel'].dropna().tolist()

    dates_val = []
    dates1 = []
    dates2 = []

    for t in time_val:
        dates_val.append(start_min + datetime.timedelta(hours=t))
    for t in time1:
        dates1.append(start_min + datetime.timedelta(hours=t))
    for t in time2:
        dates2.append(start_min + datetime.timedelta(hours=t))


    plt.figure(figsize=(8, 4.8))
    axs = plt.axes()
    plt.plot(dates2, cum2, "-", color="blue", label="MatchTemplate + Histograms (thresh=0.71)")
    plt.plot(dates1, cum1, "-", color="red", label="MatchTemplate + MeanShift (mean)")
    plt.plot(dates_val, cum_val, ".", color="black", label="Validation Data")
    for label in axs.get_xticklabels():
        label.set_ha("right")
        label.set_rotation(18)
    # Format the date into months & days
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

    # Change the tick interval
    axs.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    axs.set_ylabel('Displacement [m]',fontsize='medium', fontweight= 'heavy')
    axs.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend()

    plt.show()
