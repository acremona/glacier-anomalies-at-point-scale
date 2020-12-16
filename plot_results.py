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
start_minutes = [datetime.datetime(2019, 6, 28, 5, 30)]
custom_start = False
custom_end = False

sheets = ["Sheet2"]
temp_date = start_date
delta_day = datetime.timedelta(days=7)
dx = 7

for i in range(len(sheets)):
    # import data from excel
    data = pd.read_excel("C:\\Users\\joelb\\Downloads\\comparison2.xlsx",sheet_name=sheets[i])
    start_min = start_minutes[i]

    times = []
    val_times = []
    val_values = []
    values = []
    dates_all = []
    val_dates_all = []

    for d in range(2):
        times.append(data['time'+str(d+1)].dropna().tolist())
        values.append(data['val' + str(d + 1)].dropna().tolist())
        val_times.append(data['val_time'+str(d+1)].dropna().tolist())
        val_values.append(data['val_val' + str(d + 1)].dropna().tolist())
        dates = []
        val_dates = []
        for t in times[d]:
            dates.append(start_min + datetime.timedelta(hours=t))
        for t in val_times[d]:
            val_dates.append(start_min + datetime.timedelta(hours=t))
        dates_all.append(dates)
        val_dates_all.append(val_dates)

    if custom_start:
        new_values_all = []
        for t in range(len(values)):
            startfound = False
            new_values = []
            for d in range(len(values[t])):
                if dates_all[t][d] >= custom_start and not startfound:
                    zero_value = values[t][d]
                    index = d
                    new_values.append(0)
                    startfound = True
                else:
                    if startfound and (not custom_end or dates_all[t][d] <= custom_end):
                        new_values.append(values[t][d]-zero_value)
                if custom_end and dates_all[t][d] > custom_end:
                    index2 = d
                    break
            new_values_all.append(new_values)
            if custom_end:
                del dates_all[t][index2:]
            del dates_all[t][:index]

    if custom_start:
        new_val_values_all = []
        for t in range(len(val_values)):
            startfound = False
            new_val_values = []
            for d in range(len(val_values[t])):
                if val_dates_all[t][d] >= custom_start and not startfound:
                    zero_value = -0.01
                    index = d
                    new_val_values.append(0)
                    startfound = True
                else:
                    if startfound and (not custom_end or val_dates_all[t][d] <= custom_end):
                        new_val_values.append(val_values[t][d]-zero_value)
                if custom_end and val_dates_all[t][d] > custom_end:
                    index2 = d
                    break
            new_val_values_all.append(new_val_values)
            if custom_end:
                del val_dates_all[t][index2:]
            del val_dates_all[t][:index]


    plt.figure(figsize=(8, 4.8))
    axs = plt.axes()

    if custom_start:
        plt.plot(dates_all[0], new_values_all[0], "-", color="red", label="1008")
        plt.plot(dates_all[1], new_values_all[1], "-", color="blue", label="1001")
        plt.plot(val_dates_all[0], new_val_values_all[0], ".", color="red", label="1008 validation")
        plt.plot(val_dates_all[1], new_val_values_all[1], ".", color="blue", label="1001 validation")
        # plt.plot(val_dates_all[2], new_val_values_all[2], ".", color="green", label="1007")
        # plt.plot(val_dates_all[3], new_val_values_all[3], ".", color="orange", label="1009")
    else:
        plt.plot(dates_all[0], values[0], "-", color="red", label="1008")
        plt.plot(val_dates_all[0], val_values[0], ".", color="red", label="1008 validation")
        plt.plot(dates_all[1], values[1], "-", color="blue", label="1001")
        plt.plot(val_dates_all[1], val_values[1], ".", color="blue", label="1001 validation")
    for label in axs.get_xticklabels():
        label.set_ha("right")
        label.set_rotation(18)
    # Format the date into months & days
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))

    # Change the tick interval
    axs.xaxis.set_major_locator(mdates.DayLocator(interval=dx))
    axs.set_ylabel('Displacement [m]',fontsize='medium', fontweight= 'heavy')
    axs.grid(color='gray', linestyle=':', linewidth=0.5)
    plt.legend()

    plt.show()
